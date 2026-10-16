#!/usr/bin/env python3
"""
EXECUTION stage -- the "worker" half of the planning/execution split.
generate_sweep_config.py writes configs + PENDING rows (fast, no cluster
cost); this script is what anyone on the team runs to actually submit one
or more cases to the cluster (slow -- hours of queue time per case on this
shared system).

Usage:
    python run_next_pending.py                 # claim + run the first PENDING case
    python run_next_pending.py --loop           # keep doing that until none are left
    python run_next_pending.py --parallel 4      # claim + run 4 cases concurrently
    python run_next_pending.py --parallel 4 --loop   # 4 concurrent workers, each looping

What it does, each time:
  1. Locks job_manifest.csv, finds the first row with status=PENDING.
  2. Double-checks results/<tag>/soda_output/performance.json doesn't
     already show benchmark_valid=true (safety net in case the manifest
     drifted from reality -- don't trust the CSV blindly).
  3. Marks it RUNNING_BY:<user>, unlocks, then actually runs soda_inference.py
     with the matching config from configs/<tag>.yaml, capturing its output
     to logs/<tag>.log and parsing it for real wsjob/imgjob IDs and compile
     hashes (same regex convention as the old build_job_manifest.py).
  4. Updates the row to SUCCESS or FAILURE(exit=N) with elapsed seconds and
     the real job_ids/compile_hashes, same convention as the rest of
     job_manifest.csv.

--parallel N runs N claim+run workers concurrently (threads -- each spends
nearly all its time blocked in the cluster subprocess call, which releases
the GIL, so this is safe and genuinely concurrent). The file lock on
job_manifest.csv means two workers can never claim the same case. Since
compile and execute are separate pools on this cluster, running several
compiles in parallel is a real throughput win even though only one case
can actually execute on the wafer at a time.

Needs PATH and PYTHONPATH set to the Cerebras venv + SODA/src before running
(same two env vars that caused real failures earlier when set manually and
incorrectly -- this script sets them itself so that mistake can't recur).
"""
import argparse
import csv
import fcntl
import getpass
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("SWEEP_ROOT", Path(__file__).resolve().parent.parent))
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
LOG_DIR = ROOT / "logs"
MANIFEST_PATH = ROOT / "job_manifest.csv"
MANIFEST_COLUMNS = ["tag", "status", "elapsed_s", "job_ids", "compile_hashes", "source_log"]

# Same convention as the old build_job_manifest.py, so job IDs recorded by
# this script and by the earlier bash-based runs look identical in the CSV.
JOBID_RE = re.compile(r"Job id: (wsjob-[\w]+|imgjob-[\w]+)")
COMPILE_HASH_RE = re.compile(r"Compile hash is: (cs_\w+)")

CEREBRAS_VENV = Path(os.environ.get("SODA_CEREBRAS_VENV", "/cra-1684/tutorial/pvellaisamy/venvs/r2.10"))
SODA_SRC = Path(os.environ.get("SODA_SRC", Path(__file__).resolve().parent.parent / "src"))
SODA_SCRIPT = SODA_SRC / "soda" / "cerebras" / "soda_inference.py"


def _run_env():
    env = os.environ.copy()
    env["PATH"] = f"{CEREBRAS_VENV / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{SODA_SRC}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return env


def already_has_valid_result(tag: str) -> bool:
    perf_path = RESULTS_DIR / tag / "soda_output" / "performance.json"
    if not perf_path.exists():
        return False
    try:
        return json.loads(perf_path.read_text()).get("benchmark_valid") is True
    except Exception:
        return False


def _rewrite_manifest(rows):
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def claim_next_pending():
    """Locks the manifest, finds+claims the first PENDING row, returns its
    tag (or None if nothing's pending). Written back as RUNNING_BY:<user>
    before the lock releases, so two workers running this at the same
    moment can't both grab the same case."""
    MANIFEST_PATH.touch(exist_ok=True)
    with open(MANIFEST_PATH, "r+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            rows = list(csv.DictReader(f))
            for r in rows:
                if r.get("status") == "PENDING":
                    tag = r["tag"]
                    if already_has_valid_result(tag):
                        r["status"] = "SUCCESS"
                        print(f"'{tag}' was PENDING but already has a valid result on disk -- marking SUCCESS, not re-running.")
                        _rewrite_manifest(rows)
                        continue
                    r["status"] = f"RUNNING_BY:{getpass.getuser()}"
                    _rewrite_manifest(rows)
                    return tag
            return None
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def update_status(tag: str, status: str, elapsed_s: float, job_ids=(), compile_hashes=(), log_name=None):
    with open(MANIFEST_PATH, "r+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            rows = list(csv.DictReader(f))
            for r in rows:
                if r["tag"] == tag:
                    r["status"] = status
                    r["elapsed_s"] = f"{elapsed_s:.1f}"
                    if job_ids:
                        r["job_ids"] = ";".join(sorted(set(job_ids)))
                    if compile_hashes:
                        r["compile_hashes"] = ";".join(sorted(set(compile_hashes)))
                    r["source_log"] = f"run_next_pending.py ({log_name})" if log_name else "run_next_pending.py"
            _rewrite_manifest(rows)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run_case(tag: str) -> bool:
    cfg_path = CONFIG_DIR / f"{tag}.yaml"
    if not cfg_path.exists():
        print(f"ERROR: claimed '{tag}' but {cfg_path} doesn't exist -- marking FAILURE.")
        update_status(tag, "FAILURE(config_missing)", 0.0)
        return False

    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{tag}.log"

    print(f"=== Running {tag} === (log: {log_path})")
    job_ids, compile_hashes = [], []
    start = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            [str(CEREBRAS_VENV / "bin" / "python"), str(SODA_SCRIPT), "CSX",
             "--params", str(cfg_path), "--disable_version_check", "-o", str(out_dir)],
            env=_run_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in proc.stdout:
            logf.write(line)
            m = JOBID_RE.search(line)
            if m:
                job_ids.append(m.group(1))
            m = COMPILE_HASH_RE.search(line)
            if m:
                compile_hashes.append(m.group(1))
        proc.wait()
    elapsed = time.time() - start

    if proc.returncode == 0 and already_has_valid_result(tag):
        update_status(tag, "SUCCESS", elapsed, job_ids, compile_hashes, log_path.name)
        print(f"SUCCESS: {tag} after {elapsed:.0f}s (job_ids={job_ids})")
        return True
    else:
        update_status(tag, f"FAILURE(exit={proc.returncode})", elapsed, job_ids, compile_hashes, log_path.name)
        print(f"FAILURE: {tag} exit={proc.returncode} after {elapsed:.0f}s")
        return False


def worker_loop(loop: bool):
    while True:
        tag = claim_next_pending()
        if tag is None:
            print(f"[{getpass.getuser()}] No PENDING cases left.")
            return
        run_case(tag)
        if not loop:
            return


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--loop", action="store_true", help="Keep claiming+running cases until none are PENDING")
    p.add_argument("--parallel", type=int, default=1, help="Run N claim+run workers concurrently (default 1)")
    args = p.parse_args()

    if args.parallel > 1:
        print(f"Starting {args.parallel} parallel workers...")
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = [ex.submit(worker_loop, args.loop) for _ in range(args.parallel)]
            for fut in futures:
                fut.result()  # surface any worker exception instead of swallowing it
    else:
        worker_loop(args.loop)
