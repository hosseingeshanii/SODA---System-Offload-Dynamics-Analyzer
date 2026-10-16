#!/usr/bin/env python3
"""
PLANNING stage only -- this script writes configs + PENDING rows in
job_manifest.csv. It does NOT run anything on the cluster. Generating
configs is nearly instant, so one person can run this for the whole sweep
in seconds; actually executing each case (slow: hours of queue time) is a
separate step, done by run_next_pending.py, which anyone on the team can
run independently and repeatedly to pick up whatever's not done yet.

Two ways to use it:

  1. One-off, single config:
       python generate_sweep_config.py --model llama3p2_1b --mode decode \
           --isl 8192 --osl 1024 --batch_size 128

  2. A whole sweep from a spec file (see sweep_spec_example.yaml):
       python generate_sweep_config.py --spec sweep_spec_example.yaml

Either way, before writing anything, it checks job_manifest.csv for the
exact (model, mode, isl, osl, batch_size) tuple:
  - already SUCCESS, or already has a PENDING/CLAIMED/RUNNING row -> skip,
    don't write a duplicate config or manifest row
  - otherwise                   -> write a PENDING row (file-locked, so two
                                    people generating the same spec at the
                                    same time don't double-write)
then writes the YAML config, auto-creating the dummy input data for that
ISL if it doesn't exist yet.

NOTE on --osl for prefill: prefill measures prompt-processing only, so
output_tokens is forced to 1 regardless of --osl (a warning is printed if
you passed something else) -- this is a real modeling decision, not
obviously "correct," and is flagged in the team's checklist as something
to confirm at the meeting, not something this script silently decides for
good.
"""
import argparse
import csv
import os
import fcntl
import itertools
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(os.environ.get("SWEEP_ROOT", Path(__file__).resolve().parent.parent))
CONFIG_OUT_DIR = ROOT / "configs"
MANIFEST_PATH = ROOT / "job_manifest.csv"
MANIFEST_COLUMNS = ["tag", "status", "elapsed_s", "job_ids", "compile_hashes", "source_log"]

WARMUP_STEPS = 5
MEASURED_STEPS = 2

# Same architecture blocks as full_sweep_config_generator.py, kept in sync by
# hand for now -- copy any future edits there into here too (or better,
# factor into one shared module; not done yet to avoid touching the
# existing, working sweep while this is new).
MODELS = {
    "gpt2": {
        "name": "gpt2", "vocab_size": 50257, "hidden_size": 768, "num_hidden_layers": 12,
        "num_heads": 12, "filter_size": 3072, "nonlinearity": "gelu", "norm_type": "layernorm",
        "layer_norm_epsilon": 1.0e-05, "position_embedding_type": "learned",
        "share_embedding_weights": True,
    },
    "llama3p2_1b": {
        "name": "llama", "vocab_size": 128256, "hidden_size": 2048, "num_hidden_layers": 16,
        "num_heads": 32, "extra_attention_params": {"num_kv_groups": 8},
        "attention_module": "multiquery_attention", "filter_size": 8192, "nonlinearity": "swiglu",
        "norm_type": "rmsnorm", "layer_norm_epsilon": 1.0e-05, "position_embedding_type": "rotary",
        "rotary_dim": 64, "rope_theta": 500000.0, "pos_scaling_type": "llama3",
        "pos_scaling_factor": 32.0,
        "pos_scaling_extra_args": {"high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192},
        "share_embedding_weights": True,
    },
    "llama3p2_3b": {
        "name": "llama", "vocab_size": 128256, "hidden_size": 3072, "num_hidden_layers": 28,
        "num_heads": 24, "extra_attention_params": {"num_kv_groups": 8},
        "attention_module": "multiquery_attention", "filter_size": 8192, "nonlinearity": "swiglu",
        "norm_type": "rmsnorm", "layer_norm_epsilon": 1.0e-05, "position_embedding_type": "rotary",
        "rotary_dim": 128, "rope_theta": 500000.0, "pos_scaling_type": "llama3",
        "pos_scaling_factor": 32.0,
        "pos_scaling_extra_args": {"high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192},
        "share_embedding_weights": True,
    },
    "olmoe_1b7b": {
        "name": "gpt2", "vocab_size": 50304, "hidden_size": 2048, "num_hidden_layers": 16,
        "num_heads": 16, "filter_size": 1024, "nonlinearity": "swiglu", "norm_type": "rmsnorm",
        "layer_norm_epsilon": 1.0e-05, "position_embedding_type": "rotary", "rotary_dim": 128,
        "rope_theta": 10000.0, "share_embedding_weights": False,
        "moe": {"num_experts": 64, "top_k": 8, "load_balancing_loss_coef": 0.01, "routing_algorithm": "learned"},
    },
    "qwen1p5_moe_a2p7b": {
        "name": "gpt2", "vocab_size": 151936, "hidden_size": 2048, "num_hidden_layers": 24,
        "num_heads": 16, "filter_size": 1408, "nonlinearity": "swiglu", "norm_type": "rmsnorm",
        "layer_norm_epsilon": 1.0e-06, "position_embedding_type": "rotary", "rotary_dim": 128,
        "rope_theta": 1000000.0, "share_embedding_weights": False,
        "moe": {"num_experts": 60, "num_shared_experts": 1, "top_k": 4, "load_balancing_loss_coef": 0.001, "routing_algorithm": "learned"},
    },
    # Placeholders only -- real HF architecture dims not pulled in yet.
    # Selecting these raises NotImplementedError rather than silently
    # running with made-up numbers.
    "llama3p1_8b": None,
    "llama2_70b": None,
}


def ensure_dummy_data(isl: int) -> Path:
    out_dir = ROOT / f"dummy_data_msl{isl}"
    out_path = out_dir / "data-00000.h5"
    if out_path.exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.ones((1000, 3, isl), dtype=np.int32)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=data)
    print(f"  (created missing dummy data: {out_path}, shape={data.shape})")
    return out_dir


def build_config(model_key: str, mode_key: str, isl: int, osl: int, batch_size: int) -> dict:
    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Known: {list(MODELS)}")
    if MODELS[model_key] is None:
        raise NotImplementedError(
            f"'{model_key}' has no real architecture spec yet -- needs HF config "
            f"dims added to MODELS before it can be used. Not fabricating numbers."
        )
    m = dict(MODELS[model_key])
    moe = m.pop("moe", None)

    output_tokens = osl
    if mode_key == "prefill" and osl != 1:
        print(f"  WARNING: mode=prefill forces output_tokens=1 (prefill measures "
              f"prompt processing only); ignoring --osl={osl} for this case.")
        output_tokens = 1

    model_block = {
        **m,
        "start_token": m["vocab_size"],
        "max_tokens": output_tokens,
        "max_position_embeddings": isl + output_tokens,
        "fp16_type": "cbfloat16",
    }
    if moe:
        model_block["moe"] = moe

    dummy_data_dir = ensure_dummy_data(isl)

    return {
        "soda_benchmark": {
            "mode": mode_key,
            "prompt_tokens": isl,
            "output_tokens": output_tokens,
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "save_final_prediction": True,
        },
        "model": model_block,
        "runconfig": {
            "precision_opt_level": 1,
            "log_steps": 1,
            "inference_steps": WARMUP_STEPS + MEASURED_STEPS,
        },
        "inference_input": {
            "data_processor": "GptHDF5MapDataProcessor",
            "data_dir": str(dummy_data_dir),
            "batch_size": batch_size,
            "micro_batch_size": batch_size,
            "drop_last": False,
        },
        "optimizer": {"loss_scaling_factor": 1.0, "log_summaries": False},
    }


def make_tag(model_key: str, mode_key: str, isl: int, osl: int, batch_size: int) -> str:
    return f"{model_key}_{mode_key}_isl{isl}_osl{osl}_bs{batch_size}"


def _read_manifest_rows() -> list:
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, newline="") as f:
        return list(csv.DictReader(f))


ACTIVE_STATUS_PREFIXES = ("PENDING", "CLAIMED_BY:", "RUNNING_BY:")


def register_pending(tag: str, force: bool) -> bool:
    """Returns True if the caller should proceed to write this config.
    Locks the manifest file for the whole read-check-write sequence so two
    concurrent invocations (e.g. two team members generating the same spec
    at once) don't double-register the same tag. A tag that already has a
    FAILURE row is a deliberate retry: that row is reset to PENDING *in
    place* rather than appending a second row with the same tag (a real
    bug this once produced -- duplicate rows for the same case, which then
    confuses anything that assumes one row per tag)."""
    MANIFEST_PATH.touch(exist_ok=True)
    with open(MANIFEST_PATH, "r+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            rows = list(csv.DictReader(f))

            existing = [r for r in rows if r.get("tag") == tag]
            for r in existing:
                status = r.get("status", "")
                if status == "SUCCESS":
                    print(f"SKIP: '{tag}' already has a SUCCESS result.")
                    return False
                if any(status.startswith(p) for p in ACTIVE_STATUS_PREFIXES) and not force:
                    print(f"SKIP: '{tag}' already has a config (status={status}).")
                    return False

            if existing:
                # Retry: reset the one existing row in place, don't duplicate it.
                for r in rows:
                    if r["tag"] == tag:
                        r.update({"status": "PENDING", "elapsed_s": "", "job_ids": "",
                                  "compile_hashes": "", "source_log": "generate_sweep_config.py (retry)"})
                f.seek(0); f.truncate()
                writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
                return True

            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            if not rows:  # file was empty -- need a header before the first row
                writer.writeheader()
            writer.writerow({
                "tag": tag,
                "status": "PENDING",
                "elapsed_s": "", "job_ids": "", "compile_hashes": "",
                "source_log": "generate_sweep_config.py",
            })
            f.flush()
            return True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def generate_one(model_key: str, mode_key: str, isl: int, osl: int, batch_size: int, force: bool = False):
    tag = make_tag(model_key, mode_key, isl, osl, batch_size)
    if not register_pending(tag, force):
        return None
    cfg = build_config(model_key, mode_key, isl, osl, batch_size)
    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONFIG_OUT_DIR / f"{tag}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"PENDING + wrote {out_path}")
    return out_path


def generate_from_spec(spec_path: Path, force: bool = False):
    spec = yaml.safe_load(open(spec_path))
    written = []
    for entry in spec.get("sweeps", []):
        name = entry.get("name", "unnamed")
        print(f"=== sweep: {name} ===")
        grid = itertools.product(
            entry["models"], entry["modes"], entry["isl"], entry["osl"], entry["batch_size"],
        )
        for model_key, mode_key, isl, osl, batch_size in grid:
            path = generate_one(model_key, mode_key, isl, osl, batch_size, force=force)
            if path:
                written.append(path)
    print(f"\nTotal new configs written: {len(written)}")
    return written


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(MODELS))
    p.add_argument("--mode", choices=["prefill", "decode"])
    p.add_argument("--isl", type=int)
    p.add_argument("--osl", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--spec", type=Path, help="Path to a sweep spec YAML (see sweep_spec_example.yaml)")
    p.add_argument("--force", action="store_true", help="Write a new PENDING config even if one already exists for this tag")
    args = p.parse_args()

    if args.spec:
        generate_from_spec(args.spec, force=args.force)
    elif args.model and args.mode and args.isl and args.osl and args.batch_size:
        generate_one(args.model, args.mode, args.isl, args.osl, args.batch_size, force=args.force)
    else:
        p.error("Either --spec, or all of --model/--mode/--isl/--osl/--batch_size, are required.")
