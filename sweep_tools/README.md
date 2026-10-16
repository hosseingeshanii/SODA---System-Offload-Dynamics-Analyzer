# Sweep Tools

Two scripts for running cases from the sweep plan, usable from any environment with Cerebras SDK/cstorch access --- not just the shared cluster directory. Anyone on the team can use this, not just whoever it was handed to first.

## 1. One-time setup

```bash
git clone <this fork> soda_sweep
cd soda_sweep
git checkout sweep-tools
cd sweep_tools
```

Set these three environment variables to match your own setup (defaults assume Hossein's paths, which you won't have access to):

```bash
export SWEEP_ROOT=/path/to/wherever/you/want/configs+results+job_manifest.csv
export SODA_SRC=/path/to/your/SODA/src          # the SODA clone with the fix applied (see step 2)
export SODA_CEREBRAS_VENV=/path/to/your/cerebras/venv
```

`SWEEP_ROOT` is just a working directory for this kit — it'll create `configs/`, `results/`, and `job_manifest.csv` there automatically. It does **not** need to be shared with anyone; this is your own local tracking, separate from Hossein's.

## 2. Apply the fix first

This branch (`sweep-tools`) already includes the Trainer-context fix to `src/soda/cerebras/soda_inference.py` — the bug that was crashing every MoE run in under a minute. If you're using a different SODA clone, apply the same fix there (see the commit on this branch for the exact diff).

## 3. Generate whatever's assigned to you

Check the shared spreadsheet for your current assignment, then generate each case:

```bash
python generate_sweep_config.py --model <model> --mode <prefill|decode> --isl <N> --osl <N> --batch_size <N>
```

Example — Qwen1.5-MoE-A2.7B, prefill, context length 1024:
```bash
python generate_sweep_config.py --model qwen1p5_moe_a2p7b --mode prefill --isl 1024 --osl 1 --batch_size 1
```

Or generate a whole batch at once from a spec file — see `sweep_spec_example.yaml` for the format.

Each case is instant to generate — just writes a config and a `PENDING` row in your local `job_manifest.csv`.

Then run them (this part is slow — hours of cluster queue time per case):

```bash
python run_next_pending.py --loop
```

It'll work through everything `PENDING` automatically, one at a time, recording `SUCCESS` or `FAILURE` with the real job IDs for each. You don't need to babysit it — leave it running and check back.

## 4. When you're done

Let Hossein know your results — he'll update the shared spreadsheet. No need to touch anything on his end directly.
