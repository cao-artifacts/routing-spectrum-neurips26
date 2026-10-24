#!/usr/bin/env python3
"""
Batch driver for the Appendix H measurement matrix (labelled E1 to E3 below).

Enumerates the E1/E2/E3 runs, calls experiment.py once per cell in a fresh
subprocess (so a CUDA OOM or a bad checkpoint cannot poison the rest of the
sweep), supports resume (a cell whose JSON *and* .npz already exist is skipped)
and appends one JSON line per run to a manifest.

  E1  base       14 published models, c4, N=512, S=400, quant auto
                 (first 200 prompts = published calibration set, next 200 = held out)
  E2  grid       14 models x {c4, wikitext, code} x N in {256,512,1024}, S=200, auto
  E3  pairs      qwen3-14b, mistral-nemo-12b, qwen3-32b in BOTH fp16 and nf4
                 (c4, N=512, S=200)
  E3_70B         llama3-70b fp16 (needs 2x H100, device_map auto) + its nf4 twin,
                 same cell geometry, so the pair is self-contained on that host

Usage (from the repository root; outputs default to results/measure/):
    python src/measure/run_batch.py --only e1 --out_dir results/measure
    python src/measure/run_batch.py --only all --dry_run
    python src/measure/run_batch.py --only e2 --models qwen3-8b gemma2-9b
    python src/measure/run_batch.py --only e2 --models gemma2-2b gemma2-9b --quant_override bf16

HF_TOKEN is read from the environment and forwarded to the child process
through the environment only (never as a command-line argument).
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # repository root (src/measure/ -> root)
sys.path.insert(0, str(HERE))
from experiment import MODELS, TABLE1_MODELS, DATASETS, dump_stem, sanitize_key  # noqa: E402

# CPU-runnable harness check (no GPU, ~10 s): exercises the full pipeline on a
# 2-layer random Llama so a broken GPU environment is caught before GPU hours burn.
SMOKE_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"

# ── Matrix definition ─────────────────────────────────────────────
E1_SEQ_LEN = 512
E1_NUM_PROMPTS = 400          # 200 published + 200 held out
E2_SEQ_LENS = [256, 512, 1024]
E2_DATASETS = ["c4", "wikitext", "code"]
E2_NUM_PROMPTS = 200
E3_MODELS = ["qwen3-14b", "mistral-nemo-12b", "qwen3-32b"]
E3_QUANTS = ["fp16", "nf4"]
E3_SEQ_LEN = 512
E3_NUM_PROMPTS = 200
E3_70B_MODEL = "llama3-70b"

# wikitext-103 test only has ~322 docs > 1024 chars and 9 docs > 2048 chars, so
# the pool has to be scanned deeply and overflows test -> validation -> train.
POOL_SCAN_LIMIT = {"c4": None, "wikitext": 400000, "code": 400000}


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hf_cache_dir(hf_name):
    """Local HF hub folder for a repo id (models--org--name)."""
    root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    return Path(root) / ("models--" + hf_name.replace("/", "--"))


def purge_model_cache(hf_name):
    """Delete one model's folder (models--org--name) from the HF cache (the 14
    checkpoints total ~344 GB in fp16, which does not fit a 400 GB volume all at
    once). This removes every file of that model in the cache, including
    downloads made before this run and other revisions; point HF_HOME or
    HF_HUB_CACHE at a dedicated cache when using it."""
    import shutil
    d = hf_cache_dir(hf_name)
    if not d.exists():
        return 0
    freed = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    shutil.rmtree(d, ignore_errors=True)
    print(f"      purged HF cache {d} ({freed/1e9:.1f} GB freed)", flush=True)
    return freed


def make_run(phase, model, dataset, seq_len, num_prompts, quant, note="", model_id=None):
    return {
        "phase": phase,
        "model": model,
        "model_id": model_id,
        "dataset": dataset,
        "seq_len": seq_len,
        "num_prompts": num_prompts,
        "quant": quant,
        "note": note,
        # JSON name uses the *requested* quant tag; the .npz name uses the
        # *resolved* one (auto -> fp16/nf4), so resume globs for the npz.
        # Consequence: an E3 cell (explicit nf4) and the E2 cell of the same
        # geometry (auto -> nf4) share one npz name and overwrite each other
        # with byte-equivalent content; their JSONs stay separate.
        "stem": dump_stem(model, dataset, seq_len, num_prompts, quant),
        "npz_glob": dump_stem(model, dataset, seq_len, num_prompts, "*") + ".npz",
    }


def build_matrix(only, models_filter=None, skip_dup=False, smoke=False):
    runs = []

    if smoke:
        return [make_run("smoke", sanitize_key(SMOKE_MODEL_ID), "wikitext", 64, 3, "auto",
                         note="CPU harness check on a tiny random Llama",
                         model_id=SMOKE_MODEL_ID)]

    if only in ("e1", "all"):
        for m in TABLE1_MODELS:
            runs.append(make_run("e1", m, "c4", E1_SEQ_LEN, E1_NUM_PROMPTS, "auto",
                                 note="superset: first 200 = published set, next 200 = held out"))

    if only in ("e2", "all"):
        for m in TABLE1_MODELS:
            for ds in E2_DATASETS:
                for n in E2_SEQ_LENS:
                    dup = (ds == "c4" and n == 512)
                    if dup and skip_dup:
                        continue
                    runs.append(make_run(
                        "e2", m, ds, n, E2_NUM_PROMPTS, "auto",
                        note="duplicate of the published cell; kept for symmetry and as an "
                             "exact-reproduction check against E1's first 200 prompts" if dup else ""))

    if only in ("e3", "all"):
        for m in E3_MODELS:
            for q in E3_QUANTS:
                runs.append(make_run("e3", m, "c4", E3_SEQ_LEN, E3_NUM_PROMPTS, q,
                                     note="fp16 vs NF4 precision ablation"))

    if only in ("e3_70b", "all"):
        for q in ["fp16", "nf4"]:
            runs.append(make_run("e3_70b", E3_70B_MODEL, "c4", E3_SEQ_LEN, E3_NUM_PROMPTS, q,
                                 note="fp16 needs 2x H100 (device_map auto); nf4 twin is the "
                                      "matched pair partner (same cell as E2 c4/N512)"))

    if models_filter:
        unknown = [m for m in models_filter if m not in MODELS]
        if unknown:
            raise SystemExit(f"unknown model keys: {unknown}\nknown: {sorted(MODELS)}")
        keep = set(models_filter)
        runs = [r for r in runs if r["model"] in keep]
    return runs


assert all(d in DATASETS for d in E2_DATASETS), "E2 datasets must exist in experiment.DATASETS"


# ── Execution ─────────────────────────────────────────────────────
def already_done(run, json_dir, npz_dir, want_dump=True):
    json_path = json_dir / (run["stem"] + ".json")
    if not json_path.exists():
        return False, None, []
    npz_paths = sorted(glob.glob(str(npz_dir / run["npz_glob"])))
    if want_dump and not npz_paths:
        return False, str(json_path), []
    # a truncated JSON from a killed run must not count as done
    try:
        with open(json_path) as f:
            payload = json.load(f)
        if not payload:
            return False, str(json_path), npz_paths
    except Exception:
        return False, str(json_path), npz_paths
    return True, str(json_path), npz_paths


def build_cmd(run, args, json_dir, npz_dir, prompt_dir=None):
    cmd = [args.python, str(Path(args.experiment).resolve())]
    if run.get("model_id"):
        cmd += ["--model_id", run["model_id"]]
    else:
        cmd += ["--models", run["model"]]
    cmd += [
        "--dataset", run["dataset"],
        "--seq_len", str(run["seq_len"]),
        "--num_prompts", str(run["num_prompts"]),
        "--quant", run["quant"],
        "--seed", str(args.seed),
        "--device_map", args.device_map,
        "--output", str(json_dir / (run["stem"] + ".json")),
    ]
    if not args.no_dump:
        cmd += ["--dump_grams", str(npz_dir)]
    if prompt_dir is not None:
        cmd += ["--prompt_cache", str(prompt_dir)]
    scan = POOL_SCAN_LIMIT.get(run["dataset"])
    if scan:
        cmd += ["--pool_scan_limit", str(scan)]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all", choices=["e1", "e2", "e3", "e3_70b", "all"])
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset filter on model keys.")
    parser.add_argument("--out_dir", default=str(ROOT / "results" / "measure"))
    parser.add_argument("--experiment", default=str(HERE / "experiment.py"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--quant_override", default=None,
                        choices=["fp16", "bf16", "nf4", "fp32"],
                        help="Force the precision for every selected run (e.g. re-running "
                             "Gemma-2 cells in bf16 after an fp16 overflow).")
    parser.add_argument("--skip_dup", action="store_true",
                        help="Drop the E2 c4/N=512 cells that duplicate the published protocol.")
    parser.add_argument("--no_dump", action="store_true", help="Do not write .npz Gram dumps.")
    parser.add_argument("--no_prompt_cache", action="store_true",
                        help="Re-select the calibration pool from scratch for every run.")
    parser.add_argument("--force", action="store_true", help="Re-run cells that already have output.")
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None, help="Per-run timeout in seconds.")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--purge_hf_cache", action="store_true",
                        help="Delete a checkpoint's folder from the HF cache once its last run "
                             "in this invocation finishes (keeps a 400 GB volume from filling). "
                             "WARNING: this deletes every cached file of that model, including "
                             "earlier downloads and other revisions; use it with a dedicated "
                             "HF_HOME or HF_HUB_CACHE.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help=f"Ignore the matrix and run one CPU harness check on {SMOKE_MODEL_ID}.")
    args = parser.parse_args()

    runs = build_matrix(args.only, args.models, skip_dup=args.skip_dup, smoke=args.smoke)
    if args.quant_override:
        for r in runs:
            r["quant"] = args.quant_override
            r["stem"] = dump_stem(r["model"], r["dataset"], r["seq_len"], r["num_prompts"],
                                  args.quant_override)
            r["npz_glob"] = r["stem"] + ".npz"

    out_dir = Path(args.out_dir).resolve()
    json_dir = out_dir / "json"
    npz_dir = out_dir / "npz"
    log_dir = out_dir / "logs"
    prompt_dir = None if args.no_prompt_cache else out_dir / "prompts"
    manifest = out_dir / "manifest.jsonl"

    # ── dry run: print the matrix ──
    if args.dry_run:
        print(f"Matrix for --only {args.only}"
              + (f" --models {' '.join(args.models)}" if args.models else "")
              + (f" --quant_override {args.quant_override}" if args.quant_override else ""))
        print(f"out_dir = {out_dir}")
        print("-" * 100)
        hdr = f"{'#':>4}  {'phase':<7} {'model':<18} {'data':<9} {'N':>5} {'S':>5} {'quant':<6}  stem"
        print(hdr)
        print("-" * 100)
        per_phase, per_model, per_dataset = {}, {}, {}
        for i, r in enumerate(runs, 1):
            print(f"{i:>4}  {r['phase']:<7} {r['model']:<18} {r['dataset']:<9} "
                  f"{r['seq_len']:>5} {r['num_prompts']:>5} {r['quant']:<6}  {r['stem']}")
            per_phase[r["phase"]] = per_phase.get(r["phase"], 0) + 1
            per_model[r["model"]] = per_model.get(r["model"], 0) + 1
            per_dataset[r["dataset"]] = per_dataset.get(r["dataset"], 0) + 1
        print("-" * 100)
        print(f"TOTAL RUNS: {len(runs)}")
        print("per phase   : " + json.dumps(per_phase))
        print("per dataset : " + json.dumps(per_dataset))
        print("per model   : " + json.dumps(per_model))
        forward_passes = sum(r["num_prompts"] for r in runs)
        print(f"total forward passes: {forward_passes}")
        print("resume: a cell is skipped when json/<stem>.json and npz/<glob> both exist")
        print("example command for run #1:")
        print("  " + " ".join(build_cmd(runs[0], args, json_dir, npz_dir, prompt_dir))
              if runs else "  (none)")
        return

    for d in (json_dir, npz_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    if prompt_dir is not None:
        prompt_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()  # HF_TOKEN is inherited, never placed on the command line
    if not env.get("HF_TOKEN"):
        print("NOTE: HF_TOKEN is not set; gated checkpoints (Llama 3, Llama 3.2, Gemma 2) will fail.")
    if any(r["phase"] == "e3_70b" and r["quant"] == "fp16" for r in runs):
        try:
            import torch
            ngpu = torch.cuda.device_count()
        except Exception:
            ngpu = 0
        if ngpu < 2:
            print(f"NOTE: llama3-70b fp16 needs >= 2 GPUs (141 GB of weights); "
                  f"this host reports {ngpu}. Run that phase on a 2x H100 host.")

    n_done = n_skipped = n_failed = 0
    t_start = time.time()
    executed = 0
    # index of the last run of each model, for --purge_hf_cache
    last_run_of_model = {r["model"]: i for i, r in enumerate(runs, 1)}

    for i, run in enumerate(runs, 1):
        if args.max_runs is not None and executed >= args.max_runs:
            print(f"--max_runs {args.max_runs} reached; stopping.")
            break

        done, jpath, npaths = already_done(run, json_dir, npz_dir, want_dump=not args.no_dump)
        if done and not args.force:
            print(f"[{i}/{len(runs)}] SKIP (done): {run['stem']}")
            n_skipped += 1
            continue

        cmd = build_cmd(run, args, json_dir, npz_dir, prompt_dir)
        log_path = log_dir / (run["stem"] + ".log")
        print(f"[{i}/{len(runs)}] RUN {run['stem']}  ->  {log_path}")
        print("      " + " ".join(cmd), flush=True)

        started = utcnow()
        t0 = time.time()
        status, rc = "ok", 0
        try:
            with open(log_path, "w") as lf:
                lf.write(" ".join(cmd) + "\n\n")
                lf.flush()
                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                      env=env, timeout=args.timeout)
            rc = proc.returncode
            status = "ok" if rc == 0 else "failed"
        except subprocess.TimeoutExpired:
            status, rc = "timeout", -1
        except KeyboardInterrupt:
            status, rc = "interrupted", -2
            raise
        finally:
            wall = time.time() - t0
            executed += 1

            record = dict(run)
            record.update({
                "cmd": cmd,
                "started_at": started,
                "finished_at": utcnow(),
                "wall_seconds": round(wall, 1),
                "status": status,
                "returncode": rc,
                "log": str(log_path),
                "json": str(json_dir / (run["stem"] + ".json")),
                "npz": sorted(glob.glob(str(npz_dir / run["npz_glob"]))),
            })
            # pull a few headline numbers out of the JSON for the manifest
            try:
                with open(json_dir / (run["stem"] + ".json")) as f:
                    payload = json.load(f)
                res = payload.get(run["model"], {})
                layers = res.get("layers", {})
                reffs = sorted(v["r_eff"] for v in layers.values())
                record.update({
                    "quant_used": res.get("quant_used"),
                    "revision": res.get("revision"),
                    "num_layers": res.get("num_layers"),
                    "num_prompts_actual": res.get("num_prompts"),
                    "any_corrupted": res.get("any_corrupted"),
                    "corrupted_layers": res.get("corrupted_layers"),
                    "r_eff_median": float(reffs[len(reffs) // 2]) if reffs else None,
                    "dump_size_mb": res.get("dump_size_mb"),
                })
            except Exception as e:
                record["manifest_parse_error"] = f"{type(e).__name__}: {e}"

            with open(manifest, "a") as mf:
                mf.write(json.dumps(record) + "\n")

        if status == "ok":
            n_done += 1
            flag = " [CORRUPTED LAYERS]" if record.get("any_corrupted") else ""
            print(f"      done in {wall/60:.1f} min  r_eff_median={record.get('r_eff_median')}"
                  f"  quant_used={record.get('quant_used')}{flag}", flush=True)
        else:
            n_failed += 1
            print(f"      {status.upper()} (rc={rc}) after {wall/60:.1f} min; see {log_path}", flush=True)
            if args.stop_on_error:
                break

        if args.purge_hf_cache and last_run_of_model.get(run["model"]) == i:
            hf_name = MODELS.get(run["model"], {}).get("hf_name") or run.get("model_id")
            if hf_name:
                purge_model_cache(hf_name)

    print("\n" + "=" * 60)
    print(f"phase={'smoke' if args.smoke else args.only}  ran={n_done}  skipped={n_skipped}  "
          f"failed={n_failed}  "
          f"total_wall={(time.time() - t_start)/3600:.2f} h")
    print(f"manifest: {manifest}")
    print("=" * 60)


if __name__ == "__main__":
    main()
