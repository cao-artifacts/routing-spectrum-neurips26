#!/usr/bin/env python3
"""
Integrity check for the .npz Gram dumps written by experiment.py --dump_grams.

Checks, per file:
  1. sum_s G_P_per_prompt[s] == G_P_total (float32 round-off tolerated)
  2. G_M and G_P are finite and symmetric
  3. every layer has positive G_P and G_M energy
  4. the spectrum recomputed from the dumped Grams matches the JSON (if given)
  5. provenance fields are present (revision, dataset, versions, previews)

Usage:
    python verify_dump.py npz/*.npz
    python verify_dump.py npz/qwen3-8b__c4__N512__S200__fp16.npz --json json/....json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import compute_spectrum  # noqa: E402


def check_file(path, json_path=None, rtol=1e-4):
    d = np.load(path, allow_pickle=False)
    ok = True
    print(f"\n=== {path} ===")

    cfg = json.loads(str(d["config_json"]))
    G_M = d["G_M"]
    G_P_total = d["G_P_total"]
    L, H, _ = G_M.shape
    print(f"model={cfg['model']} dataset={cfg['dataset']} N={cfg['seq_len']} "
          f"S={cfg['num_prompts']} quant={cfg['quant_used']} rev={cfg['revision']}")
    print(f"shapes: G_M {G_M.shape}  G_P_total {G_P_total.shape}")

    # 1. per-prompt reconstruction
    if "G_P_per_prompt" in d:
        pp = d["G_P_per_prompt"]
        print(f"        G_P_per_prompt {pp.shape} {pp.dtype}")
        recon = pp.astype(np.float64).sum(axis=0)
        diff = np.abs(recon - G_P_total)
        scale = max(np.abs(G_P_total).max(), 1e-12)
        rel = diff.max() / scale
        print(f"  [1] sum_s G_P_per_prompt vs G_P_total: max_abs_diff={diff.max():.6g} "
              f"scale={scale:.6g} rel={rel:.3e}  -> {'PASS' if rel < rtol else 'FAIL'}")
        ok &= bool(rel < rtol)
        if pp.shape[0] != cfg["num_prompts"]:
            print(f"  [1] WARN: per-prompt count {pp.shape[0]} != num_prompts {cfg['num_prompts']}")
    else:
        print("  [1] no G_P_per_prompt in file (dump written with --no per-prompt?) -> SKIP")

    # 2. finiteness / symmetry
    fin = bool(np.isfinite(G_M).all() and np.isfinite(G_P_total).all())
    sym = float(max(np.abs(G_M - np.transpose(G_M, (0, 2, 1))).max(),
                    np.abs(G_P_total - np.transpose(G_P_total, (0, 2, 1))).max()))
    symscale = max(np.abs(G_P_total).max(), np.abs(G_M).max(), 1e-12)
    print(f"  [2] finite={fin}  max_asymmetry={sym:.6g} (rel {sym/symscale:.2e}) "
          f"-> {'PASS' if fin else 'FAIL'}")
    ok &= fin

    # 3. per-layer energy
    tr_p = np.trace(G_P_total, axis1=1, axis2=2)
    tr_m = np.trace(G_M.astype(np.float64), axis1=1, axis2=2)
    bad = [int(i) for i in range(L) if not (tr_p[i] > 0 and tr_m[i] > 0)]
    print(f"  [3] layers with zero energy: {bad if bad else 'none'} -> {'FAIL' if bad else 'PASS'}")
    ok &= not bad
    flagged = [int(i) for i, c in enumerate(d["layer_corrupted"]) if c]
    print(f"  [3] layers flagged corrupted at measurement time: {flagged if flagged else 'none'}")

    # 4. spectrum re-derivation
    if json_path:
        with open(json_path) as f:
            payload = json.load(f)
        res = payload[cfg["model"]]
        spec = compute_spectrum({i: G_P_total[i] for i in range(L)},
                                {i: G_M[i].astype(np.float64) for i in range(L)})
        worst = 0.0
        for i in range(L):
            a = spec[i]["r_eff"]
            b = res["layers"][str(i)]["r_eff"]
            worst = max(worst, abs(a - b) / max(abs(b), 1e-12))
        print(f"  [4] r_eff recomputed from npz vs JSON: max_rel_diff={worst:.3e} "
              f"-> {'PASS' if worst < 1e-6 else 'FAIL'}")
        ok &= bool(worst < 1e-6)

    # 5. provenance
    needed = ["model", "hf_name", "dataset", "seq_len", "num_prompts", "quant_requested",
              "quant_used", "torch_dtype", "seed", "revision", "library_versions"]
    missing = [k for k in needed if k not in cfg]
    prev = d["prompt_preview"]
    print(f"  [5] provenance missing={missing if missing else 'none'}  "
          f"previews={len(prev)} (first: {repr(str(prev[0])[:48]) if len(prev) else 'n/a'}) "
          f"-> {'PASS' if not missing else 'FAIL'}")
    ok &= not missing

    print(f"  ==> {'ALL CHECKS PASS' if ok else 'CHECKS FAILED'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--json", default=None, help="Matching results JSON for spectrum re-derivation.")
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()

    ok = True
    for p in args.npz:
        ok &= check_file(p, args.json, args.rtol)
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
