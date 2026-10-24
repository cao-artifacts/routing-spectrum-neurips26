#!/usr/bin/env python3
"""
E4a -> E4b bridge: turn a Gram dump into phi JSON files that route_patch_ppl.py
can apply.

`analyze_merge.py` reports merge *error* per method and rank, and records the
representative sets, but it does not emit the phi maps themselves.
This script recomputes the same maps with `analyze_merge`'s own constructors
(so the numbers and the applied map cannot drift apart) and writes them in the
schema documented in PHI_SCHEMA.md.

    python src/analysis/export_phi.py \
        --npz results/measure/npz/qwen3-8b__c4__N512__S400__fp16.npz \
        --methods greedy eigen similarity random --ranks 4 8 16 --sel_range 0:200 \
        --model Qwen/Qwen3-8B --out_dir results/patching/phis

    # consistency check against the representatives recorded by analyze_merge.py,
    # written to a separate folder so the maps above are not overwritten:
    # --sel_range 0:200 is compared with the held-out analysis of e4a_merge.json
    # (selected on prompts 0-199), --sel_range all with the in-sample analysis
    python src/analysis/export_phi.py \
        --npz results/measure/npz/qwen3-8b__c4__N512__S400__fp16.npz \
        --methods greedy eigen similarity --ranks 4 8 16 --sel_range 0:200 \
        --model Qwen/Qwen3-8B --out_dir results/patching/phis_check \
        --check_json results/analysis/e4a_merge.json

Then:

    python src/patching/route_patch_ppl.py --model qwen3-8b \
        --phi_json results/patching/phis/phi_Qwen3-8B_greedy_R8.json --strict_phi
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root

import analyze_merge as AM  # noqa: E402
from spectral_common import load_dump, parse_range  # noqa: E402

METHODS = ["greedy", "eigen", "similarity", "similarity_gram", "random", "random_nearest"]


def phi_paths_for_layer(GP, GM, ranks, methods, rng):
    """{method: {R: phi}} for one layer, using analyze_merge's own constructors."""
    out = {}
    if "greedy" in methods:
        out["greedy"] = {R: p for R, (p, _) in AM.greedy_path(GP, GM, ranks).items()}
    if "eigen" in methods:
        out["eigen"] = {R: p for R, (p, _) in AM.eigen_path(GP, GM, ranks).items()}
    if "similarity" in methods:
        out["similarity"] = {R: p for R, (p, _) in AM.similarity_path(GP, ranks, "columns").items()}
    if "similarity_gram" in methods:
        out["similarity_gram"] = {R: p for R, (p, _) in AM.similarity_path(GP, ranks, "routes").items()}
    H = GP.shape[0]
    for mode, name in (("uniform", "random"), ("nearest", "random_nearest")):
        if name in methods:
            out[name] = {R: AM.random_phis(rng, H, R, 1, GP, GM, mode)[0] for R in ranks}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="Gram dump (see spectral_common.load_dump)")
    ap.add_argument("--methods", nargs="+", default=["greedy", "eigen", "similarity"],
                    choices=METHODS)
    ap.add_argument("--ranks", nargs="+", type=int, default=AM.DEFAULT_RANKS)
    ap.add_argument("--sel_range", default="all",
                    help="prompt range used to SELECT phi, e.g. '0:200' or 'all'")
    ap.add_argument("--seed", type=int, default=0, help="only used by the random baselines")
    ap.add_argument("--model", default=None, help="HF checkpoint name to stamp into the phi files")
    ap.add_argument("--out_dir", default=os.path.join(_ROOT, "results", "patching", "phis"))
    ap.add_argument("--check_json", default=None,
                    help="e4a_merge.json from analyze_merge.py; verify that the representative "
                         "sets agree layer by layer with its in-sample analysis (--sel_range all) "
                         "or its held-out analysis (any other --sel_range, which must equal the "
                         "--fit-range of analyze_merge.py, default 0:200)")
    args = ap.parse_args()

    dump = load_dump(args.npz)
    sel_idx = parse_range(args.sel_range, dump.S)
    model = args.model or dump.model
    H = dump.H
    print(f"dump {args.npz}: model={dump.model} S={dump.S} L={dump.L} H={H} "
          f"| selecting phi on prompts {args.sel_range} ({len(sel_idx)} of {dump.S})")

    rng = np.random.default_rng(args.seed)
    # {method: {R: {layer_id: phi list}}}
    collected = {m: {R: {} for R in args.ranks} for m in args.methods}
    for l in range(dump.L):
        GP = dump.gp_sum(l, sel_idx if len(sel_idx) else None)
        GM = dump.gm(l)
        per = phi_paths_for_layer(GP, GM, args.ranks, args.methods, rng)
        lid = dump.layer_id(l)
        for m in args.methods:
            for R in args.ranks:
                phi = per.get(m, {}).get(R)
                if phi is None:
                    continue
                phi = np.asarray(phi, dtype=int)
                assert phi.shape == (H,), (m, R, phi.shape)
                assert phi.min() >= 0 and phi.max() < H, (m, R)
                assert all(phi[phi[h]] == phi[h] for h in range(H)), \
                    f"{m} R={R} layer {lid}: phi not idempotent"
                n_reps = len(set(int(x) for x in phi))
                if n_reps != R:
                    print(f"  [warn] {m} R={R} layer {lid}: image size {n_reps} != R")
                collected[m][R][lid] = [int(x) for x in phi]

    os.makedirs(args.out_dir, exist_ok=True)
    short = model.split("/")[-1]
    written = []
    for m in args.methods:
        for R in args.ranks:
            layers = collected[m][R]
            if not layers:
                continue
            path = os.path.join(args.out_dir, f"phi_{short}_{m}_R{R}.json")
            with open(path, "w") as f:
                json.dump({
                    "schema_version": "1",
                    "model": model,
                    "num_query_heads": int(H),
                    "num_kv_heads": int(dump.config.get("num_kv_heads", 0)) or None,
                    "num_layers": int(max(layers) + 1),
                    "method": m,
                    "R": int(R),
                    "meta": {
                        "source": "export_phi.py",
                        "dump": os.path.basename(args.npz),
                        "sel_range": args.sel_range,
                        "seed": args.seed,
                        "dataset": dump.dataset,
                        "seq_len": dump.seq_len,
                    },
                    "layers": {str(k): layers[k] for k in sorted(layers)},
                }, f, indent=2)
            written.append(path)
            print(f"  wrote {path}  ({len(layers)} layers)")

    if args.check_json:
        # layout written by analyze_merge.py:
        #   results["models"][i]["analyses"][j]["layers"][k]["ranks"][str(R)]["reps"][method]
        with open(args.check_json) as f:
            res = json.load(f)
        want = os.path.basename(args.npz)
        # phi is SELECTED on one prompt range, so only the analysis whose selection
        # range matches ours is comparable: in_sample selects on all prompts,
        # heldout on analyze_merge.py's --fit-range (default 0:200)
        want_tag = "in_sample" if args.sel_range in ("all", ":", "") else "heldout"
        print(f"comparing with the '{want_tag}' analysis in {args.check_json}")
        n_ok = n_bad = 0
        for mb in res.get("models", []):
            if os.path.basename(mb.get("path", "")) != want:
                continue
            for an in mb.get("analyses", []):
                if an.get("tag") != want_tag:
                    continue
                for rec in an.get("layers", []):
                    lid = int(rec["layer"])
                    for R_key, r in rec.get("ranks", {}).items():
                        R = int(R_key)
                        for m, reps in r.get("reps", {}).items():
                            if m not in collected or R not in collected[m]:
                                continue
                            if lid not in collected[m][R]:
                                continue
                            mine = sorted(set(collected[m][R][lid]))
                            theirs = sorted(int(x) for x in reps)
                            if mine == theirs:
                                n_ok += 1
                            else:
                                n_bad += 1
                                print(f"  [MISMATCH] {m} R={R} layer {lid}: exported reps {mine} "
                                      f"!= analyze_merge reps {theirs}")
        print(f"consistency vs {args.check_json}: {n_ok} agree, {n_bad} mismatch")
        if n_bad or n_ok == 0:
            print("!!! representative sets could not be confirmed -- do NOT ship these phi files")
            sys.exit(1)

    print(f"Done: {len(written)} phi files in {args.out_dir}")


if __name__ == "__main__":
    main()
