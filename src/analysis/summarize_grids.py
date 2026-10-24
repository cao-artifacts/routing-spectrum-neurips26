#!/usr/bin/env python3
"""Summarize E2 (domain x seq-len grid, Appendix H.2) and E3 (fp16 vs NF4 pairs,
Appendix H.4) from run JSONs.

Reads the per-cell JSONs written by src/measure/experiment.py from --cells
(default: the shipped data/robustness/cells/; use --cells results/measure/json
after a re-measurement). Latest clean version per cell: the bf16 rerun
supersedes the corrupted qwen3-8b x code fp16 cells. Cells missing from --cells
are left out of the grid, and a missing precision pair is listed as MISSING.
Writes e2_grid.{json,md} and e3_quant.{json,md} to --out_dir (default
results/analysis/).
"""
import json, glob, os, statistics, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
CELLS = os.path.join(ROOT, "data", "robustness", "cells")

MODEL_ORDER = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b",
               "qwen3-32b", "llama3.2-1b", "llama3.2-3b", "llama3-8b", "llama3-70b",
               "gemma2-2b", "gemma2-9b", "mistral-7b-v03", "mistral-nemo-12b"]
DATASETS = ["c4", "wikitext", "code"]
LENS = [256, 512, 1024]


def load_cell(path):
    if not os.path.exists(path):  # e.g. a partial re-measurement without the H.4 pairs
        return None
    r = list(json.load(open(path)).values())[0]
    layers = [v for v in r["layers"].values() if not v.get("corrupted")]
    n_bad = r["num_layers"] - len(layers)
    if not layers:
        return None
    med = lambda key: statistics.median(v[key] for v in layers)
    t = lambda R: statistics.median(v["tail_energy"][str(R)] if isinstance(v["tail_energy"], dict)
                                    else v["tail_energy"][{1: 0, 4: 1, 8: 2}[R]] for v in layers)
    def tail(v, R):
        te = v["tail_energy"]
        return te[str(R)] if isinstance(te, dict) else te[{1: 0, 4: 1, 8: 2}[R]]
    return {
        "r_eff_median": med("r_eff"),
        "T1": statistics.median(tail(v, 1) for v in layers),
        "T4": statistics.median(tail(v, 4) for v in layers),
        "T8": statistics.median(tail(v, 8) for v in layers),
        "corrupted_layers": n_bad,
        "quant_used": r.get("quant_used"),
        "revision": r.get("revision"),
        "src": os.path.relpath(path, ROOT),
        "per_layer_r_eff": {k: v["r_eff"] for k, v in r["layers"].items() if not v.get("corrupted")},
    }


def find_cell(model, ds, N, S=200):
    # bf16 fix wins over the corrupted auto cell
    cands = (glob.glob(os.path.join(CELLS, f"{model}__{ds}__N{N}__S{S}__bf16.json")) +
             glob.glob(os.path.join(CELLS, f"{model}__{ds}__N{N}__S{S}__auto.json")))
    for p in cands:
        c = load_cell(p)
        if c and c["corrupted_layers"] == 0:
            return c
    return load_cell(cands[-1]) if cands else None


def e2_grid():
    grid, rows = {}, []
    for m in MODEL_ORDER:
        for ds in DATASETS:
            for N in LENS:
                c = find_cell(m, ds, N)
                if c:
                    grid[f"{m}|{ds}|{N}"] = c
                    rows.append((m, ds, N, c))
    lines = ["# E2: domain x sequence-length grid (layer-median, S=200)", "",
             "| Model | Dataset | N | r_eff | T_1 | T_4 | T_8 | quant | corrupt |",
             "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for m, ds, N, c in rows:
        lines.append(f"| {m} | {ds} | {N} | {c['r_eff_median']:.2f} | {c['T1']:.3f} | "
                     f"{c['T4']:.3f} | {c['T8']:.3f} | {c['quant_used']} | {c['corrupted_layers']} |")
    # headline check: does T8 < 10% hold everywhere?
    viol = [(m, ds, N) for m, ds, N, c in rows if c["T8"] >= 0.10]
    lines += ["", f"**T8 < 10% holds in {len(rows) - len(viol)} / {len(rows)} cells.**"]
    if viol:
        lines.append(f"Violations: {viol}")
    return grid, "\n".join(lines)


def e3_pairs():
    pairs = [("qwen3-14b", os.path.join(CELLS, "qwen3-14b__c4__N512__S200__{q}.json")),
             ("mistral-nemo-12b", os.path.join(CELLS, "mistral-nemo-12b__c4__N512__S200__{q}.json")),
             ("qwen3-32b", os.path.join(CELLS, "qwen3-32b__c4__N512__S200__{q}.json")),
             ("llama3-70b", os.path.join(CELLS, "llama3-70b__c4__N512__S200__{q}.json"))]
    out, lines = {}, ["# E3: fp16 vs NF4 matched pairs (same checkpoint, c4/N=512/S=200)", "",
                      "| Model | r_eff fp16 | r_eff nf4 | dr_eff med | dr_eff max/layer | T8 fp16 | T8 nf4 |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
    for m, pat in pairs:
        a, b = load_cell(pat.format(q="fp16")), load_cell(pat.format(q="nf4"))
        if not (a and b):
            lines.append(f"| {m} | MISSING | | | | | |")
            continue
        common = set(a["per_layer_r_eff"]) & set(b["per_layer_r_eff"])
        deltas = [abs(a["per_layer_r_eff"][k] - b["per_layer_r_eff"][k]) for k in common]
        out[m] = {"fp16": a, "nf4": b,
                  "delta_r_eff_median": statistics.median(deltas),
                  "delta_r_eff_max": max(deltas)}
        lines.append(f"| {m} | {a['r_eff_median']:.3f} | {b['r_eff_median']:.3f} | "
                     f"{statistics.median(deltas):.3f} | {max(deltas):.3f} | "
                     f"{a['T8']:.4f} | {b['T8']:.4f} |")
    return out, "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E2 grid and E3 precision-pair summaries.")
    ap.add_argument("--cells", default=CELLS, help="directory with per-cell measurement JSONs")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results", "analysis"))
    args = ap.parse_args()
    CELLS = args.cells
    OUT = args.out_dir
    os.makedirs(OUT, exist_ok=True)
    grid, md2 = e2_grid()
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "per_layer_r_eff"} for k, v in grid.items()}
    json.dump(slim, open(os.path.join(OUT, "e2_grid.json"), "w"), indent=1)
    open(os.path.join(OUT, "e2_grid.md"), "w").write(md2 + "\n")
    pairs, md3 = e3_pairs()
    slim3 = {m: {"delta_r_eff_median": v["delta_r_eff_median"], "delta_r_eff_max": v["delta_r_eff_max"],
                 "fp16_r_eff": v["fp16"]["r_eff_median"], "nf4_r_eff": v["nf4"]["r_eff_median"],
                 "fp16_T8": v["fp16"]["T8"], "nf4_T8": v["nf4"]["T8"],
                 "fp16_src": v["fp16"]["src"], "nf4_src": v["nf4"]["src"]} for m, v in pairs.items()}
    json.dump(slim3, open(os.path.join(OUT, "e3_quant.json"), "w"), indent=1)
    open(os.path.join(OUT, "e3_quant.md"), "w").write(md3 + "\n")
    print(md2[:800]); print(); print(md3)
