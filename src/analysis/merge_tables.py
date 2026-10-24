#!/usr/bin/env python3
"""
Complete route-merging tables for every merge width R (appendix I.2 prints
only R in {4, 8}).

Reads the E4a output written by analyze_merge.py (default: the shipped
data/merge/e4a_merge.json) and writes, for the in-sample and held-out variants
and every R < H_q in {1, 2, 4, 8, 16}:

  * the layer-median calibration error fraction of every method: oracle
    (rank-R tail energy, a lower bound for any merge map), greedy and eigen
    (spectrum-guided), similarity (routing-only cosine of G_P columns),
    Gram-similarity (normalized G_P), and both random controls (uniform
    assignment and nearest-representative assignment; mean and best of the
    seeded draws);
  * per-model layer counts: greedy vs similarity, greedy vs Gram-similarity
    and eigen vs similarity (wins-losses-ties), and the number of layers in
    which greedy beats the mean of, or matches-or-beats the best of, each
    random control;
  * the same counts pooled over all models.

Nothing is recomputed from Gram matrices. Medians and counts are read from the
JSON summaries; only greedy vs Gram-similarity is tallied from the per-layer
records, with the tie tolerance of analyze_merge.summarize.

    python src/analysis/merge_tables.py
    python src/analysis/merge_tables.py --merge_json results/analysis/e4a_merge.json
"""

import argparse
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root

MODEL_ORDER = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b",
               "qwen3-32b", "llama3.2-1b", "llama3.2-3b", "llama3-8b", "llama3-70b",
               "gemma2-2b", "gemma2-9b", "mistral-7b-v03", "mistral-nemo-12b"]

VARIANTS = [
    ("in_sample", "In-sample (merge map selected and scored on all 400 prompts)"),
    ("heldout", "Held-out (merge map selected on prompts 0-199, scored on prompts 200-399)"),
]

# (key in summary["median"] or "oracle", markdown header)
ERROR_COLS = [
    ("oracle", "oracle"), ("greedy", "greedy"), ("eigen", "eigen"),
    ("similarity", "similarity"), ("similarity_gram", "Gram-sim."),
    ("random_mean", "random mean"), ("random_min", "random best"),
    ("random_nearest_mean", "nearest-random mean"), ("random_nearest_min", "nearest-random best"),
]

COUNT_COLS = [
    # (csv prefix, markdown header)
    ("greedy_vs_similarity", "greedy vs similarity W-L-T"),
    ("greedy_vs_similarity_gram", "greedy vs Gram-sim. W-L-T"),
    ("eigen_vs_similarity", "eigen vs similarity W-L-T"),
]

RANDOM_COLS = [
    ("greedy_beats_random_mean", "greedy < random mean"),
    ("greedy_le_random_best", "greedy <= random best"),
    ("greedy_beats_nearest_random_mean", "greedy < nearest-random mean"),
    ("greedy_le_nearest_random_best", "greedy <= nearest-random best"),
]


def tally(a_vals, b_vals):
    """Wins/losses/ties of a vs b (lower is better), same tolerance as analyze_merge.summarize."""
    w = l_ = t = 0
    for a, b in zip(a_vals, b_vals):
        if a < b * (1 - 1e-12):
            w += 1
        elif a > b * (1 + 1e-12):
            l_ += 1
        else:
            t += 1
    return w, l_, t


def model_rank(name):
    return MODEL_ORDER.index(name) if name in MODEL_ORDER else len(MODEL_ORDER)


def build_rows(results):
    rows = []
    models = sorted(results["models"], key=lambda m: model_rank(m["model"]))
    for tag, _ in VARIANTS:
        for m in models:
            an = next((a for a in m.get("analyses", []) if a.get("tag") == tag), None)
            if an is None:
                continue
            for R in sorted(int(k) for k in an["summary"]):
                s = an["summary"][str(R)]
                per_layer = [x["ranks"][str(R)] for x in an["layers"] if str(R) in x["ranks"]]
                if len(per_layer) != s["n_layers"]:
                    raise SystemExit("%s %s R=%d: %d per-layer records vs n_layers=%d"
                                     % (m["model"], tag, R, len(per_layer), s["n_layers"]))
                g_gram = tally([r["methods"]["greedy"] for r in per_layer],
                               [r["methods"]["similarity_gram"] for r in per_layer])
                row = {"variant": tag, "model": m["model"], "H": m["H"], "L": m["L"], "R": R,
                       "n_layers": s["n_layers"], "oracle": s["oracle"]}
                for key, _ in ERROR_COLS[1:]:
                    row[key] = s["median"][key]
                for prefix, (w, l_, t) in (
                        ("greedy_vs_similarity", (s["vs_similarity"]["greedy"]["wins"],
                                                  s["vs_similarity"]["greedy"]["losses"],
                                                  s["vs_similarity"]["greedy"]["ties"])),
                        ("greedy_vs_similarity_gram", g_gram),
                        ("eigen_vs_similarity", (s["vs_similarity"]["eigen"]["wins"],
                                                 s["vs_similarity"]["eigen"]["losses"],
                                                 s["vs_similarity"]["eigen"]["ties"]))):
                    row[prefix + "_wins"], row[prefix + "_losses"], row[prefix + "_ties"] = w, l_, t
                vr, vn = s["vs_random"]["random"], s["vs_random"]["random_nearest"]
                row["random_draws"] = vr["n_draws"]
                row["greedy_beats_random_mean"] = vr["layers_beating_random_mean"]
                row["greedy_le_random_best"] = vr["layers_beating_best_draw"]
                row["greedy_beats_nearest_random_mean"] = vn["layers_beating_random_mean"]
                row["greedy_le_nearest_random_best"] = vn["layers_beating_best_draw"]
                rows.append(row)
    return rows


def pooled(rows, tag, R):
    sel = [r for r in rows if r["variant"] == tag and r["R"] == R]
    out = {"n_models": len(sel), "n_layers": sum(r["n_layers"] for r in sel)}
    for prefix, _ in COUNT_COLS:
        for k in ("wins", "losses", "ties"):
            out[prefix + "_" + k] = sum(r[prefix + "_" + k] for r in sel)
    for key, _ in RANDOM_COLS:
        out[key] = sum(r[key] for r in sel)
    return out


def pct(n, d):
    return "%.1f%%" % (100.0 * n / d) if d else "--"


def write_markdown(rows, path, source):
    lines = ["# Route-merging study: complete tables for every merge width R", "",
             "Source: `%s` (written by `src/analysis/analyze_merge.py` from the C4, N=512, "
             "S=400 Gram dumps of the 14 checkpoints)." % source, "",
             "Error columns: layer-median calibration error fraction "
             "tr(C^T G_P C G_M) / tr(G_P G_M) of the merge map at width R (lower is better). "
             "`oracle` is the rank-R tail energy, a lower bound for any merge map. "
             "`random` assigns each head to a uniformly random representative, "
             "`nearest-random` to its nearest randomly chosen representative; "
             "`mean` and `best` are over the seeded draws per layer.", "",
             "Count columns: number of layers (out of `layers`). W-L-T = wins-losses-ties, "
             "lower error wins.", ""]
    for tag, title in VARIANTS:
        ranks = sorted({r["R"] for r in rows if r["variant"] == tag})
        for R in ranks:
            sel = [r for r in rows if r["variant"] == tag and r["R"] == R]
            lines += ["## %s, R = %d" % (title, R), ""]
            hdr = ["Model", "layers"] + [h for _, h in ERROR_COLS]
            lines.append("| " + " | ".join(hdr) + " |")
            lines.append("|" + "|".join(["---"] + ["---:"] * (len(hdr) - 1)) + "|")
            for r in sel:
                vals = ["%.4f" % r[k] for k, _ in ERROR_COLS]
                lines.append("| " + " | ".join([r["model"], str(r["n_layers"])] + vals) + " |")
            lines.append("")
            hdr = ["Model", "layers"] + [h for _, h in COUNT_COLS] + [h for _, h in RANDOM_COLS]
            lines.append("| " + " | ".join(hdr) + " |")
            lines.append("|" + "|".join(["---"] + ["---:"] * (len(hdr) - 1)) + "|")
            for r in sel:
                cells = ["%d-%d-%d" % (r[p + "_wins"], r[p + "_losses"], r[p + "_ties"]) for p, _ in COUNT_COLS]
                cells += [str(r[k]) for k, _ in RANDOM_COLS]
                lines.append("| " + " | ".join([r["model"], str(r["n_layers"])] + cells) + " |")
            P = pooled(rows, tag, R)
            n = P["n_layers"]
            cells = ["%d-%d-%d (%s)" % (P[p + "_wins"], P[p + "_losses"], P[p + "_ties"], pct(P[p + "_wins"], n))
                     for p, _ in COUNT_COLS]
            cells += ["%d (%s)" % (P[k], pct(P[k], n)) for k, _ in RANDOM_COLS]
            lines.append("| " + " | ".join(["**pooled, %d models**" % P["n_models"], str(n)] + cells) + " |")
            lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s" % path)


def write_csv(rows, path):
    fields = ["variant", "model", "H", "L", "R", "n_layers"] + [k for k, _ in ERROR_COLS]
    for p, _ in COUNT_COLS:
        fields += [p + "_wins", p + "_losses", p + "_ties"]
    fields += ["random_draws"] + [k for k, _ in RANDOM_COLS]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print("wrote %s" % path)


def main():
    ap = argparse.ArgumentParser(description="Route-merging tables for every merge width R.")
    ap.add_argument("--merge_json", default=os.path.join(ROOT, "data", "merge", "e4a_merge.json"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results", "analysis"))
    args = ap.parse_args()

    with open(args.merge_json) as f:
        results = json.load(f)
    rows = build_rows(results)
    os.makedirs(args.out_dir, exist_ok=True)
    source = os.path.relpath(os.path.abspath(args.merge_json), ROOT)
    write_csv(rows, os.path.join(args.out_dir, "merge_tables_all_R.csv"))
    write_markdown(rows, os.path.join(args.out_dir, "merge_tables_all_R.md"), source)

    print("\npooled over models (greedy vs similarity | vs Gram-similarity | "
          "<= best random | <= best nearest-random):")
    for tag, _ in VARIANTS:
        for R in sorted({r["R"] for r in rows if r["variant"] == tag}):
            P = pooled(rows, tag, R)
            n = P["n_layers"]
            print("  %-9s R=%-2d layers=%-4d  %4d (%s) | %4d (%s) | %4d (%s) | %4d (%s)" % (
                tag, R, n,
                P["greedy_vs_similarity_wins"], pct(P["greedy_vs_similarity_wins"], n),
                P["greedy_vs_similarity_gram_wins"], pct(P["greedy_vs_similarity_gram_wins"], n),
                P["greedy_le_random_best"], pct(P["greedy_le_random_best"], n),
                P["greedy_le_nearest_random_best"], pct(P["greedy_le_nearest_random_best"], n)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
