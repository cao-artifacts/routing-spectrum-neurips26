#!/usr/bin/env python3
"""
Tiny synthetic Gram dumps for testing the offline analyzers.

Nothing here touches torch or the network. The files carry the arrays the
analyzers read (spectral_common.load_dump), in the layout of the dumps written
by experiment.py --dump_grams (see spectral_common.py for the full schema):

    G_M             float32 [L, H, H]     message Gram, weights only -> identical
                                          for every dataset of the same model
    G_P_per_prompt  float32 [S, L, H, H]  per-prompt route Gram contributions g_s;
                                          G_P^C = sum_{s in C} g_s
    G_P_total       float64 [L, H, H]     sum over all S prompts
    layer_index     int32   [L]
    config_json     str                   JSON blob, np.array(json.dumps(cfg))
    prompt_index    int32   [S]           index in the source stream

They leave out the measurement provenance of real dumps (quant_used, revision,
library versions and the other run metadata in config_json; the prompt
previews and per-layer corruption flags), so src/measure/verify_dump.py, which
checks that provenance, does not accept them.

The synthetic routes are built with planted structure so the analyzers produce
non-degenerate numbers: heads come in groups that share a route direction, the
per-prompt route energy varies, and the cross-domain dumps re-use the SAME G_M
with different route statistics.

    python src/analysis/make_synthetic_dump.py --outdir results/synthetic_dumps
"""

import argparse
import json
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root


def make_message_gram(rng, L, H, m2):
    """G_M[l] = B_l^T B_l with B_l in R^{m2 x H} (m2 = hidden_dim^2 in the real thing)."""
    GM = np.empty((L, H, H), dtype=np.float64)
    for l in range(L):
        B = rng.standard_normal((m2, H))
        # give the message maps unequal norms, like real W^V W^O factors
        B *= (0.5 + rng.random(H))[None, :]
        GM[l] = B.T @ B
    return GM


def make_per_prompt_gp(rng, S, L, H, n, groups, drift=0.0):
    """
    Per-prompt route Grams g_s = A_s^T A_s with planted group structure.

    A_s[:, h] = shared route of h's group + head-specific noise, so heads in the
    same group are genuinely mergeable and the spectrum is low-rank-ish.
    `drift` rotates the group routes to simulate a different corpus.
    """
    GP = np.empty((S, L, H, H), dtype=np.float32)
    n_groups = len(set(groups))
    for l in range(L):
        head_scale = 0.4 + rng.random(H)
        for s in range(S):
            Z = rng.standard_normal((n, n_groups))           # group route bases
            if drift > 0:
                Z = Z + drift * rng.standard_normal((n, n_groups))
            A = np.empty((n, H))
            for h in range(H):
                A[:, h] = head_scale[h] * (Z[:, groups[h]] + 0.35 * rng.standard_normal(n))
            A *= (0.7 + 0.6 * rng.random())                  # per-prompt energy jitter
            GP[s, l] = (A.T @ A).astype(np.float32)
    return GP


def write_dump(path, GM, GP, cfg, prompt_index=None):
    """Write the arrays the analyzers read (a subset of an experiment.py dump)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {
        "G_M": GM.astype(np.float32),
        "G_P_per_prompt": GP.astype(np.float32),
        "G_P_total": GP.astype(np.float64).sum(axis=0),
        "layer_index": np.arange(GM.shape[0], dtype=np.int32),
        "config_json": np.array(json.dumps(cfg, indent=2)),
    }
    if prompt_index is not None:
        arrays["prompt_index"] = np.asarray(prompt_index, dtype=np.int32)
    np.savez(path, **arrays)
    mb = os.path.getsize(path) / 1e6
    print("wrote %s  (G_M %s, G_P %s, %.2f MB)" % (path, GM.shape, GP.shape, mb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(_ROOT, "results", "synthetic_dumps"))
    ap.add_argument("--model", default="synth-16h")
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--prompts", type=int, default=400)
    ap.add_argument("--rows", type=int, default=24, help="stand-in for N^2 per prompt")
    ap.add_argument("--m2", type=int, default=20, help="stand-in for hidden_dim^2")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    H, L, S = args.heads, args.layers, args.prompts
    groups = [h % max(1, H // 4) for h in range(H)]   # H/4 groups of 4 heads

    rng = np.random.default_rng(args.seed)
    GM = make_message_gram(rng, L, H, args.m2)

    base_cfg = {
        "model": args.model,
        "hf_name": "synthetic/%s" % args.model,
        "num_query_heads": H,
        "num_kv_heads": max(1, H // 2),
        "num_layers": L,
        "num_prompts": S,
        "seq_len": args.seq_len,
        "schema_version": 1,
        "quantization": "none",
        "note": "synthetic fixture from make_synthetic_dump.py -- not a real model",
    }

    for dataset, drift, seed in (("c4", 0.0, 1), ("wikitext", 0.8, 2), ("code", 1.6, 3)):
        r = np.random.default_rng(args.seed + seed)
        n_prompts = S if dataset == "c4" else S // 2
        GP = make_per_prompt_gp(r, n_prompts, L, H, args.rows, groups, drift=drift)
        cfg = dict(base_cfg)
        cfg["dataset"] = dataset
        cfg["num_prompts"] = n_prompts
        cfg["split"] = "validation"
        write_dump(os.path.join(args.outdir, "%s_%s_n%d.npz" % (args.model, dataset, args.seq_len)),
                   GM, GP, cfg, prompt_index=np.arange(n_prompts))


if __name__ == "__main__":
    main()
