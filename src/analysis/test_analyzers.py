#!/usr/bin/env python3
"""
Standalone property tests for the offline analyzers of Appendices H and I.

No .npz, no GPU, no network: every test builds explicit tiny matrices A and B,
forms Psi = A B^T, and checks that the closed-form Gram expressions used by
analyze_heldout.py and analyze_merge.py agree with the explicit object they are
supposed to compute.  Exits nonzero if anything fails.

    python src/analysis/test_analyzers.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_heldout  # noqa: E402
import analyze_merge  # noqa: E402
import spectral_common as sc  # noqa: E402


# ── a few checks on the shared layer itself ───────────────────────

def test_common_helpers(verbose=True):
    """spectrum_sigma_sq / total_energy / psd_sqrt agree with explicit Psi quantities."""
    rng = np.random.default_rng(1234)
    H, rows_A, rows_B = 7, 45, 33
    A = rng.standard_normal((rows_A, H))
    B = rng.standard_normal((rows_B, H))
    Psi = A @ B.T
    GP, GM = A.T @ A, B.T @ B

    ss = sc.spectrum_sigma_sq(GP, GM)                       # length H
    sv2 = np.sort(np.linalg.svd(Psi, compute_uv=False) ** 2)[::-1][:H]  # rank(Psi) <= H
    d_spec = float(np.abs(ss - sv2).max() / sv2.max())

    d_energy = abs(sc.total_energy(GP, GM) - float(np.linalg.norm(Psi) ** 2)) / float(np.linalg.norm(Psi) ** 2)

    half, half_pinv, rank = sc.psd_sqrt(GM)
    d_sqrt = float(np.abs(half @ half - GM).max() / np.abs(GM).max())
    d_pinv = float(np.abs(half @ half_pinv @ half - half).max() / max(np.abs(half).max(), 1e-300))

    # r_eff and T_R must reproduce the published experiment.py definitions
    sigma = np.sqrt(sv2)
    d_reff = abs(sc.r_eff(ss) - float(sigma.sum() ** 2 / sv2.sum()))
    d_tail = max(abs(sc.tail_fraction(ss, R) - float(sv2[R:].sum() / sv2.sum())) for R in (1, 2, 4))

    worst = max(d_spec, d_energy, d_sqrt, d_pinv, d_reff, d_tail)
    ok = worst < 1e-9 and rank == H
    if verbose:
        print("  [c1]  spectrum/energy/sqrt/r_eff/T_R vs explicit Psi: max rel diff = %.3e -> %s"
              % (worst, "PASS" if ok else "FAIL"))
    return ok, worst


def test_per_prompt_additivity(verbose=True):
    """Per-prompt contributions g_s must sum to G_P (the split machinery depends on it)."""
    rng = np.random.default_rng(77)
    H, S, n = 6, 9, 12
    blocks = [rng.standard_normal((n, H)) for _ in range(S)]
    A = np.vstack(blocks)
    GP_full = A.T @ A
    GP_sum = sum(b.T @ b for b in blocks)
    d = float(np.abs(GP_full - GP_sum).max() / np.abs(GP_full).max())
    idx = [0, 3, 4, 7]
    sub = np.vstack([blocks[i] for i in idx])
    d2 = float(np.abs(sub.T @ sub - sum(blocks[i].T @ blocks[i] for i in idx)).max()
               / np.abs(GP_full).max())
    ok = max(d, d2) < 1e-12
    if verbose:
        print("  [c2]  G_P^C = sum_{s in C} g_s (full and subset): max rel diff = %.3e -> %s"
              % (max(d, d2), "PASS" if ok else "FAIL"))
    return ok, max(d, d2)


def test_schema_roundtrip(verbose=True):
    """A dump written to the documented schema must load back with the same numbers."""
    import json
    import tempfile

    rng = np.random.default_rng(5)
    S, L, H = 5, 3, 4
    GM = np.stack([sc.sym(rng.standard_normal((H, H)) @ rng.standard_normal((H, H)).T)
                   for _ in range(L)])
    GP = np.abs(rng.standard_normal((S, L, H, H))).astype(np.float32)
    GP = 0.5 * (GP + np.swapaxes(GP, -1, -2))
    cfg = {"model": "unit-test", "hf_name": "n/a", "dataset": "c4", "seq_len": 512,
           "num_query_heads": H, "num_kv_heads": H // 2, "num_layers": L,
           "num_prompts": S, "schema_version": sc.SCHEMA_VERSION}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "unit.npz")
        np.savez(path, G_M=GM, G_P=GP, config=np.array(json.dumps(cfg)))
        d = sc.load_dump(path)
        ok = (d.H == H and d.L == L and d.S == S and d.model == "unit-test"
              and d.dataset == "c4" and d.seq_len == 512)
        d_gm = float(np.abs(d.gm(1) - sc.sym(GM[1])).max())
        want = sc.sym(GP[[0, 2, 4], 1].astype(np.float64).sum(axis=0))
        d_gp = float(np.abs(d.gp_sum(1, [0, 2, 4]) - want).max())
        flat = d.gp_layer_flat(2)
        d_flat = float(np.abs(flat.reshape(S, H, H) - GP[:, 2].astype(np.float64)).max())
    ok = ok and max(d_gm, d_gp, d_flat) < 1e-9
    if verbose:
        print("  [c3]  .npz schema round-trip (G_M [L,H,H], G_P [S,L,H,H], config json): "
              "max diff = %.3e -> %s" % (max(d_gm, d_gp, d_flat), "PASS" if ok else "FAIL"))
    return ok, max(d_gm, d_gp, d_flat)


def common_tests(verbose=True):
    if verbose:
        print("spectral_common property tests")
    return [
        ("common helpers vs explicit Psi", ) + test_common_helpers(verbose),
        ("per-prompt additivity", ) + test_per_prompt_additivity(verbose),
        ("npz schema round-trip", ) + test_schema_roundtrip(verbose),
    ]


def main():
    t0 = time.time()
    print("=" * 68)
    print("property tests for the offline analyzers (Appendices H and I)")
    print("=" * 68)
    results = []
    results += [("common:" + n, ok, v) for n, ok, v in common_tests(verbose=True)]
    print("")
    results += [("heldout:" + n, ok, v) for n, ok, v in analyze_heldout.property_tests(verbose=True)]
    print("")
    results += [("merge:" + n, ok, v) for n, ok, v in analyze_merge.property_tests(verbose=True)]
    print("")
    print("-" * 68)
    failed = [n for n, ok, _ in results if not ok]
    for n, ok, v in results:
        print("  %-4s %s" % ("PASS" if ok else "FAIL", n))
    print("-" * 68)
    print("%d/%d tests passed in %.1fs" % (len(results) - len(failed), len(results), time.time() - t0))
    if failed:
        print("FAILURES: %s" % ", ".join(failed))
        return 1
    print("ALL PROPERTY TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
