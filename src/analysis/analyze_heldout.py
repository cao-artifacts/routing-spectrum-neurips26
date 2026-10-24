#!/usr/bin/env python3
"""
E1: held-out calibration validation + bootstrap confidence intervals.

Offline, pure numpy, CPU.  Consumes the .npz dumps written by the GPU
measurement drivers (schema documented in spectral_common.py) and tests whether
the low routing-message rank is a property of the model or an artefact of
fitting 200 C4 prompts.

Three analyses per model:

  (a) held-out split   fit the width-R student on prompts [0:200] (the published
                       calibration set), score it on prompts [200:400];
                       report per-layer and layer-median residual for R in {1,4,8}
                       next to the in-sample tail energy T_R.
  (b) bootstrap        resample 200-prompt subsets with replacement from the
                       prompt pool (seeded, 500 reps) and report 2.5/97.5
                       percentile CIs for r_eff and T_8 (median over layers).
  (c) cross-domain     fit W on the c4 dump, score on the wikitext / code dump
                       of the SAME model and seq_len (G_M must be identical --
                       asserted, since it comes from weights only).

THE MATH (derivation, so the numbers can be checked by hand)
------------------------------------------------------------
Psi = A B^T with A = [a_1..a_H] (routes, G_P = A^T A) and B = [b_1..b_H]
(messages, G_M = B^T B).  A width-R student keeps the message maps fixed but
may use only R fixed combinations m_j = B w_j of them, with per-input routes
free; its stacked matrix is Atilde (B W)^T for an unconstrained Atilde.  The
best such approximation of Psi projects Psi's rows onto span(B W), so

    residual(W) = ||Psi (I - Pi_{BW})||_F^2 / ||Psi||_F^2
                = 1 - tr(G_P G_M W (W^T G_M W)^{-1} W^T G_M) / tr(G_P G_M).

Maximizing the captured term over W with W = G_M^{-1/2} U reduces to
tr((U^T U)^{-1} U^T S U) with S = G_M^{1/2} G_P G_M^{1/2}, i.e. U = top-R
eigenvectors of S, equivalently  G_P G_M w_j = lambda_j w_j.  Because the row
space of Psi already lies in col(B), the top-R right singular vectors of Psi
are reachable, so on the fitting split the residual coincides exactly with the
SVD tail energy T_R reported in the paper.  On a different split it does not --
that gap is the held-out generalization gap reported in Appendix H.1.

Usage (from the repository root; dumps come from src/measure/run_batch.py)
-----
    python src/analysis/analyze_heldout.py \
        --npz results/measure/npz/qwen3-8b__c4__N512__S400__fp16.npz \
        --cross results/measure/npz/qwen3-8b__wikitext__N512__S200__fp16.npz \
                results/measure/npz/qwen3-8b__code__N512__S200__bf16.npz
    python src/analysis/analyze_heldout.py --self-test
Outputs default to results/analysis/e1_heldout.{json,md}.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root

from spectral_common import (  # noqa: E402
    assert_same_message_gram, fit_route_basis, load_dump, md_table, median,
    parse_range, pct, psd_sqrt, r_eff, residual_fraction, spectrum_sigma_sq,
    sym, tail_fraction, write_json, write_text,
)

DEFAULT_RANKS = [1, 4, 8]


# ══════════════════════════════════════════════════════════════════
# property tests (no npz needed; also run by test_analyzers.py)
# ══════════════════════════════════════════════════════════════════

def _explicit_case(seed=0, H=6, rows_A=40, rows_B=30):
    """Build an explicit A, B, Psi with the paper's shapes (S*N^2=40, m^2=30)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((rows_A, H))
    B = rng.standard_normal((rows_B, H))
    Psi = A @ B.T
    return A, B, Psi


def _explicit_residual(Psi_eval, B, W):
    """|| Psi_eval (I - Pi_V) ||_F^2 / ||Psi_eval||_F^2 with V = B W, built explicitly."""
    V = B @ W
    if V.size == 0:
        return 1.0
    Pi = V @ np.linalg.pinv(V.T @ V) @ V.T
    num = np.linalg.norm(Psi_eval @ (np.eye(Pi.shape[0]) - Pi)) ** 2
    den = np.linalg.norm(Psi_eval) ** 2
    return float(num / den) if den > 0 else 0.0


def test_full_split_equals_svd_tail(tol=1e-8, verbose=True):
    """(i) split A == split B == full set  =>  residual(R) == SVD tail energy of Psi."""
    A, B, Psi = _explicit_case(seed=0)
    H = A.shape[1]
    GP = A.T @ A
    GM = B.T @ B
    lam, W_full = fit_route_basis(GP, GM)
    sv = np.linalg.svd(Psi, compute_uv=False)
    ss = sv ** 2
    worst = 0.0
    for R in range(1, H + 1):
        got = residual_fraction(GP, GM, W_full[:, :R])
        want = float(ss[R:].sum() / ss.sum())
        worst = max(worst, abs(got - want))
    # the fitted eigenvalues must themselves be the squared singular values
    worst_lam = float(np.abs(np.sort(lam)[::-1] - np.sort(ss)[::-1][:H]).max() / ss.max())
    ok = worst < tol and worst_lam < 1e-10
    if verbose:
        print("  [i]   full-set residual vs SVD tail: max |diff| = %.3e (tol %.0e); "
              "lambda vs sigma^2 rel = %.3e -> %s" % (worst, tol, worst_lam, "PASS" if ok else "FAIL"))
    return ok, worst


def test_split_matches_explicit_projection(tol=1e-8, verbose=True, n_cases=5):
    """(ii) A != B random splits: closed form == explicit ||Psi_B (I - Pi_V)||_F^2 / ||Psi_B||_F^2."""
    worst = 0.0
    for seed in range(n_cases):
        rng = np.random.default_rng(100 + seed)
        A, B, _ = _explicit_case(seed=seed)
        rows, H = A.shape
        perm = rng.permutation(rows)
        cut = rows // 2
        iA, iB = perm[:cut], perm[cut:]
        GP_A = A[iA].T @ A[iA]
        GP_B = A[iB].T @ A[iB]
        GM = B.T @ B
        _, W_full = fit_route_basis(GP_A, GM)
        Psi_B = A[iB] @ B.T
        for R in range(1, H + 1):
            W = W_full[:, :R]
            got = residual_fraction(GP_B, GM, W)
            want = _explicit_residual(Psi_B, B, W)
            worst = max(worst, abs(got - want))
    ok = worst < tol
    if verbose:
        print("  [ii]  held-out closed form vs explicit projection: max |diff| = %.3e "
              "(tol %.0e) -> %s" % (worst, tol, "PASS" if ok else "FAIL"))
    return ok, worst


def test_monotone_in_R(tol=1e-10, verbose=True, n_cases=5):
    """(iii) residual is nonincreasing in R (nested eigenvector basis), in and out of sample."""
    worst = -np.inf
    for seed in range(n_cases):
        rng = np.random.default_rng(200 + seed)
        A, B, _ = _explicit_case(seed=seed)
        rows, H = A.shape
        perm = rng.permutation(rows)
        iA, iB = perm[: rows // 2], perm[rows // 2:]
        GM = B.T @ B
        GP_A = A[iA].T @ A[iA]
        GP_B = A[iB].T @ A[iB]
        _, W_full = fit_route_basis(GP_A, GM)
        for GP in (GP_A, GP_B):
            prev = None
            for R in range(1, H + 1):
                cur = residual_fraction(GP, GM, W_full[:, :R])
                if prev is not None:
                    worst = max(worst, cur - prev)  # must be <= 0
                prev = cur
    ok = worst < tol
    if verbose:
        print("  [iii] residual monotone nonincreasing in R: max increase = %.3e "
              "(tol %.0e) -> %s" % (worst, tol, "PASS" if ok else "FAIL"))
    return ok, worst


def test_rank_deficient_gm(tol=1e-8, verbose=True):
    """Extra: rank-deficient G_M (duplicate / dependent message maps) must not explode."""
    rng = np.random.default_rng(7)
    H, rows_A, rows_B = 8, 30, 3  # m^2 = 3 < H  ->  G_M has rank 3
    A = rng.standard_normal((rows_A, H))
    B = rng.standard_normal((rows_B, H))
    Psi = A @ B.T
    GP, GM = A.T @ A, B.T @ B
    _, W_full = fit_route_basis(GP, GM)
    sv = np.linalg.svd(Psi, compute_uv=False)
    ss = sv ** 2
    worst = 0.0
    for R in range(1, H + 1):
        got = residual_fraction(GP, GM, W_full[:, :R])
        want = float(ss[R:].sum() / ss.sum())
        worst = max(worst, abs(got - want))
    ok = worst < tol and np.all(np.isfinite(W_full))
    if verbose:
        print("  [iv]  rank-deficient G_M (rank %d < H=%d): max |diff| vs SVD tail = %.3e -> %s"
              % (np.linalg.matrix_rank(GM), H, worst, "PASS" if ok else "FAIL"))
    return ok, worst


def test_generalized_eigen_relation(tol=1e-8, verbose=True):
    """Extra: the fitted columns really solve G_P G_M w = lambda w."""
    A, B, _ = _explicit_case(seed=3)
    GP, GM = A.T @ A, B.T @ B
    lam, W = fit_route_basis(GP, GM)
    R = 4
    resid = np.abs(GP @ GM @ W[:, :R] - W[:, :R] * lam[:R]).max()
    scale = np.abs(W[:, :R] * lam[:R]).max()
    rel = resid / max(scale, 1e-300)
    ok = rel < tol
    if verbose:
        print("  [v]   G_P G_M w_j = lambda_j w_j: relative residual = %.3e -> %s"
              % (rel, "PASS" if ok else "FAIL"))
    return ok, rel


def test_insample_equals_tail_energy(tol=1e-10, verbose=True):
    """Extra: in-sample residual == T_R from the published spectrum routine."""
    A, B, _ = _explicit_case(seed=11)
    GP, GM = A.T @ A, B.T @ B
    lam, W_full = fit_route_basis(GP, GM)
    ss = spectrum_sigma_sq(GP, GM)
    worst = 0.0
    for R in (1, 2, 4, 8):
        got = residual_fraction(GP, GM, W_full[:, :R])
        worst = max(worst, abs(got - tail_fraction(ss, R)))
    ok = worst < tol
    if verbose:
        print("  [vi]  in-sample residual == T_R of spectrum_sigma_sq: max |diff| = %.3e -> %s"
              % (worst, "PASS" if ok else "FAIL"))
    return ok, worst


def property_tests(verbose=True):
    """Run every held-out property test.  Returns list of (name, ok, value)."""
    if verbose:
        print("analyze_heldout property tests")
    out = [
        ("full-split residual == SVD tail", ) + test_full_split_equals_svd_tail(verbose=verbose),
        ("held-out residual == explicit projection", ) + test_split_matches_explicit_projection(verbose=verbose),
        ("monotone nonincreasing in R", ) + test_monotone_in_R(verbose=verbose),
        ("rank-deficient G_M", ) + test_rank_deficient_gm(verbose=verbose),
        ("generalized eigen relation", ) + test_generalized_eigen_relation(verbose=verbose),
        ("in-sample residual == T_R", ) + test_insample_equals_tail_energy(verbose=verbose),
    ]
    return out


# ══════════════════════════════════════════════════════════════════
# analyses
# ══════════════════════════════════════════════════════════════════

def analyze_heldout_split(dump, fit_idx, eval_idx, ranks):
    """(a) fit on fit_idx, score on eval_idx; per layer + median."""
    layers = []
    for l in range(dump.L):
        GM = dump.gm(l)
        GMh, GMh_pinv, gm_rank = psd_sqrt(GM)
        GP_fit = dump.gp_sum(l, fit_idx)
        GP_eval = dump.gp_sum(l, eval_idx)
        lam, W_full = fit_route_basis(GP_fit, GM, GMh, GMh_pinv)
        ss_fit = np.sort(lam)[::-1]
        ss_eval = spectrum_sigma_sq(GP_eval, GM, GMh)

        rec = {
            "layer": dump.layer_id(l),
            "gm_rank": gm_rank,
            "r_eff_fit": r_eff(ss_fit),
            "r_eff_eval": r_eff(ss_eval),
            "energy_fit": float(ss_fit.sum()),
            "energy_eval": float(ss_eval.sum()),
            "in_sample_T_R": {},
            "heldout_residual": {},
            "eval_oracle_T_R": {},
            "gap": {},
        }
        for R in ranks:
            t_in = tail_fraction(ss_fit, R)
            t_or = tail_fraction(ss_eval, R)          # best possible on the eval split
            res = residual_fraction(GP_eval, GM, W_full[:, :R])
            rec["in_sample_T_R"][str(R)] = t_in
            rec["eval_oracle_T_R"][str(R)] = t_or
            rec["heldout_residual"][str(R)] = res
            rec["gap"][str(R)] = res - t_in
        layers.append(rec)

    summary = {"ranks": [int(R) for R in ranks], "median": {}}
    summary["median"]["r_eff_fit"] = median([x["r_eff_fit"] for x in layers])
    summary["median"]["r_eff_eval"] = median([x["r_eff_eval"] for x in layers])
    for R in ranks:
        k = str(R)
        summary["median"].setdefault("in_sample_T_R", {})[k] = median([x["in_sample_T_R"][k] for x in layers])
        summary["median"].setdefault("heldout_residual", {})[k] = median([x["heldout_residual"][k] for x in layers])
        summary["median"].setdefault("eval_oracle_T_R", {})[k] = median([x["eval_oracle_T_R"][k] for x in layers])
        summary["median"].setdefault("gap", {})[k] = median([x["gap"][k] for x in layers])
    return {"layers": layers, "summary": summary}


def analyze_bootstrap(dump, pool_idx, n_reps, subset_size, seed, tail_R=8, ranks=(8,)):
    """
    (b) bootstrap over prompt subsets.

    Each rep draws `subset_size` prompts WITH replacement from `pool_idx`,
    rebuilds G_P as the (multiplicity-weighted) sum of per-prompt contributions,
    and recomputes the spectrum.  Reported statistic per rep is the median over
    layers, matching how the paper reports r_eff and T_R.
    """
    rng = np.random.default_rng(seed)
    pool = np.asarray(pool_idx, dtype=int)
    S = dump.S
    draws = rng.integers(0, pool.size, size=(n_reps, subset_size))
    counts = np.zeros((n_reps, S), dtype=np.float64)
    for b in range(n_reps):
        idx = pool[draws[b]]
        counts[b] += np.bincount(idx, minlength=S)

    tracked = sorted(set([int(tail_R)] + [int(x) for x in ranks]))
    per_rep_reff = np.zeros((n_reps, dump.L))
    per_rep_tail = {R: np.zeros((n_reps, dump.L)) for R in tracked}

    for l in range(dump.L):
        GM = dump.gm(l)
        GMh, _, _ = psd_sqrt(GM)
        flat = dump.gp_layer_flat(l)              # [S, H*H]
        boot = counts @ flat                      # [reps, H*H]
        for b in range(n_reps):
            GP = sym(boot[b].reshape(dump.H, dump.H))
            ss = spectrum_sigma_sq(GP, GM, GMh)
            per_rep_reff[b, l] = r_eff(ss)
            for R in tracked:
                per_rep_tail[R][b, l] = tail_fraction(ss, R)

    med_reff = np.median(per_rep_reff, axis=1)
    out = {
        "n_reps": int(n_reps),
        "subset_size": int(subset_size),
        "pool_size": int(pool.size),
        "seed": int(seed),
        "r_eff": {
            "mean": float(np.mean(med_reff)),
            "ci_lo": pct(med_reff, 2.5),
            "ci_hi": pct(med_reff, 97.5),
            "std": float(np.std(med_reff, ddof=1)) if n_reps > 1 else 0.0,
        },
        "tail": {},
    }
    for R in tracked:
        med_t = np.median(per_rep_tail[R], axis=1)
        out["tail"]["T_%d" % R] = {
            "mean": float(np.mean(med_t)),
            "ci_lo": pct(med_t, 2.5),
            "ci_hi": pct(med_t, 97.5),
            "std": float(np.std(med_t, ddof=1)) if n_reps > 1 else 0.0,
        }

    # point estimates on the same pool, for reference
    point_reff, point_tail = [], {R: [] for R in tracked}
    for l in range(dump.L):
        GM = dump.gm(l)
        ss = spectrum_sigma_sq(dump.gp_sum(l, pool), GM)
        point_reff.append(r_eff(ss))
        for R in tracked:
            point_tail[R].append(tail_fraction(ss, R))
    out["point_estimate"] = {
        "r_eff": median(point_reff),
        "tail": {"T_%d" % R: median(point_tail[R]) for R in tracked},
    }
    return out


def analyze_cross_domain(base, cross, fit_idx, ranks, eval_idx=None):
    """(c) fit W on the base (c4) dump, score on another domain of the same model."""
    rel = assert_same_message_gram(base, cross)
    layers = []
    for l in range(base.L):
        GM = base.gm(l)
        GMh, GMh_pinv, _ = psd_sqrt(GM)
        GP_fit = base.gp_sum(l, fit_idx)
        GP_x = cross.gp_sum(l, eval_idx)
        lam, W_full = fit_route_basis(GP_fit, GM, GMh, GMh_pinv)
        ss_fit = np.sort(lam)[::-1]
        ss_x = spectrum_sigma_sq(GP_x, GM, GMh)
        rec = {"layer": base.layer_id(l),
               "r_eff_fit": r_eff(ss_fit),
               "r_eff_cross": r_eff(ss_x),
               "in_sample_T_R": {}, "cross_residual": {}, "cross_oracle_T_R": {}, "gap": {}}
        for R in ranks:
            t_in = tail_fraction(ss_fit, R)
            t_or = tail_fraction(ss_x, R)
            res = residual_fraction(GP_x, GM, W_full[:, :R])
            rec["in_sample_T_R"][str(R)] = t_in
            rec["cross_oracle_T_R"][str(R)] = t_or
            rec["cross_residual"][str(R)] = res
            rec["gap"][str(R)] = res - t_in
        layers.append(rec)

    summary = {"ranks": [int(R) for R in ranks],
               "gm_relative_difference": rel,
               "median": {"r_eff_fit": median([x["r_eff_fit"] for x in layers]),
                          "r_eff_cross": median([x["r_eff_cross"] for x in layers])}}
    for R in ranks:
        k = str(R)
        for field in ("in_sample_T_R", "cross_residual", "cross_oracle_T_R", "gap"):
            summary["median"].setdefault(field, {})[k] = median([x[field][k] for x in layers])
    return {"layers": layers, "summary": summary,
            "base": base.label(), "cross": cross.label(),
            "cross_dataset": cross.dataset, "cross_num_prompts": int(cross.S)}


# ══════════════════════════════════════════════════════════════════
# reporting
# ══════════════════════════════════════════════════════════════════

def _f(x, nd=3):
    if x is None or not np.isfinite(x):
        return "--"
    return ("%." + str(nd) + "f") % x


def build_markdown(results, ranks):
    lines = ["# E1: held-out calibration validation and bootstrap CIs", ""]
    lines.append("Fit split = published calibration prompts; eval split = disjoint held-out prompts.")
    lines.append("Residual = 1 - captured/total energy of the width-R student "
                 "(fixed message maps, free routes). Values are medians over layers.")
    lines.append("")

    lines.append("## (a) Held-out split")
    lines.append("")
    headers = ["Model", "H", "L", "fit", "eval", "r_eff (fit)", "r_eff (eval)"]
    for R in ranks:
        headers += ["T_%d in-sample" % R, "R=%d held-out" % R, "gap"]
    rows = []
    for m in results["models"]:
        h = m.get("heldout")
        if not h:
            continue
        row = [m["model"], m["H"], m["L"], m["fit_range"], m["eval_range"],
               _f(h["summary"]["median"]["r_eff_fit"], 2),
               _f(h["summary"]["median"]["r_eff_eval"], 2)]
        for R in ranks:
            k = str(R)
            row += [_f(h["summary"]["median"]["in_sample_T_R"][k]),
                    _f(h["summary"]["median"]["heldout_residual"][k]),
                    _f(h["summary"]["median"]["gap"][k])]
        rows.append(row)
    lines.append(md_table(headers, rows) if rows else "_no held-out split available_")
    lines.append("")

    lines.append("## (b) Bootstrap CIs (median over layers, 2.5/97.5 pct)")
    lines.append("")
    headers = ["Model", "reps", "subset", "r_eff point", "r_eff mean", "r_eff 95% CI",
               "T_8 point", "T_8 mean", "T_8 95% CI"]
    rows = []
    for m in results["models"]:
        b = m.get("bootstrap")
        if not b:
            continue
        t8 = b["tail"].get("T_8", {})
        t8p = b["point_estimate"]["tail"].get("T_8")
        rows.append([m["model"], b["n_reps"], b["subset_size"],
                     _f(b["point_estimate"]["r_eff"], 2), _f(b["r_eff"]["mean"], 2),
                     "[%s, %s]" % (_f(b["r_eff"]["ci_lo"], 2), _f(b["r_eff"]["ci_hi"], 2)),
                     _f(t8p), _f(t8.get("mean")),
                     "[%s, %s]" % (_f(t8.get("ci_lo")), _f(t8.get("ci_hi")))])
    lines.append(md_table(headers, rows) if rows else "_no bootstrap run_")
    lines.append("")

    lines.append("## (c) Cross-domain transfer (fit on base dump, score on another corpus)")
    lines.append("")
    headers = ["Model", "fit corpus", "eval corpus", "r_eff (eval corpus)"]
    for R in ranks:
        headers += ["T_%d in-sample" % R, "R=%d cross" % R, "R=%d oracle" % R]
    rows = []
    for m in results["models"]:
        for c in m.get("cross_domain", []):
            row = [m["model"], m["dataset"], c["cross_dataset"],
                   _f(c["summary"]["median"]["r_eff_cross"], 2)]
            for R in ranks:
                k = str(R)
                row += [_f(c["summary"]["median"]["in_sample_T_R"][k]),
                        _f(c["summary"]["median"]["cross_residual"][k]),
                        _f(c["summary"]["median"]["cross_oracle_T_R"][k])]
            rows.append(row)
    lines.append(md_table(headers, rows) if rows else "_no cross-domain dumps supplied_")
    lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--npz", nargs="*", default=[],
                    help="base dumps (one per model; normally the c4 dumps)")
    ap.add_argument("--cross", nargs="*", default=[],
                    help="cross-domain dumps; matched to a base dump by (model, seq_len)")
    ap.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    ap.add_argument("--fit-range", default="0:200", help="prompt slice used for fitting")
    ap.add_argument("--eval-range", default="200:400", help="held-out prompt slice")
    ap.add_argument("--bootstrap-reps", type=int, default=500)
    ap.add_argument("--bootstrap-size", type=int, default=200)
    ap.add_argument("--bootstrap-pool", default="all",
                    help="prompt slice resampled by the bootstrap (default: all prompts)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--out-json", default=os.path.join(_ROOT, "results", "analysis", "e1_heldout.json"))
    ap.add_argument("--out-md", default=os.path.join(_ROOT, "results", "analysis", "e1_heldout.md"))
    ap.add_argument("--self-test", action="store_true", help="run property tests and exit")
    args = ap.parse_args()

    if args.self_test:
        res = property_tests(verbose=True)
        bad = [n for n, ok, _ in res if not ok]
        print("%d/%d property tests passed" % (len(res) - len(bad), len(res)))
        return 1 if bad else 0

    if not args.npz:
        ap.error("--npz is required (or use --self-test)")

    print("property tests (always run before analysis):")
    res = property_tests(verbose=True)
    if any(not ok for _, ok, _ in res):
        print("PROPERTY TESTS FAILED -- refusing to report numbers")
        return 1
    print("")

    ranks = sorted(set(int(r) for r in args.ranks))
    cross_dumps = [load_dump(p) for p in args.cross]
    results = {"ranks": ranks, "fit_range": args.fit_range, "eval_range": args.eval_range,
               "models": [], "property_tests": [{"name": n, "ok": bool(ok), "value": float(v)}
                                                for n, ok, v in res]}

    for path in args.npz:
        t0 = time.time()
        d = load_dump(path)
        print("=" * 60)
        print("%s  (H=%d, L=%d, S=%d)%s" % (d.label(), d.H, d.L, d.S,
                                          "" if d.per_prompt else "  [G_P_total only:"
                                          " no per-prompt splits]"))
        fit_idx = parse_range(args.fit_range, d.S)
        eval_idx = parse_range(args.eval_range, d.S)
        pool_idx = parse_range(args.bootstrap_pool, d.S)
        overlap = np.intersect1d(fit_idx, eval_idx)
        if overlap.size:
            print("  [warn] fit and eval ranges overlap in %d prompts" % overlap.size)

        entry = {"model": d.model, "dataset": d.dataset, "seq_len": d.seq_len,
                 "path": os.path.abspath(path), "H": d.H, "L": d.L, "S": d.S,
                 "num_kv_heads": d.config.get("num_kv_heads"),
                 "fit_range": args.fit_range, "eval_range": args.eval_range,
                 "n_fit": int(fit_idx.size), "n_eval": int(eval_idx.size)}

        if eval_idx.size == 0:
            print("  [skip] no held-out prompts in %s (S=%d)" % (args.eval_range, d.S))
        elif fit_idx.size == 0:
            print("  [skip] no fitting prompts in %s" % args.fit_range)
        else:
            print("  held-out split: fit %d prompts, eval %d prompts" % (fit_idx.size, eval_idx.size))
            entry["heldout"] = analyze_heldout_split(d, fit_idx, eval_idx, ranks)
            med = entry["heldout"]["summary"]["median"]
            for R in ranks:
                print("    R=%-2d  in-sample T_R=%.4f   held-out=%.4f   gap=%+.4f"
                      % (R, med["in_sample_T_R"][str(R)], med["heldout_residual"][str(R)],
                         med["gap"][str(R)]))

        if not args.no_bootstrap and d.S > 1:
            size = min(args.bootstrap_size, pool_idx.size) if pool_idx.size else 0
            if size > 0:
                print("  bootstrap: %d reps x %d prompts from a pool of %d"
                      % (args.bootstrap_reps, size, pool_idx.size))
                entry["bootstrap"] = analyze_bootstrap(
                    d, pool_idx, args.bootstrap_reps, size, args.seed, tail_R=8, ranks=ranks)
                b = entry["bootstrap"]
                print("    r_eff = %.2f  95%% CI [%.2f, %.2f]"
                      % (b["point_estimate"]["r_eff"], b["r_eff"]["ci_lo"], b["r_eff"]["ci_hi"]))
                if "T_8" in b["tail"]:
                    print("    T_8   = %.4f  95%% CI [%.4f, %.4f]"
                          % (b["point_estimate"]["tail"]["T_8"], b["tail"]["T_8"]["ci_lo"],
                             b["tail"]["T_8"]["ci_hi"]))

        entry["cross_domain"] = []
        for c in cross_dumps:
            if c.key() != d.key() or c.dataset == d.dataset:
                continue
            print("  cross-domain: fit on %s, score on %s (%d prompts)"
                  % (d.dataset, c.dataset, c.S))
            cd = analyze_cross_domain(d, c, fit_idx if fit_idx.size else None, ranks)
            entry["cross_domain"].append(cd)
            for R in ranks:
                print("    R=%-2d  in-sample T_R=%.4f   cross=%.4f   oracle=%.4f"
                      % (R, cd["summary"]["median"]["in_sample_T_R"][str(R)],
                         cd["summary"]["median"]["cross_residual"][str(R)],
                         cd["summary"]["median"]["cross_oracle_T_R"][str(R)]))

        entry["elapsed_seconds"] = time.time() - t0
        results["models"].append(entry)
        print("  done in %.1fs" % entry["elapsed_seconds"])

    write_json(args.out_json, results)
    write_text(args.out_md, build_markdown(results, ranks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
