#!/usr/bin/env python3
"""
E4a: route-merging (PatternMerge) comparison.

Offline, pure numpy, CPU.  Consumes the same .npz dumps as analyze_heldout.py
(schema in spectral_common.py) and tests whether the spectrum is more
informative than a simpler attention-redundancy measure when deciding which
heads may share a route.

THE MATH
--------
A PatternMerge map is a function phi: [H] -> [H] whose image is a set of R
representative heads.  The merged model reuses the ROUTE of the representative
while keeping the ORIGINAL message map of each head, i.e. column phi(h) of A
paired with column h of B:

    Psitilde = A Pi_phi B^T,   Pi_phi[i,h] = 1 iff i = phi(h).

Calibration error and its reported fraction are therefore

    error(phi)    = ||Psi - Psitilde||_F^2 = tr(C^T G_P C G_M),  C = I - Pi_phi
    fraction(phi) = error(phi) / tr(G_P G_M).

Because rank(Pi_phi) = R, every phi with R representatives obeys
error(phi) >= sum_{j>R} sigma_j^2 (Eckart-Young), so the signed rank-R tail
energy T_R that the paper reports is an oracle lower bound for ANY merging
scheme -- spectrum-guided or not.  That inequality is property-tested below.

METHODS COMPARED (per layer, per R)
-----------------------------------
  oracle            rank-R tail energy from the eigendecomposition (lower bound)
  greedy            spectrum-guided: greedy forward selection of representatives
                    on the exact closed-form objective
  eigen             spectrum-guided: representatives read off the top-R
                    eigenbasis loadings, assignment by nearest representative
                    under the G_M-weighted route metric
  similarity        routing-only similarity baseline: average-linkage
                    clustering on the cosine similarity of G_P columns ONLY
                    (G_M never enters), representative = cluster medoid
  similarity_gram   same idea but on the direct route cosine
                    G_P[h,h']/sqrt(G_P[h,h]G_P[h',h']) (reported for context)
  random            uniform representatives + uniform assignment, >=1000 draws
  random_nearest    uniform representatives + nearest-representative assignment
                    (a much stronger random control)

Held-out and cross-domain variants select phi on one set of Grams and score it
on another; the oracle bound is always computed on the SCORING Grams, where it
remains valid because A_B Pi_phi B^T still has rank <= R.

Usage (from the repository root; dumps come from src/measure/run_batch.py)
-----
    python src/analysis/analyze_merge.py \
        --npz results/measure/npz/qwen3-8b__c4__N512__S400__fp16.npz
    python src/analysis/analyze_merge.py --self-test
--cross <dumps> adds a cross-domain variant (not used in the paper).
Outputs default to results/analysis/e4a_merge.{json,md}.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root

from spectral_common import (  # noqa: E402
    assert_same_message_gram, load_dump, md_table, median, merge_error,
    merge_error_batch, merge_error_fraction, merge_matrix, parse_range, pct,
    phi_rank, psd_sqrt, spectrum_sigma_sq, sym, tail_fraction, total_energy,
    write_json, write_text,
)

DEFAULT_RANKS = [1, 2, 4, 8, 16]
SPECTRUM_METHODS = ["greedy", "eigen"]
DETERMINISTIC_METHODS = ["greedy", "eigen", "similarity", "similarity_gram"]
RANDOM_METHODS = ["random", "random_nearest"]
HEADLINE_SPECTRUM = "greedy"


# ══════════════════════════════════════════════════════════════════
# phi constructors
# ══════════════════════════════════════════════════════════════════

def assign_nearest(GP, GM, reps):
    """
    Assign every head to the representative closest under the G_M-weighted
    route metric  d(h,i) = G_M[h,h] * ||a_h - a_i||^2
                         = G_M[h,h] * (G_P[h,h] - 2 G_P[h,i] + G_P[i,i]).
    Representatives always map to themselves, so the image has exactly R heads.
    Ties break toward the earliest representative (deterministic).
    """
    reps = np.asarray(list(reps), dtype=int)
    diag = np.diag(GP)
    w = np.maximum(np.diag(GM), 0.0)
    d = w[:, None] * (diag[:, None] - 2.0 * GP[:, reps] + diag[reps][None, :])
    phi = reps[np.argmin(d, axis=1)]
    phi[reps] = reps
    return phi


def greedy_path(GP, GM, ranks):
    """
    Spectrum-guided variant 1: greedy forward selection of representatives.

    At each step every not-yet-selected head is tried as an additional
    representative, the induced assignment is built with assign_nearest, and the
    candidate minimizing the exact closed-form error tr(C^T G_P C G_M) is kept.
    Strictly-less comparison makes ties resolve to the lowest head index.

    The selection is nested (reps for R are a prefix of reps for R+1), so one
    pass to max(ranks) yields every requested width; that also makes the greedy
    error curve monotone nonincreasing in R.  Returns {R: (phi, reps)}.
    """
    H = GP.shape[0]
    want = sorted(set(int(r) for r in ranks))
    Rmax = max(want)
    reps, snaps = [], {}
    for _ in range(Rmax):
        cands = [c for c in range(H) if c not in reps]
        phis = np.stack([assign_nearest(GP, GM, reps + [c]) for c in cands])
        errs = merge_error_batch(GP, GM, phis)
        k = int(np.argmin(errs))            # argmin keeps the lowest head index on ties
        reps.append(cands[k])
        if len(reps) in want:
            snaps[len(reps)] = (phis[k].copy(), list(reps))
    return snaps


def greedy_phi(GP, GM, R):
    return greedy_path(GP, GM, [int(R)])[int(R)]


def eigen_path(GP, GM, ranks):
    """
    Spectrum-guided variant 2: representatives from top-R eigenbasis loadings.

    With D = diag(sqrt(diag(G_M))) the merge error under a diagonal G_M is
    sum_h G_M[h,h] ||a_h - a_phi(h)||^2, whose head geometry is exactly the
    Gram D G_P D.  Head h's loading on component j is sqrt(mu_j) |V[h,j]|; the
    representative of component j is the unclaimed head with the largest
    loading.  Assignment then uses the same G_M-weighted nearest rule.
    Representative selection is nested in R.  Returns {R: (phi, reps)}.
    """
    want = sorted(set(int(r) for r in ranks))
    Rmax = max(want)
    d = np.sqrt(np.maximum(np.diag(GM), 0.0))
    K = sym((GP * d[None, :]) * d[:, None])
    mu, V = np.linalg.eigh(K)
    order = np.argsort(mu)[::-1]
    mu, V = np.maximum(mu[order], 0.0), V[:, order]
    reps, snaps = [], {}
    for j in range(Rmax):
        jj = min(j, V.shape[1] - 1)
        load = np.abs(V[:, jj]) * np.sqrt(mu[jj])
        for h in np.argsort(-load, kind="stable"):
            if int(h) not in reps:
                reps.append(int(h))
                break
        if len(reps) in want:
            snaps[len(reps)] = (assign_nearest(GP, GM, reps), list(reps))
    return snaps


def eigen_phi(GP, GM, R):
    return eigen_path(GP, GM, [int(R)])[int(R)]


def cosine_columns(GP):
    """Cosine similarity between the COLUMNS of G_P (the routing-only similarity baseline)."""
    X = np.asarray(GP, dtype=np.float64)
    n = np.linalg.norm(X, axis=0)
    n = np.where(n > 0, n, 1.0)
    Xn = X / n
    return np.clip(Xn.T @ Xn, -1.0, 1.0)


def cosine_routes(GP):
    """Cosine between the attention maps themselves: G_P[h,h']/sqrt(G_P[hh]G_P[h'h'])."""
    d = np.sqrt(np.maximum(np.diag(GP), 0.0))
    d = np.where(d > 0, d, 1.0)
    return np.clip(GP / np.outer(d, d), -1.0, 1.0)


def agglomerative_snapshots(D, ranks):
    """
    Deterministic average-linkage (UPGMA) clustering, one pass, snapshotted at
    every requested cluster count.  Uses the Lance-Williams update
        d(i+j, k) = (n_i d(i,k) + n_j d(j,k)) / (n_i + n_j),
    which reproduces the mean pairwise distance exactly.  The closest pair is
    taken with a flat argmin, so ties resolve to the lowest (i, j).
    """
    want = set(int(r) for r in ranks)
    H = D.shape[0]
    Dc = np.array(D, dtype=np.float64, copy=True)
    np.fill_diagonal(Dc, np.inf)
    sizes = np.ones(H)
    members = {i: [i] for i in range(H)}
    alive = list(range(H))
    snaps = {}
    if H in want:
        snaps[H] = [list(members[i]) for i in alive]
    for _ in range(H - 1):
        flat = int(np.argmin(Dc))
        i, j = divmod(flat, H)
        if i > j:
            i, j = j, i
        if not np.isfinite(Dc[i, j]):
            break
        new = (sizes[i] * Dc[i, :] + sizes[j] * Dc[j, :]) / (sizes[i] + sizes[j])
        Dc[i, :] = new
        Dc[:, i] = new
        Dc[i, i] = np.inf
        Dc[j, :] = np.inf
        Dc[:, j] = np.inf
        sizes[i] += sizes[j]
        members[i] = members[i] + members[j]
        del members[j]
        alive.remove(j)
        if len(alive) in want:
            snaps[len(alive)] = [list(members[a]) for a in alive]
    return snaps


def _clusters_to_phi(D, clusters, H):
    phi = np.zeros(H, dtype=int)
    reps = []
    for c in clusters:
        sub = D[np.ix_(c, c)].sum(axis=1)
        rep = int(c[int(np.argmin(sub))])   # medoid, ties -> lowest index in cluster
        reps.append(rep)
        for h in c:
            phi[h] = rep
    return phi, sorted(reps)


def similarity_path(GP, ranks, mode="columns"):
    """
    Attention-similarity baseline: cluster heads on route similarity ONLY
    (G_M is never touched), representative = cluster medoid.
    Returns {R: (phi, reps)}.
    """
    sim = cosine_columns(GP) if mode == "columns" else cosine_routes(GP)
    D = 1.0 - sim
    np.fill_diagonal(D, 0.0)
    H = GP.shape[0]
    snaps = agglomerative_snapshots(D, ranks)
    return {R: _clusters_to_phi(D, cl, H) for R, cl in snaps.items()}


def similarity_phi(GP, R, mode="columns"):
    return similarity_path(GP, [int(R)], mode)[int(R)]


def random_phis(rng, H, R, B, GP=None, GM=None, mode="uniform"):
    """
    B random maps at once, returned as [B, H].

    Representatives are a uniform R-subset of the heads.  mode="uniform"
    assigns every remaining head to a uniformly random representative (the
    naive control); mode="nearest" keeps the same G_M-weighted nearest-
    representative assignment as the guided methods, which is a much stronger
    control because only the representative CHOICE is random.
    """
    R = int(R)
    reps = np.sort(np.argsort(rng.random((B, H)), axis=1)[:, :R], axis=1)   # [B, R]
    rows = np.arange(B)[:, None]
    if mode == "nearest":
        diag = np.diag(GP)
        w = np.maximum(np.diag(GM), 0.0)
        gpr = GP[:, reps.reshape(-1)].reshape(H, B, R).transpose(1, 0, 2)   # [B, H, R]
        d = w[None, :, None] * (diag[None, :, None] - 2.0 * gpr + diag[reps][:, None, :])
        phi = np.take_along_axis(reps, np.argmin(d, axis=2), axis=1)
    else:
        phi = np.take_along_axis(reps, rng.integers(0, R, size=(B, H)), axis=1)
    phi[rows, reps] = reps       # representatives always map to themselves
    return phi


def random_phi(rng, H, R, GP=None, GM=None, mode="uniform"):
    """Single random phi (thin wrapper used by the property tests)."""
    phi = random_phis(rng, H, R, 1, GP, GM, mode)[0]
    return phi, sorted(set(int(x) for x in phi))


# ══════════════════════════════════════════════════════════════════
# property tests
# ══════════════════════════════════════════════════════════════════

def _case(seed=0, H=6, rows_A=40, rows_B=30):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((rows_A, H))
    B = rng.standard_normal((rows_B, H))
    return A, B, A @ B.T


def test_identity_phi_zero(tol=1e-12, verbose=True):
    """(i) identity phi merges nothing -> error exactly 0."""
    worst = 0.0
    for seed in range(4):
        A, B, _ = _case(seed=seed)
        GP, GM = A.T @ A, B.T @ B
        worst = max(worst, abs(merge_error(GP, GM, np.arange(A.shape[1]))))
    ok = worst < tol
    if verbose:
        print("  [i]   identity phi error: max |err| = %.3e -> %s" % (worst, "PASS" if ok else "FAIL"))
    return ok, worst


def test_explicit_merge_error(tol=1e-8, verbose=True):
    """(ii) explicit Psi and merged Psitilde: Frobenius error == closed form (abs and rel)."""
    worst_abs, worst_rel = 0.0, 0.0
    rng = np.random.default_rng(5)
    for seed in range(4):
        A, B, Psi = _case(seed=seed)
        H = A.shape[1]
        GP, GM = A.T @ A, B.T @ B
        phis = [np.array([0, 0, 2, 2, 4, 4]), np.zeros(H, dtype=int), np.arange(H)]
        phis += [rng.integers(0, H, size=H) for _ in range(6)]
        for phi in phis:
            Pi = merge_matrix(phi, H)
            explicit = float(np.linalg.norm(Psi - A @ Pi @ B.T) ** 2)
            C = np.eye(H) - Pi
            trace_form = float(np.trace(C.T @ GP @ C @ GM))
            index_form = merge_error(GP, GM, phi)
            for other in (trace_form, index_form):
                worst_abs = max(worst_abs, abs(other - explicit))
                worst_rel = max(worst_rel, abs(other - explicit) / max(explicit, 1.0))
            # fraction consistency
            frac = merge_error_fraction(GP, GM, phi)
            worst_rel = max(worst_rel, abs(frac - explicit / float(np.linalg.norm(Psi) ** 2)))
    ok = worst_abs < tol and worst_rel < tol
    if verbose:
        print("  [ii]  explicit ||Psi - Psitilde||_F^2 vs tr(C^T G_P C G_M): abs %.3e / rel %.3e "
              "(tol %.0e) -> %s" % (worst_abs, worst_rel, tol, "PASS" if ok else "FAIL"))
    return ok, max(worst_abs, worst_rel)


def test_batch_matches_scalar(tol=1e-8, verbose=True):
    """(iii') the batched closed form must agree with the per-phi index form."""
    rng = np.random.default_rng(13)
    A, B, _ = _case(seed=6, H=8, rows_A=50, rows_B=40)
    GP, GM = A.T @ A, B.T @ B
    PHI = np.stack([random_phi(rng, 8, R)[0] for R in (1, 2, 3, 4, 5, 6, 7)]
                   + [np.arange(8), np.zeros(8, dtype=int)])
    batch = merge_error_batch(GP, GM, PHI)
    scalar = np.array([merge_error(GP, GM, p) for p in PHI])
    denom = np.maximum(np.abs(scalar), 1.0)
    worst = float((np.abs(batch - scalar) / denom).max())
    ok = worst < tol
    if verbose:
        print("  [iii'] merge_error_batch == merge_error: max rel diff = %.3e -> %s"
              % (worst, "PASS" if ok else "FAIL"))
    return ok, worst


def test_error_above_rank_tail(tol=1e-10, verbose=True):
    """(iii) every phi with R representatives must cost at least the rank-R tail."""
    rng = np.random.default_rng(9)
    worst = -np.inf   # max of (bound - fraction); must stay <= 0
    checked = 0
    for seed in range(3):
        A, B, _ = _case(seed=seed, H=8, rows_A=50, rows_B=40)
        GP, GM = A.T @ A, B.T @ B
        ss = spectrum_sigma_sq(GP, GM)
        for R in (1, 2, 4, 6):
            bound = tail_fraction(ss, R)
            cands = [greedy_phi(GP, GM, R)[0], eigen_phi(GP, GM, R)[0],
                     similarity_phi(GP, R, "columns")[0], similarity_phi(GP, R, "routes")[0]]
            cands += [random_phi(rng, 8, R)[0] for _ in range(50)]
            cands += [random_phi(rng, 8, R, GP, GM, "nearest")[0] for _ in range(50)]
            for phi in cands:
                assert phi_rank(phi) == R, "phi image size %d != R %d" % (phi_rank(phi), R)
                frac = merge_error_fraction(GP, GM, phi)
                worst = max(worst, bound - frac)   # want <= 0
                checked += 1
    ok = worst < tol
    if verbose:
        print("  [iii] error(phi) >= rank-R tail over %d phis: max violation = %.3e (tol %.0e) -> %s"
              % (checked, worst, tol, "PASS" if ok else "FAIL"))
    return ok, worst


def test_heldout_bound_still_holds(tol=1e-10, verbose=True):
    """Extra: phi picked on split A, scored on split B, still >= split-B rank-R tail."""
    rng = np.random.default_rng(21)
    worst = -np.inf
    A, B, _ = _case(seed=2, H=8, rows_A=60, rows_B=40)
    perm = rng.permutation(A.shape[0])
    iA, iB = perm[:30], perm[30:]
    GP_A, GP_B, GM = A[iA].T @ A[iA], A[iB].T @ A[iB], B.T @ B
    ssB = spectrum_sigma_sq(GP_B, GM)
    for R in (1, 2, 4):
        bound = tail_fraction(ssB, R)
        for phi in (greedy_phi(GP_A, GM, R)[0], eigen_phi(GP_A, GM, R)[0],
                    similarity_phi(GP_A, R)[0]):
            worst = max(worst, bound - merge_error_fraction(GP_B, GM, phi))
    ok = worst < tol
    if verbose:
        print("  [iv]  held-out selection still >= split-B tail: max violation = %.3e -> %s"
              % (worst, "PASS" if ok else "FAIL"))
    return ok, worst


def test_methods_are_deterministic(tol=0.0, verbose=True):
    """Extra: the deterministic methods really are deterministic and give R distinct reps."""
    A, B, _ = _case(seed=4, H=8, rows_A=50, rows_B=40)
    GP, GM = A.T @ A, B.T @ B
    ok = True
    for R in (1, 2, 4, 6):
        for fn in (lambda: greedy_phi(GP, GM, R)[0], lambda: eigen_phi(GP, GM, R)[0],
                   lambda: similarity_phi(GP, R)[0]):
            a, b = fn(), fn()
            ok = ok and np.array_equal(a, b) and phi_rank(a) == R
    if verbose:
        print("  [v]   deterministic methods reproducible with |image| = R -> %s"
              % ("PASS" if ok else "FAIL"))
    return ok, 0.0


def test_greedy_beats_random_on_average(tol=0.0, verbose=True):
    """Extra sanity: on synthetic data the spectrum-guided phi should beat mean random."""
    rng = np.random.default_rng(33)
    A, B, _ = _case(seed=1, H=8, rows_A=60, rows_B=40)
    GP, GM = A.T @ A, B.T @ B
    ok = True
    for R in (2, 4):
        g = merge_error_fraction(GP, GM, greedy_phi(GP, GM, R)[0])
        rnd = [merge_error_fraction(GP, GM, random_phi(rng, 8, R)[0]) for _ in range(200)]
        ok = ok and g <= float(np.mean(rnd))
    if verbose:
        print("  [vi]  greedy <= mean(random) on synthetic data -> %s" % ("PASS" if ok else "FAIL"))
    return ok, 0.0


def property_tests(verbose=True):
    if verbose:
        print("analyze_merge property tests")
    return [
        ("identity phi -> zero error", ) + test_identity_phi_zero(verbose=verbose),
        ("explicit merged Frobenius == closed form", ) + test_explicit_merge_error(verbose=verbose),
        ("batched == scalar merge error", ) + test_batch_matches_scalar(verbose=verbose),
        ("error(phi) >= rank-R tail", ) + test_error_above_rank_tail(verbose=verbose),
        ("held-out selection >= scoring-split tail", ) + test_heldout_bound_still_holds(verbose=verbose),
        ("deterministic methods", ) + test_methods_are_deterministic(verbose=verbose),
        ("greedy <= mean random", ) + test_greedy_beats_random_on_average(verbose=verbose),
    ]


# ══════════════════════════════════════════════════════════════════
# per-layer comparison
# ══════════════════════════════════════════════════════════════════

def compare_layer(GP_sel, GP_score, GM, ranks, n_random, seed, GMh=None):
    """
    Build phi on GP_sel, score every method on GP_score.

    Returns {R: {method: fraction, ...}} plus random statistics.  When
    GP_sel is GP_score this is the in-sample comparison.  All deterministic
    methods are computed with one nested pass over max(ranks); the random
    draws are scored with the batched closed form.
    """
    H = GP_sel.shape[0]
    ranks = [int(R) for R in ranks if int(R) < H]
    if not ranks:
        return {}
    ss_score = spectrum_sigma_sq(GP_score, GM, GMh)
    tot_score = total_energy(GP_score, GM)
    scale = 1.0 / tot_score if tot_score > 0 else 0.0

    paths = {
        "greedy": greedy_path(GP_sel, GM, ranks),
        "eigen": eigen_path(GP_sel, GM, ranks),
        "similarity": similarity_path(GP_sel, ranks, "columns"),
        "similarity_gram": similarity_path(GP_sel, ranks, "routes"),
    }

    out = {}
    for R in ranks:
        rec = {"R": int(R), "oracle": tail_fraction(ss_score, R), "methods": {}, "reps": {}}
        for name, path in paths.items():
            if R not in path:
                continue
            phi, reps = path[R]
            rec["methods"][name] = merge_error(GP_score, GM, phi) * scale
            rec["reps"][name] = [int(x) for x in reps]

        rng = np.random.default_rng(seed + 1000 * int(R))
        for mode, name in (("uniform", "random"), ("nearest", "random_nearest")):
            PHI = random_phis(rng, H, R, n_random, GP_sel, GM, mode)
            vals = merge_error_batch(GP_score, GM, PHI) * scale
            rec[name] = {"mean": float(vals.mean()), "min": float(vals.min()),
                         "p5": pct(vals, 5), "p50": pct(vals, 50), "p95": pct(vals, 95),
                         "max": float(vals.max()), "n": int(n_random)}
            rec["methods"][name + "_mean"] = float(vals.mean())
            rec["methods"][name + "_min"] = float(vals.min())
            # empirical p: how often does a random draw match or beat the headline method?
            hl = rec["methods"][HEADLINE_SPECTRUM]
            rec[name]["p_random_better_or_equal"] = float(np.mean(vals <= hl + 1e-15))
        out[int(R)] = rec
    return out


def analyze_dump(dump, ranks, n_random, seed, sel_idx=None, score_idx=None,
                 score_dump=None, tag="in_sample"):
    """Run compare_layer over all layers of a dump (optionally cross-split/cross-domain)."""
    score_dump = score_dump or dump
    layers = []
    for l in range(dump.L):
        GM = dump.gm(l)
        GMh, _, _ = psd_sqrt(GM)
        GP_sel = dump.gp_sum(l, sel_idx)
        GP_score = score_dump.gp_sum(l, score_idx)
        layers.append({"layer": dump.layer_id(l),
                       "ranks": compare_layer(GP_sel, GP_score, GM, ranks, n_random, seed, GMh)})
    return {"tag": tag, "layers": layers, "summary": summarize(layers, ranks)}


def summarize(layers, ranks):
    """Per-R medians over layers, win/loss counts, and 'beats random' rates."""
    methods = DETERMINISTIC_METHODS + [m + "_mean" for m in RANDOM_METHODS] \
        + [m + "_min" for m in RANDOM_METHODS]
    summary = {}
    for R in ranks:
        rows = [x["ranks"][R] for x in layers if R in x["ranks"]]
        if not rows:
            continue
        s = {"n_layers": len(rows),
             "oracle": median([r["oracle"] for r in rows]),
             "median": {}, "mean": {}}
        for m in methods:
            vals = [r["methods"][m] for r in rows if m in r["methods"]]
            if vals:
                s["median"][m] = median(vals)
                s["mean"][m] = float(np.mean(vals))
        # spectrum vs the routing-only similarity baseline
        for spec in SPECTRUM_METHODS:
            w = l_ = t = 0
            for r in rows:
                a, b = r["methods"][spec], r["methods"]["similarity"]
                if a < b * (1 - 1e-12):
                    w += 1
                elif a > b * (1 + 1e-12):
                    l_ += 1
                else:
                    t += 1
            s.setdefault("vs_similarity", {})[spec] = {
                "wins": w, "losses": l_, "ties": t,
                "win_rate": w / float(len(rows)),
                "median_relative_improvement": median(
                    [(r["methods"]["similarity"] - r["methods"][spec]) /
                     r["methods"]["similarity"] if r["methods"]["similarity"] > 0 else 0.0
                     for r in rows]),
            }
        # spectrum vs random
        for rnd in RANDOM_METHODS:
            beats_mean = sum(1 for r in rows if r["methods"][HEADLINE_SPECTRUM] < r["methods"][rnd + "_mean"])
            beats_best = sum(1 for r in rows if r["methods"][HEADLINE_SPECTRUM] <= r["methods"][rnd + "_min"] + 1e-15)
            s.setdefault("vs_random", {})[rnd] = {
                "n_draws": int(rows[0][rnd]["n"]),
                "layers_beating_random_mean": beats_mean,
                "frac_layers_beating_random_mean": beats_mean / float(len(rows)),
                "layers_beating_best_draw": beats_best,
                "frac_layers_beating_best_draw": beats_best / float(len(rows)),
                "median_p_random_better_or_equal": median([r[rnd]["p_random_better_or_equal"] for r in rows]),
            }
        # excess over the oracle lower bound
        s["excess_over_oracle"] = {
            m: median([r["methods"][m] - r["oracle"] for r in rows if m in r["methods"]])
            for m in methods
        }
        summary[int(R)] = s
    return summary


# ══════════════════════════════════════════════════════════════════
# reporting
# ══════════════════════════════════════════════════════════════════

def _f(x, nd=4):
    if x is None or not np.isfinite(x):
        return "--"
    return ("%." + str(nd) + "f") % x


def _section(title, entries, ranks):
    lines = ["## %s" % title, ""]
    if not entries:
        return lines + ["_not available_", ""]
    headers = ["Model", "R", "oracle", "greedy", "eigen", "similarity", "sim(gram)",
               "random mean", "random best", "rand-near mean"]
    rows = []
    for model, an in entries:
        for R in ranks:
            s = an["summary"].get(R)
            if not s:
                continue
            rows.append([model, R, _f(s["oracle"]), _f(s["median"].get("greedy")),
                         _f(s["median"].get("eigen")), _f(s["median"].get("similarity")),
                         _f(s["median"].get("similarity_gram")),
                         _f(s["median"].get("random_mean")), _f(s["median"].get("random_min")),
                         _f(s["median"].get("random_nearest_mean"))])
    lines.append(md_table(headers, rows))
    lines.append("")
    lines.append("Win/loss of spectrum-guided vs the attention-similarity baseline, and how "
                 "often the spectrum-guided phi beats random:")
    lines.append("")
    headers = ["Model", "R", "greedy W-L-T vs sim", "greedy win rate",
               "median rel. improvement", "beats random mean", "beats best random draw",
               "median p(random <= greedy)"]
    rows = []
    for model, an in entries:
        for R in ranks:
            s = an["summary"].get(R)
            if not s:
                continue
            v = s["vs_similarity"]["greedy"]
            rr = s["vs_random"]["random"]
            rows.append([model, R, "%d-%d-%d" % (v["wins"], v["losses"], v["ties"]),
                         "%.0f%%" % (100 * v["win_rate"]),
                         _f(v["median_relative_improvement"], 3),
                         "%.0f%% of layers" % (100 * rr["frac_layers_beating_random_mean"]),
                         "%.0f%% of layers" % (100 * rr["frac_layers_beating_best_draw"]),
                         _f(rr["median_p_random_better_or_equal"], 4)])
    lines.append(md_table(headers, rows))
    lines.append("")
    return lines


def build_markdown(results, ranks):
    lines = ["# E4a: route merging -- spectrum-guided vs simpler redundancy measures", ""]
    lines.append("Entries are the layer-median calibration error fraction "
                 "tr(C^T G_P C G_M) / tr(G_P G_M) at merge width R "
                 "(lower is better; `oracle` is the rank-R tail energy, an unattainable "
                 "lower bound for any merging map).")
    lines.append("")
    for tag, title in (("in_sample", "In-sample (select and score on the same calibration set)"),
                       ("heldout", "Held-out (phi selected on the fit split, scored on the held-out split)"),
                       ("cross_domain", "Cross-domain (phi selected on the base corpus, scored on another corpus)")):
        entries = []
        for m in results["models"]:
            for an in m.get("analyses", []):
                if an["tag"].startswith(tag):
                    name = m["model"] if an["tag"] == tag else "%s (%s)" % (m["model"], an["tag"].split(":")[-1])
                    entries.append((name, an))
        lines += _section(title, entries, ranks)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--npz", nargs="*", default=[])
    ap.add_argument("--cross", nargs="*", default=[])
    ap.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    ap.add_argument("--fit-range", default="0:200")
    ap.add_argument("--eval-range", default="200:400")
    ap.add_argument("--random-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-heldout", action="store_true")
    ap.add_argument("--out-json", default=os.path.join(_ROOT, "results", "analysis", "e4a_merge.json"))
    ap.add_argument("--out-md", default=os.path.join(_ROOT, "results", "analysis", "e4a_merge.md"))
    ap.add_argument("--self-test", action="store_true")
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
    results = {"ranks": ranks, "random_draws": args.random_draws, "seed": args.seed,
               "models": [],
               "property_tests": [{"name": n, "ok": bool(ok), "value": float(v)} for n, ok, v in res]}

    for path in args.npz:
        t0 = time.time()
        d = load_dump(path)
        print("=" * 60)
        print("%s  (H=%d, L=%d, S=%d)%s" % (d.label(), d.H, d.L, d.S,
                                          "" if d.per_prompt else "  [G_P_total only:"
                                          " no per-prompt splits]"))
        usable = [R for R in ranks if R < d.H]
        if not usable:
            print("  [skip] no rank R < H=%d among %s" % (d.H, ranks))
            continue
        fit_idx = parse_range(args.fit_range, d.S)
        eval_idx = parse_range(args.eval_range, d.S)

        entry = {"model": d.model, "dataset": d.dataset, "seq_len": d.seq_len,
                 "path": os.path.abspath(path), "H": d.H, "L": d.L, "S": d.S,
                 "ranks": usable, "analyses": []}

        print("  in-sample (%s)" % ("all %d prompts" % d.S if d.per_prompt
                                   else "full accumulated calibration set"))
        entry["analyses"].append(analyze_dump(d, usable, args.random_draws, args.seed, tag="in_sample"))
        _print_summary(entry["analyses"][-1], usable)

        if not args.skip_heldout and fit_idx.size and eval_idx.size:
            print("  held-out (select on %d prompts, score on %d)" % (fit_idx.size, eval_idx.size))
            entry["analyses"].append(analyze_dump(d, usable, args.random_draws, args.seed,
                                                  sel_idx=fit_idx, score_idx=eval_idx, tag="heldout"))
            _print_summary(entry["analyses"][-1], usable)

        for c in cross_dumps:
            if c.key() != d.key() or c.dataset == d.dataset:
                continue
            rel = assert_same_message_gram(d, c)
            print("  cross-domain -> %s (G_M identical, rel diff %.2e)" % (c.dataset, rel))
            entry["analyses"].append(analyze_dump(d, usable, args.random_draws, args.seed,
                                                  sel_idx=fit_idx if fit_idx.size else None,
                                                  score_dump=c, score_idx=None,
                                                  tag="cross_domain:%s" % c.dataset))
            _print_summary(entry["analyses"][-1], usable)

        entry["elapsed_seconds"] = time.time() - t0
        results["models"].append(entry)
        print("  done in %.1fs" % entry["elapsed_seconds"])

    write_json(args.out_json, results)
    write_text(args.out_md, build_markdown(results, ranks))
    return 0


def _print_summary(analysis, ranks):
    for R in ranks:
        s = analysis["summary"].get(R)
        if not s:
            continue
        print("    R=%-3d oracle=%.4f greedy=%.4f eigen=%.4f sim=%.4f rand=%.4f "
              "(greedy wins vs sim in %d/%d layers)"
              % (R, s["oracle"], s["median"]["greedy"], s["median"]["eigen"],
                 s["median"]["similarity"], s["median"]["random_mean"],
                 s["vs_similarity"]["greedy"]["wins"], s["n_layers"]))


if __name__ == "__main__":
    sys.exit(main())
