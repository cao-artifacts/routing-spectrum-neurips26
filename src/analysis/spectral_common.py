#!/usr/bin/env python3
"""
Shared numpy helpers for the offline analyzers of Appendices H and I.

Pure numpy (no torch, no scipy, no frameworks). Everything here is CPU-only and
operates on the H_q x H_q Gram matrices dumped by the GPU measurement drivers.

--------------------------------------------------------------------------
NOTATION (must match src/measure/experiment.py exactly)
--------------------------------------------------------------------------
For one layer with H = H_q query heads:

  a_h  in R^{S*N^2}   route side: vec of the eager attention map P_h(X^s),
                      concatenated over the S calibration prompts.
                      A = [a_1 ... a_H]  ->  G_P = A^T A,  i.e.
                      G_P[h,h'] = sum_s <P_h(X^s), P_h'(X^s)>_F.
                      Per-prompt contributions g_s (H x H, PSD) sum to G_P.

  b_h  in R^{m^2}     message side: vec of M_h = W^V_{kappa(h)} W^O_h
                      (dequantized value-output factors).
                      B = [b_1 ... b_H]  ->  G_M = B^T B,  i.e.
                      G_M[h,h'] = <M_h, M_h'>_F.

  Psi = A B^T = sum_h a_h b_h^T   (the calibration-stacked routing-message
  operator).  Psi^T Psi = B G_P B^T, so the nonzero eigenvalues of G_P G_M
  are exactly the squared singular values of Psi -- this is the H x H Gram
  trick used by the published measurement script, and nothing here changes it.

  ||Psi||_F^2 = tr(G_P G_M) = total energy.

--------------------------------------------------------------------------
.npz DUMP SCHEMA  (version 1)  -- the contract with the GPU drivers
--------------------------------------------------------------------------
One .npz per (model, dataset, seq_len), as written by
`experiment.py --dump_grams`.  Canonical keys (aliases in brackets are also
accepted so a renamed dump still loads):

  "G_M"             float32/64 [L, H, H]      message Gram per layer.  Depends
                    on weights only, so it is IDENTICAL across datasets for the
                    same model+quantization -- the cross-domain analyses assert
                    exactly that.                        [alias: gram_M, GM]

  "G_P_per_prompt"  float32    [S, L, H, H]   PER-PROMPT route Gram
                    contributions g_s.  G_P^C for any prompt subset C is the
                    plain sum over s in C, which is what makes held-out splits,
                    bootstrap resampling and cross-domain scoring possible.
                                              [alias: G_P, gram_P, gram_P_per_prompt]

  "G_P_total"       float64    [L, H, H]      the accumulated sum over all S
                    prompts.  Used only as a fallback when the per-prompt
                    tensor is absent; with it alone, splits and bootstrap CIs
                    cannot be computed (the analyzers say so and skip them).

  "config_json"     str        JSON blob, stored as np.array(json.dumps(cfg)).
                                              [alias: config, cfg, meta]

  Required config fields:
      model            str   registry key, e.g. "qwen3-8b"
      hf_name          str   HF checkpoint id
      dataset          str   "c4" | "wikitext" | "code" | ...
      seq_len          int   token truncation length (512 in the paper)
      num_query_heads  int   H
      num_kv_heads     int   H_kv
      num_layers       int   L
      num_prompts      int   S  (== G_P_per_prompt.shape[0])
  Recommended provenance: revision, quant_used, torch_dtype, seed,
      library_versions, script_sha256, timestamp_utc, prompts.

  Optional arrays: "layer_index" [L], "prompt_index" [S], and any further
  per-prompt provenance the driver wants to carry.

PROMPT ORDER IS PART OF THE CONTRACT: prompt s must be the same document in
every dump of a given (dataset, seq_len), and prompts 0..199 must reproduce the
published 200-prompt calibration set, because the held-out analysis addresses
[0:200] as the fit split and [200:400] as the held-out split by index.

Size note for the drivers: [S,L,H,H] float32 with S=400, L=80, H=64 is 524 MB.
Use np.savez (not savez_compressed -- random floats barely compress) and keep
the per-prompt tensor in float32.  The analyzers stream one layer at a time.
"""

import json
import os

import numpy as np

SCHEMA_VERSION = 1

# accepted key aliases so a slightly-renamed driver dump still loads.
# order matters: the per-prompt tensor is always preferred over the total.
_GM_KEYS = ["G_M", "gram_M", "GM", "gram_m"]
_GP_KEYS = ["G_P_per_prompt", "gram_P_per_prompt", "G_P", "gram_P", "GP", "gram_p"]
_GP_TOTAL_KEYS = ["G_P_total", "gram_P_total", "G_P_sum"]
_CFG_KEYS = ["config_json", "config", "cfg", "meta"]

REQUIRED_CONFIG_FIELDS = [
    "model",
    "dataset",
    "seq_len",
    "num_query_heads",
    "num_kv_heads",
    "num_layers",
    "num_prompts",
]


# ── basic linear algebra ──────────────────────────────────────────

def sym(M):
    """Clean NaN/Inf and symmetrize (same hygiene as experiment.py)."""
    M = np.asarray(M, dtype=np.float64)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    return 0.5 * (M + M.T)


def psd_sqrt(M, eps=1e-12):
    """
    Symmetric PSD square root and its pseudo-inverse.

    Returns (M^{1/2}, M^{+1/2}, rank).  Eigenvalues below eps * lambda_max are
    clipped to zero, so a rank-deficient G_M is handled without blowing up.
    """
    M = sym(M)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 0.0)
    wmax = w.max() if w.size else 0.0
    tol = eps * wmax
    keep = w > tol
    if not np.any(keep):
        n = M.shape[0]
        return np.zeros((n, n)), np.zeros((n, n)), 0
    Vk = V[:, keep]
    sk = np.sqrt(w[keep])
    half = (Vk * sk) @ Vk.T
    half_pinv = (Vk * (1.0 / sk)) @ Vk.T
    return half, half_pinv, int(keep.sum())


def total_energy(GP, GM):
    """||Psi||_F^2 = tr(G_P G_M)."""
    return float(np.sum(sym(GP) * sym(GM)))  # tr(XY) = sum(X * Y) for symmetric X, Y


def spectrum_sigma_sq(GP, GM, GMh=None):
    """
    Squared singular values of Psi, descending, length H.

    Same math as experiment.py's compute_spectrum, written with the G_M^{1/2}
    side symmetrized instead of the G_P side (identical nonzero spectrum:
    eig(G_P G_M) = eig(G_M^{1/2} G_P G_M^{1/2})).  Using the G_M side lets us
    reuse one factorization across many bootstrap resamples of G_P.
    """
    GP = sym(GP)
    if GMh is None:
        GMh, _, _ = psd_sqrt(GM)
    S = sym(GMh @ GP @ GMh)
    ss = np.linalg.eigvalsh(S)
    ss = np.maximum(ss, 0.0)
    return np.sort(ss)[::-1]


def r_eff(sigma_sq):
    """Effective rank (sum sigma)^2 / sum sigma^2 -- identical to experiment.py."""
    sigma_sq = np.asarray(sigma_sq, dtype=np.float64)
    tot = float(sigma_sq.sum())
    if tot <= 0:
        return 0.0
    sigma = np.sqrt(sigma_sq)
    return float(sigma.sum() ** 2 / tot)


def tail_fraction(sigma_sq, R):
    """T_R = sum_{j>R} sigma_j^2 / sum_j sigma_j^2 -- identical to experiment.py."""
    sigma_sq = np.asarray(sigma_sq, dtype=np.float64)
    tot = float(sigma_sq.sum())
    if tot <= 0:
        return 0.0
    if R >= sigma_sq.size:
        return 0.0
    return float(sigma_sq[R:].sum() / tot)


# ── width-R student with fixed message maps, free routes ──────────

def fit_route_basis(GP_fit, GM, GMh=None, GMh_pinv=None, eps=1e-12):
    """
    Fit the message-side coefficient matrix of the width-R student.

    The student keeps the H original message maps but is allowed only R fixed
    linear combinations of them, m_j = B w_j, while its per-input routes are
    free.  Its stacked matrix is therefore  Atilde (B W)^T  with Atilde free,
    so the best student residual is the energy of Psi outside span(B W).

    Maximizing captured energy over W in R^{H x R} gives the generalized
    eigenproblem
            G_P^fit G_M w_j = lambda_j w_j,
    solved here in the numerically stable symmetric form
            w_j = G_M^{+1/2} u_j ,
            u_j = j-th eigenvector of  S = G_M^{1/2} G_P^fit G_M^{1/2}.

    Returns (lam [H] descending, W_full [H, H] with columns w_1..w_H).
    Slicing W_full[:, :R] gives the width-R fit, and the columns are nested,
    which is what makes the residual monotone nonincreasing in R.
    """
    GP_fit = sym(GP_fit)
    if GMh is None or GMh_pinv is None:
        GMh, GMh_pinv, _ = psd_sqrt(GM, eps=eps)
    S = sym(GMh @ GP_fit @ GMh)
    lam, U = np.linalg.eigh(S)
    order = np.argsort(lam)[::-1]
    lam = np.maximum(lam[order], 0.0)
    U = U[:, order]
    W_full = GMh_pinv @ U
    return lam, W_full


def captured_energy(GP_eval, GM, W):
    """tr(G_P^eval G_M W (W^T G_M W)^{-1} W^T G_M), with pinv for rank deficiency."""
    GP_eval = sym(GP_eval)
    GM = sym(GM)
    W = np.atleast_2d(np.asarray(W, dtype=np.float64))
    if W.shape[0] != GM.shape[0]:
        raise ValueError("W has %d rows, expected %d" % (W.shape[0], GM.shape[0]))
    GMW = GM @ W                      # H x R   (== (W^T G_M)^T, G_M symmetric)
    mid = W.T @ GMW                   # R x R
    proj = GMW @ np.linalg.pinv(mid) @ GMW.T
    return float(np.sum(GP_eval * proj))  # tr(GP_eval @ proj), both symmetric


def residual_fraction(GP_eval, GM, W):
    """
    1 - tr(G_P^eval G_M W (W^T G_M W)^{-1} W^T G_M) / tr(G_P^eval G_M).

    Not clipped: a perfect fit can come back as -1e-16.
    """
    tot = total_energy(GP_eval, GM)
    if tot <= 0:
        return 0.0
    return 1.0 - captured_energy(GP_eval, GM, W) / tot


# ── route-merging (E4a) ───────────────────────────────────────────

def merge_matrix(phi, H):
    """Pi_phi[i,h] = 1 iff i == phi(h).  Rank = |image(phi)|."""
    Pi = np.zeros((H, H), dtype=np.float64)
    Pi[np.asarray(phi, dtype=int), np.arange(H)] = 1.0
    return Pi


def merge_error(GP, GM, phi):
    """
    Calibration error of a PatternMerge map phi: [H] -> [H].

    The merged operator reuses route column phi(h) with the ORIGINAL message
    column h:  Psitilde = A Pi_phi B^T.  Hence

        error(phi) = ||Psi - Psitilde||_F^2
                   = tr(C^T G_P C G_M),      C = I - Pi_phi

    and expanding C column-wise (d_h = a_h - a_{phi(h)}) gives the O(H^2)
    index form used here:

        error = sum_{h,h'} G_M[h,h'] * ( G_P[h,h'] - G_P[h,phi(h')]
                                         - G_P[phi(h),h'] + G_P[phi(h),phi(h')] )
    """
    GP = sym(GP)
    GM = sym(GM)
    p = np.asarray(phi, dtype=int)
    E = GP + GP[np.ix_(p, p)] - GP[:, p] - GP[p, :]
    return float(np.sum(GM * E))


def merge_error_batch(GP, GM, PHI):
    """
    Vectorized merge_error for a batch of maps.  PHI is [B, H]; returns [B].

    Expanding the index form and using the symmetry of G_P and G_M,

        error(phi) = tr(G_P G_M) - 2 * sum_h (G_P G_M)[phi(h), h]
                     + sum_{h,h'} G_M[h,h'] G_P[phi(h), phi(h')],

    so only the last term needs the H x H gather, which batches cleanly.
    """
    GP = sym(GP)
    GM = sym(GM)
    PHI = np.atleast_2d(np.asarray(PHI, dtype=int))
    H = GP.shape[0]
    X = GP @ GM
    t0 = float(np.sum(GP * GM))
    t2 = X[PHI, np.arange(H)[None, :]].sum(axis=1)             # [B]
    gathered = GP[PHI[:, :, None], PHI[:, None, :]]            # [B, H, H]
    t3 = np.einsum("bij,ij->b", gathered, GM)
    return t0 - 2.0 * t2 + t3


def merge_error_fraction(GP, GM, phi):
    tot = total_energy(GP, GM)
    if tot <= 0:
        return 0.0
    return merge_error(GP, GM, phi) / tot


def phi_rank(phi):
    return int(np.unique(np.asarray(phi, dtype=int)).size)


# ── .npz dump loading ─────────────────────────────────────────────

def _pick(npz, keys, what, path):
    for k in keys:
        if k in npz.files:
            return k
    raise KeyError(
        "%s: no %s array found (looked for %s; file has %s)"
        % (path, what, keys, list(npz.files))
    )


def _decode_config(raw):
    if isinstance(raw, np.ndarray):
        raw = raw.item() if raw.ndim == 0 else raw.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise ValueError("cannot decode config of type %r" % type(raw))


class Dump(object):
    """One measurement dump: G_M [L,H,H], per-prompt G_P [S,L,H,H], config dict."""

    def __init__(self, path, G_M, G_P, config, prompt_index=None, layer_index=None):
        self.path = path
        self.G_M = np.asarray(G_M, dtype=np.float64)
        self.G_P = G_P  # kept in file dtype (often float32) to save RAM
        self.config = config
        self.prompt_index = prompt_index
        self.layer_index = layer_index

        if self.G_M.ndim != 3:
            raise ValueError("%s: G_M must be [L,H,H], got %s" % (path, self.G_M.shape))
        if self.G_P.ndim == 3:  # already summed over prompts
            self.G_P = self.G_P[None, ...]
            self.per_prompt = False
        elif self.G_P.ndim == 4:
            self.per_prompt = True
        else:
            raise ValueError("%s: G_P must be [S,L,H,H], got %s" % (path, self.G_P.shape))

        self.S, self.L, self.H = self.G_P.shape[0], self.G_P.shape[1], self.G_P.shape[2]
        if self.G_M.shape != (self.L, self.H, self.H):
            raise ValueError(
                "%s: shape mismatch G_M %s vs G_P %s" % (path, self.G_M.shape, self.G_P.shape)
            )

    # -- convenience --------------------------------------------------
    @property
    def model(self):
        return self.config.get("model", "unknown")

    @property
    def dataset(self):
        return self.config.get("dataset", "unknown")

    @property
    def seq_len(self):
        return self.config.get("seq_len", -1)

    def key(self):
        return (self.model, int(self.seq_len))

    def layer_id(self, l):
        """True model layer index (dumps may carry an explicit layer_index)."""
        if self.layer_index is not None and l < len(self.layer_index):
            return int(self.layer_index[l])
        return int(l)

    def label(self):
        return "%s/%s/N%s" % (self.model, self.dataset, self.seq_len)

    # -- accumulation -------------------------------------------------
    def gp_sum(self, layer, idx=None):
        """G_P^C for layer `layer` and prompt subset `idx` (None = all), float64."""
        block = self.G_P[:, layer]
        if idx is None:
            acc = block.astype(np.float64).sum(axis=0)
        else:
            idx = np.asarray(idx, dtype=int)
            acc = block[idx].astype(np.float64).sum(axis=0)
        return sym(acc)

    def gp_layer_flat(self, layer):
        """[S, H*H] float64 view of one layer's per-prompt Grams (for bootstrap)."""
        return self.G_P[:, layer].astype(np.float64).reshape(self.S, self.H * self.H)

    def gm(self, layer):
        return sym(self.G_M[layer])


def load_dump(path, check_config=True):
    if not os.path.exists(path):
        raise IOError("dump not found: %s" % path)
    npz = np.load(path, allow_pickle=False)
    gm_key = _pick(npz, _GM_KEYS, "G_M", path)
    gp_key = next((k for k in _GP_KEYS if k in npz.files), None)
    if gp_key is None:
        gp_key = next((k for k in _GP_TOTAL_KEYS if k in npz.files), None)
        if gp_key is None:
            raise KeyError("%s: no per-prompt G_P and no G_P total (looked for %s / %s; "
                           "file has %s)" % (path, _GP_KEYS, _GP_TOTAL_KEYS, list(npz.files)))
        print("  [warn] %s: only '%s' present -- held-out splits and bootstrap CIs are "
              "unavailable (re-dump with --dump_grams per-prompt contributions)"
              % (os.path.basename(path), gp_key))
    cfg_key = next((k for k in _CFG_KEYS if k in npz.files), None)
    config = _decode_config(npz[cfg_key]) if cfg_key else {}
    prompt_index = npz["prompt_index"] if "prompt_index" in npz.files else None
    layer_index = npz["layer_index"] if "layer_index" in npz.files else None
    d = Dump(path, npz[gm_key], npz[gp_key], config, prompt_index, layer_index)

    if check_config:
        missing = [f for f in REQUIRED_CONFIG_FIELDS if f not in config]
        if missing:
            print("  [warn] %s: config missing fields %s" % (os.path.basename(path), missing))
        if config.get("num_query_heads") not in (None, d.H):
            print("  [warn] %s: config num_query_heads=%s but G_P says H=%d"
                  % (os.path.basename(path), config.get("num_query_heads"), d.H))
        if config.get("num_layers") not in (None, d.L):
            print("  [warn] %s: config num_layers=%s but G_P says L=%d"
                  % (os.path.basename(path), config.get("num_layers"), d.L))
        if d.per_prompt and config.get("num_prompts") not in (None, d.S):
            print("  [warn] %s: config num_prompts=%s but G_P says S=%d"
                  % (os.path.basename(path), config.get("num_prompts"), d.S))
    return d


def assert_same_message_gram(a, b, rtol=1e-6, atol=0.0):
    """
    G_M depends on weights only, so a cross-domain dump of the same model must
    carry a bit-comparable G_M.  Raises with the observed discrepancy.
    """
    if a.G_M.shape != b.G_M.shape:
        raise AssertionError(
            "G_M shape mismatch: %s %s vs %s %s"
            % (a.label(), a.G_M.shape, b.label(), b.G_M.shape)
        )
    scale = max(np.abs(a.G_M).max(), np.abs(b.G_M).max(), 1e-300)
    diff = np.abs(a.G_M - b.G_M).max()
    rel = diff / scale
    if not (diff <= atol + rtol * scale):
        raise AssertionError(
            "G_M differs between %s and %s (max abs %.6g, relative %.3g > rtol %.3g); "
            "cross-domain comparison requires the same model weights/quantization"
            % (a.label(), b.label(), diff, rel, rtol)
        )
    return float(rel)


# ── small reporting helpers ───────────────────────────────────────

def median(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return float("nan")
    return float(np.median(np.asarray(xs, dtype=np.float64)))


def pct(xs, q):
    xs = np.asarray([x for x in xs if np.isfinite(x)], dtype=np.float64)
    if xs.size == 0:
        return float("nan")
    return float(np.percentile(xs, q))


def md_table(headers, rows):
    """Compact markdown table; rows are lists of already-formatted strings."""
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def jsonable(obj):
    """Recursively convert numpy scalars/arrays so json.dump works."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def write_json(path, payload):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(jsonable(payload), f, indent=2)
    print("wrote %s" % path)


def write_text(path, text):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    print("wrote %s" % path)


def parse_range(spec, S):
    """'0:200' -> np.arange(0,200) clipped to S.  'all' -> arange(S)."""
    spec = str(spec).strip()
    if spec in ("all", ":", ""):
        return np.arange(S)
    if ":" not in spec:
        raise ValueError("range must look like '0:200', got %r" % spec)
    a, b = spec.split(":")
    a = 0 if a == "" else int(a)
    b = S if b == "" else int(b)
    a, b = max(0, a), min(S, b)
    if b <= a:
        return np.arange(0)
    return np.arange(a, b)
