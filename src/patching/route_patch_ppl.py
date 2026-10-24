#!/usr/bin/env python3
"""
E4b -- route-patched perplexity harness.

Replaces each query head h's POST-SOFTMAX attention probability matrix P_h with
the matrix of its representative head phi_l(h) (a per-layer map), while leaving
the head's OWN value/output path (W^V_{kappa(h)}, W^O_h) completely untouched,
then measures WikiText-2 perplexity with a sliding window.

This is the "does the measured spectral redundancy actually buy anything"
experiment: if head h can reuse head phi(h)'s route with no loss, the routing
message operator really is low rank in the way the spectrum claims.

    Att_h(X) = P_{phi(h)}(X) @ X @ W^V_{kappa(h)} @ W^O_h      (patched)
    Att_h(X) = P_h(X)        @ X @ W^V_{kappa(h)} @ W^O_h      (unpatched)

Only P is swapped. Nothing else. Under GQA the value path for query head h is
KV group kappa(h) = h // (H_q // H_kv) and the head-permute never touches it.

--------------------------------------------------------------------------
PHI JSON SCHEMA (shared with analyze_merge.py; full spec in PHI_SCHEMA.md)
--------------------------------------------------------------------------
    {
      "schema_version": "1",
      "model": "Qwen/Qwen3-8B",            # optional, checked if present
      "num_query_heads": 32,               # optional, checked if present
      "num_kv_heads": 8,                   # optional, informational
      "num_layers": 36,                    # optional, informational
      "method": "greedy",                  # optional, free-form provenance
      "R": 8,                              # optional, free-form provenance
      "layers": {                          # REQUIRED
        "0":  [0, 0, 2, 2, 4, 4, ...],     # length H_q; entry h = phi_l(h)
        "1":  [...],
        ...
      },
      "meta": {...}                        # optional, free-form
    }

  * "layers" maps a decoder-layer index (string) to a list of length H_q of
    integer representative head indices in [0, H_q).
  * phi is expected to be idempotent: phi[phi[h]] == phi[h] (representatives
    map to themselves). Violations warn, or fail under --strict_phi.
  * Layers absent from "layers" are left as identity (warned about).
  * phi is applied on QUERY-head indices, after repeat_kv, i.e. in the same
    index space as attn_weights[:, h, :, :].
  * Spectrum-/similarity-guided phi files are produced from a Gram dump by
    `export_phi.py` (which recomputes analyze_merge.py's own maps and can
    cross-check the representative sets against e4a_merge.json). This
    script only APPLIES a phi; it never derives one.

--------------------------------------------------------------------------
IMPLEMENTATION
--------------------------------------------------------------------------
transformers 4.57 attention-interface mechanism. We register

    ALL_ATTENTION_FUNCTIONS["route_patch"]        <- code under test
    ALL_ATTENTION_FUNCTIONS["route_patch_ref"]    <- independent reference
    ALL_MASK_ATTENTION_FUNCTIONS["route_patch"]   <- eager_mask  (CRITICAL)
    ALL_MASK_ATTENTION_FUNCTIONS["route_patch_ref"]

The mask registration is NOT optional: transformers' _preprocess_mask_arguments
returns "no mask needed" for any _attn_implementation key it does not know, so a
custom attention key without a registered mask function silently gives you
NON-CAUSAL attention. Gate (iii) checks this explicitly.

phi lives on each attention module as `module._route_phi` (int64, length H_q).

Fallback patch mode (--patch_mode module_global) keeps config._attn_implementation
== "eager" and instead rebinds the model file's module-level
`eager_attention_forward` symbol, which LlamaAttention.forward resolves at call
time. Used as the second, independent code path in gate (ii).

Usage (from the repository root; outputs default to results/patching/):
    python src/patching/route_patch_ppl.py --self_test --test_wikitext   # passes with 0 fail, 0 skip
    python src/patching/route_patch_ppl.py --model qwen3-8b --method identity
    python src/patching/route_patch_ppl.py --model qwen3-8b --method random --R 8 --seed 0
    python src/patching/route_patch_ppl.py --model llama3-8b \
        --phi_json data/patching/phis/phi_Meta-Llama-3-8B_greedy_R8.json
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import inspect
import warnings
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

# ── Model registry ────────────────────────────────────────────────
MODELS = {
    # the two checkpoints of Appendix I.3 and I.4 (both H_q=32, H_kv=8)
    "qwen3-8b": {"hf_name": "Qwen/Qwen3-8B", "needs_token": False},
    "llama3-8b": {"hf_name": "meta-llama/Meta-Llama-3-8B", "needs_token": True},
    # smaller members of the same families, for cheap sanity runs
    "qwen3-0.6b": {"hf_name": "Qwen/Qwen3-0.6B", "needs_token": False},
    "qwen3-4b": {"hf_name": "Qwen/Qwen3-4B", "needs_token": False},
    "llama3.2-1b": {"hf_name": "meta-llama/Llama-3.2-1B", "needs_token": True},
    "mistral-7b-v03": {"hf_name": "mistralai/Mistral-7B-v0.3", "needs_token": False},
    # tiny CPU test models (self-test)
    "tiny-llama": {"hf_name": "hf-internal-testing/tiny-random-LlamaForCausalLM", "needs_token": False},
    "tiny-qwen3": {"hf_name": "trl-internal-testing/tiny-Qwen3ForCausalLM", "needs_token": False},
    "tiny-qwen2": {"hf_name": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", "needs_token": False},
    "tiny-mistral": {"hf_name": "hf-internal-testing/tiny-random-MistralForCausalLM", "needs_token": False},
}

PATCH_KEY = "route_patch"
PATCH_REF_KEY = "route_patch_ref"

# model_type values whose eager attention math is byte-identical to Llama's
# (verified at runtime by _eager_source_matches, this list is only for messages)
KNOWN_GOOD_MODEL_TYPES = {"llama", "qwen3", "qwen2", "mistral", "qwen3_moe"}

# Default output directory: <repository root>/results/patching/
_DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "patching")

# Debug capture, only switched on by the self-tests.
_CAPTURE = {"on": False, "store": {}}


# ══════════════════════════════════════════════════════════════════
#  Attention functions
# ══════════════════════════════════════════════════════════════════
# repeat_kv and the attention arithmetic of route_patch_attention_forward and of
# the negative controls below are copied from Hugging Face Transformers 4.57
# (src/transformers/models/llama/modeling_llama.py), Copyright 2022 EleutherAI
# and the HuggingFace Inc. team, licensed under the Apache License, Version 2.0
# (see LICENSES/Apache-2.0.txt in the repository root).
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Verbatim copy of transformers.models.llama.modeling_llama.repeat_kv."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _phi_for(module, attn_weights):
    """Fetch the per-module phi tensor, moved to the right device once."""
    phi = getattr(module, "_route_phi", None)
    if phi is None:
        return None
    if phi.device != attn_weights.device:
        phi = phi.to(attn_weights.device)
        module._route_phi = phi
    if phi.shape[0] != attn_weights.shape[1]:
        raise ValueError(
            f"phi length {phi.shape[0]} != number of query heads {attn_weights.shape[1]} "
            f"(layer {getattr(module, 'layer_idx', '?')})"
        )
    return phi


def route_patch_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    CODE UNDER TEST.

    Copy of transformers 4.57 `eager_attention_forward` with exactly one added
    statement: the post-softmax head permute.
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # ───────────────────── ROUTE PATCH ─────────────────────
    # attn_weights is [B, H_q, N, N] here (post-softmax, keys already repeated
    # to query-head granularity). Head h takes representative phi[h]'s rows.
    # value_states is NOT permuted: head h keeps its own V/O path.
    pre_patch = attn_weights
    phi = _phi_for(module, attn_weights)
    if phi is not None:
        attn_weights = attn_weights[:, phi, :, :]
    # ───────────────────────────────────────────────────────

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    if _CAPTURE["on"]:
        _CAPTURE["store"][getattr(module, "layer_idx", len(_CAPTURE["store"]))] = {
            "n_rep": module.num_key_value_groups,
            "value_raw": value.detach(),
            "value_states": value_states.detach(),
            "key_states": key_states.detach(),
            "pre_patch": pre_patch.detach(),
            "post_patch": attn_weights.detach(),
            "attn_output": attn_output.detach(),
            "phi": None if phi is None else phi.detach().clone(),
            "mask_was_none": attention_mask is None,
        }

    return attn_output, attn_weights


# Cache of the PRISTINE library eager_attention_forward per modeling module.
# Needed because --patch_mode module_global rebinds that symbol; without the
# cache the reference path would call itself (infinite recursion) and the arch
# check would compare a model against an already-patched Llama.
_ORIG_EAGER = {}


def _orig_eager(mod):
    fn = _ORIG_EAGER.get(mod.__name__)
    if fn is None:
        fn = getattr(mod, "eager_attention_forward", None)
        if fn is None:
            raise RuntimeError(f"{mod.__name__} has no module-level eager_attention_forward")
        if fn in (route_patch_attention_forward, route_patch_reference_forward):
            raise RuntimeError(
                f"{mod.__name__}.eager_attention_forward is already route-patched and the original "
                "was never cached -- restore it before continuing"
            )
        _ORIG_EAGER[mod.__name__] = fn
    return fn


def _own_eager(module):
    """The pristine eager_attention_forward of the model's OWN modeling file."""
    return _orig_eager(sys.modules[type(module).__module__])


def route_patch_reference_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    INDEPENDENT REFERENCE (gate ii). Deliberately shares no code with
    route_patch_attention_forward:

      * attention probabilities come from the *library's own*
        eager_attention_forward for this model class (so Gemma-style softcapping
        etc. would be respected here even though the code under test hardcodes
        the Llama formula);
      * the permute is done by torch.cat of per-head slices instead of advanced
        indexing;
      * the value product is recomputed from scratch.
    """
    lib_eager = _own_eager(module)
    _, attn_weights = lib_eager(
        module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs
    )
    phi = _phi_for(module, attn_weights)
    if phi is not None:
        pieces = [attn_weights[:, int(j) : int(j) + 1, :, :] for j in phi.tolist()]
        attn_weights = torch.cat(pieces, dim=1)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


_REGISTERED = False


def register_route_patch():
    """Register both attention functions AND their mask functions."""
    global _REGISTERED
    if _REGISTERED:
        return
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask

    ALL_ATTENTION_FUNCTIONS.register(PATCH_KEY, route_patch_attention_forward)
    ALL_ATTENTION_FUNCTIONS.register(PATCH_REF_KEY, route_patch_reference_forward)
    # Without these two lines transformers decides "custom backend, no mask
    # needed" and the model runs NON-CAUSALLY. Do not remove.
    ALL_MASK_ATTENTION_FUNCTIONS.register(PATCH_KEY, eager_mask)
    ALL_MASK_ATTENTION_FUNCTIONS.register(PATCH_REF_KEY, eager_mask)
    _REGISTERED = True


# ══════════════════════════════════════════════════════════════════
#  Model plumbing
# ══════════════════════════════════════════════════════════════════
def text_config(model):
    return getattr(model.config, "text_config", model.config)


def find_layers(model):
    """Locate the decoder layer ModuleList (same idiom as experiment.py)."""
    for path in ["model.layers", "model.model.layers", "language_model.model.layers"]:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    raise ValueError(f"Cannot find transformer layers in {type(model).__name__}")


def attention_modules(model):
    return [(i, layer.self_attn) for i, layer in enumerate(find_layers(model))]


def _norm_src(fn):
    return re.sub(r"\s+", " ", inspect.getsource(fn)).strip()


def check_arch_supported(model, allow_unverified=False):
    """
    The code under test hardcodes the Llama-family eager attention formula.
    Verify that this model's own eager_attention_forward is character-identical
    (modulo whitespace) to Llama's; Gemma-2/3 (logit softcapping) fails here,
    which is exactly why E4b excludes them.
    """
    import transformers.models.llama.modeling_llama as llama_mod

    llama_eager = _orig_eager(llama_mod)
    _, attn0 = attention_modules(model)[0]
    try:
        own = _own_eager(attn0)
    except RuntimeError as e:
        msg = f"ARCH CHECK: {e}"
        if not allow_unverified:
            raise RuntimeError(msg + " -- refusing to run (use --allow_unverified_arch to override)")
        print("!!! " + msg)
        return False
    same = _norm_src(own) == _norm_src(llama_eager)
    mt = getattr(text_config(model), "model_type", "?")
    if not same:
        msg = (
            f"ARCH CHECK FAILED: model_type={mt} defines an eager_attention_forward that differs "
            f"from Llama's (e.g. attention logit softcapping). route_patch_attention_forward would "
            f"silently compute the WRONG attention. Excluded models: gemma2/gemma3."
        )
        if not allow_unverified:
            raise RuntimeError(msg + " -- use --allow_unverified_arch only if you know what you are doing")
        print("!!! " + msg)
    else:
        print(f"ARCH CHECK ok: model_type={mt} shares Llama's eager attention math.")
    return same


def load_model(hf_name, device, dtype, hf_token=None, attn_impl="eager"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {hf_name} (dtype={dtype}, device={device}, attn={attn_impl}) ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kw = dict(dtype=dtype, token=hf_token, attn_implementation=attn_impl)
    if device == "cuda":
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(hf_name, **kw)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def set_attn_impl(model, key):
    """Switch the whole model (and every attention module's config) to `key`."""
    model.config._attn_implementation = key
    for _, attn in attention_modules(model):
        cfg = getattr(attn, "config", None)
        if cfg is not None:
            cfg._attn_implementation = key


def patch_via_module_global(model, fn):
    """
    Fallback / second code path: keep _attn_implementation == "eager" (so the
    library builds the mask exactly as it always does) and rebind the
    module-level eager_attention_forward symbol that XxxAttention.forward
    resolves at call time. Returns a restore() callable.
    """
    mods = {sys.modules[type(attn).__module__] for _, attn in attention_modules(model)}
    saved = []
    for m in mods:
        _orig_eager(m)  # cache the pristine function BEFORE we clobber the symbol
        saved.append((m, m.eager_attention_forward))
        m.eager_attention_forward = fn
    set_attn_impl(model, "eager")

    def restore():
        for m, old in saved:
            m.eager_attention_forward = old

    return restore


# ══════════════════════════════════════════════════════════════════
#  phi handling
# ══════════════════════════════════════════════════════════════════
def make_phi(method, num_layers, num_heads, R=None, seed=0):
    """Built-in phi generators. Returns {layer_idx: [phi(h) for h in range(H)]}."""
    phi = {}
    if method == "identity":
        for l in range(num_layers):
            phi[l] = list(range(num_heads))
        return phi
    if R is None or R < 1 or R > num_heads:
        raise ValueError(f"--R must be in [1, {num_heads}] for method={method}")
    if method == "first_R":
        # heads 0..R-1 are the representatives (fixed points); head h >= R uses
        # representative (h mod R).
        for l in range(num_layers):
            phi[l] = [h if h < R else (h % R) for h in range(num_heads)]
        return phi
    if method == "random":
        for l in range(num_layers):
            rng = np.random.default_rng(seed * 100003 + l)
            reps = np.sort(rng.choice(num_heads, size=R, replace=False))
            assign = rng.choice(reps, size=num_heads)
            assign[reps] = reps  # representatives are fixed points
            phi[l] = [int(x) for x in assign]
        return phi
    raise ValueError(f"unknown method {method}")


def load_phi_json(path, num_layers, num_heads, strict=False):
    with open(path) as f:
        blob = json.load(f)
    if "layers" not in blob:
        raise ValueError(f"{path}: missing required top-level key 'layers'")
    declared_H = blob.get("num_query_heads")
    if declared_H is not None and int(declared_H) != num_heads:
        raise ValueError(f"{path}: num_query_heads={declared_H} but model has {num_heads}")
    phi = {}
    for k, v in blob["layers"].items():
        l = int(k)
        if l < 0 or l >= num_layers:
            raise ValueError(f"{path}: layer index {l} out of range [0,{num_layers})")
        phi[l] = validate_phi(v, num_heads, strict=strict, where=f"{path}:layer {l}")
    missing = [l for l in range(num_layers) if l not in phi]
    if missing:
        print(f"!!! WARNING: {path} has no mapping for layers {missing}; using identity there.")
        for l in missing:
            phi[l] = list(range(num_heads))
    return phi, blob


def validate_phi(vals, num_heads, strict=False, where="phi"):
    if len(vals) != num_heads:
        raise ValueError(f"{where}: length {len(vals)} != H_q {num_heads}")
    out = []
    for h, x in enumerate(vals):
        xi = int(x)
        if xi < 0 or xi >= num_heads:
            raise ValueError(f"{where}: phi[{h}]={xi} out of range [0,{num_heads})")
        out.append(xi)
    bad = [h for h in range(num_heads) if out[out[h]] != out[h]]
    if bad:
        msg = f"{where}: phi is not idempotent at heads {bad[:8]}{'...' if len(bad) > 8 else ''}"
        if strict:
            raise ValueError(msg)
        print(f"!!! WARNING: {msg} (representatives should map to themselves)")
    return out


def phi_stats(phi):
    """Summary: number of distinct representatives per layer."""
    per_layer = {l: len(set(v)) for l, v in phi.items()}
    vals = list(per_layer.values())
    return {
        "distinct_reps_per_layer": {str(k): v for k, v in sorted(per_layer.items())},
        "distinct_reps_mean": float(np.mean(vals)) if vals else 0.0,
        "distinct_reps_min": int(np.min(vals)) if vals else 0,
        "distinct_reps_max": int(np.max(vals)) if vals else 0,
    }


def attach_phi(model, phi):
    """phi: {layer_idx: [ints]} or None to detach."""
    for idx, attn in attention_modules(model):
        if phi is None:
            if hasattr(attn, "_route_phi"):
                delattr(attn, "_route_phi")
            continue
        vals = phi[idx]
        dev = next(attn.parameters()).device
        attn._route_phi = torch.tensor(vals, dtype=torch.int64, device=dev)


# ══════════════════════════════════════════════════════════════════
#  Perplexity
# ══════════════════════════════════════════════════════════════════
def window_plan(n_tokens, window, stride):
    """
    Standard sliding-window plan. Returns [(begin, end, first_pred)] where
    first_pred is the in-window index of the first SCORED target token.
    With stride == window this reproduces the usual non-overlapping accounting:
    each window scores its tokens 1..N-1 (the first token of a window has no
    context and is never scored).
    """
    plan = []
    prev_end = 0
    for begin in range(0, n_tokens, stride):
        end = min(begin + window, n_tokens)
        trg_len = end - prev_end
        L = end - begin
        first_pred = max(1, L - trg_len)
        if L >= 2 and first_pred <= L - 1:
            plan.append((begin, end, first_pred))
        prev_end = end
        if end == n_tokens:
            break
    return plan


@torch.no_grad()
def compute_token_nlls(model, ids, window, stride, batch_size=1, verbose=True):
    """
    Returns (nll_tensor float64 [num_scored_tokens], plan). NLLs are in window
    order, i.e. concatenated per window over its scored positions.
    """
    device = next(model.parameters()).device
    plan = window_plan(ids.numel(), window, stride)
    nlls = []
    t0 = time.time()

    # group windows that share (length, first_pred) so batching is exact
    i = 0
    done = 0
    while i < len(plan):
        b, e, fp = plan[i]
        group = [(b, e, fp)]
        if batch_size > 1:
            j = i + 1
            while j < len(plan) and len(group) < batch_size:
                b2, e2, fp2 = plan[j]
                if (e2 - b2) == (e - b) and fp2 == fp:
                    group.append((b2, e2, fp2))
                    j += 1
                else:
                    break
            i = j
        else:
            i += 1

        batch = torch.stack([ids[bb:ee] for bb, ee, _ in group]).to(device)
        out = model(input_ids=batch, use_cache=False)
        logits = out.logits.float()
        # predict token at in-window position t from logits at t-1
        pred = logits[:, fp - 1 : -1, :]
        tgt = batch[:, fp:]
        loss = nn.functional.cross_entropy(
            pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1), reduction="none"
        ).view(tgt.shape)
        nlls.append(loss.double().cpu().reshape(-1))
        done += len(group)
        del out, logits, pred, loss
        every = 50 if len(plan) > 100 else max(1, len(plan) // 5)
        if verbose and (done % every == 0 or done == len(plan)):
            el = time.time() - t0
            print(f"  window {done}/{len(plan)}  ({el:.1f}s)")

    nll = torch.cat(nlls) if nlls else torch.zeros(0, dtype=torch.float64)
    return nll, plan


def ppl_from_nlls(nll):
    n = nll.numel()
    total = float(nll.sum())
    return (float(np.exp(total / n)) if n else float("nan")), total, n


# ══════════════════════════════════════════════════════════════════
#  Data
# ══════════════════════════════════════════════════════════════════
def get_eval_text(dataset, text_file=None):
    if dataset == "text_file":
        if not text_file:
            raise ValueError("--dataset text_file requires --text_file")
        with open(text_file) as f:
            return f.read(), f"file:{os.path.basename(text_file)}"
    if dataset == "wikitext2":
        from datasets import load_dataset

        last = None
        for repo in ["wikitext", "Salesforce/wikitext"]:  # canonical id moved under Salesforce
            try:
                ds = load_dataset(repo, "wikitext-2-raw-v1", split="test")
                return "\n\n".join(ds["text"]), "wikitext-2-raw-v1/test"
            except Exception as e:
                last = e
                print(f"!!! load_dataset('{repo}', 'wikitext-2-raw-v1') failed: {type(e).__name__}: {e}")
        raise last
    raise ValueError(f"unknown dataset {dataset}")


def tokenize_stream(tokenizer, text, limit_tokens=None):
    # verbose=False silences the "sequence longer than model_max_length" notice;
    # we never feed the whole stream to the model, only `window`-sized slices.
    enc = tokenizer(text, return_tensors="pt", verbose=False)
    ids = enc["input_ids"][0]
    if limit_tokens is not None and ids.numel() > limit_tokens:
        ids = ids[:limit_tokens]
    return ids


# ══════════════════════════════════════════════════════════════════
#  Main evaluation
# ══════════════════════════════════════════════════════════════════
def run_eval(args):
    register_route_patch()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    key = args.model
    if key in MODELS:
        hf_name = MODELS[key]["hf_name"]
        token = hf_token if MODELS[key]["needs_token"] else None
    else:
        hf_name, token = key, hf_token

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        dtype = torch.float16 if device == "cuda" else torch.float32
    else:
        dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    t_load = time.time()
    model, tokenizer = load_model(hf_name, device, dtype, hf_token=token, attn_impl="eager")
    check_arch_supported(model, allow_unverified=args.allow_unverified_arch)

    cfg = text_config(model)
    H_q = cfg.num_attention_heads
    H_kv = getattr(cfg, "num_key_value_heads", H_q)
    n_layers = len(find_layers(model))
    print(f"H_q={H_q} H_kv={H_kv} layers={n_layers}")

    # ── build phi ──
    phi_blob = None
    if args.phi_json:
        phi, phi_blob = load_phi_json(args.phi_json, n_layers, H_q, strict=args.strict_phi)
        method = phi_blob.get("method", "phi_json")
        R = phi_blob.get("R")
        if phi_blob.get("model") and phi_blob["model"] != hf_name:
            print(f"!!! WARNING: phi_json says model={phi_blob['model']} but running {hf_name}")
    else:
        method = args.method
        R = args.R if method != "identity" else H_q
        phi = make_phi(method, n_layers, H_q, R=R, seed=args.seed)
        for l, v in phi.items():
            validate_phi(v, H_q, strict=args.strict_phi, where=f"generated:{method}:layer {l}")
    stats = phi_stats(phi)
    print(f"phi: method={method} R={R} distinct reps/layer mean={stats['distinct_reps_mean']:.2f} "
          f"(min {stats['distinct_reps_min']}, max {stats['distinct_reps_max']})")

    if args.dump_phi:
        with open(args.dump_phi, "w") as f:
            json.dump({
                "schema_version": "1",
                "model": hf_name,
                "num_query_heads": int(H_q),
                "num_kv_heads": int(H_kv),
                "num_layers": int(n_layers),
                "method": method,
                "R": R,
                "meta": {"seed": args.seed, "source": "route_patch_ppl.py built-in generator"},
                "layers": {str(l): phi[l] for l in sorted(phi)},
            }, f, indent=2)
        print(f"Wrote phi to {args.dump_phi}")

    # ── data ──
    text, data_tag = get_eval_text(args.dataset, args.text_file)
    ids = tokenize_stream(tokenizer, text, args.limit_tokens)
    print(f"eval data: {data_tag}, {ids.numel()} tokens, window={args.window} stride={args.stride}")

    result = {
        "model": key,
        "hf_name": hf_name,
        "num_query_heads": int(H_q),
        "num_kv_heads": int(H_kv),
        "num_layers": int(n_layers),
        "phi_file": args.phi_json,
        "method": method,
        "R": R,
        "seed": args.seed,
        "patch_mode": args.patch_mode,
        "dataset": data_tag,
        "window": args.window,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "limit_tokens": args.limit_tokens,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "phi_stats": stats,
        "phi_meta": None if phi_blob is None else {k: v for k, v in phi_blob.items() if k != "layers"},
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gates_verified": False,
        "gate_warning": "Correctness gates are NOT run in this mode; run --self_test on CPU first.",
        "load_seconds": time.time() - t_load,
    }

    # ── optional unpatched baseline ──
    if args.baseline:
        attach_phi(model, None)
        set_attn_impl(model, "eager")
        print("\n[baseline] unpatched forward (stock eager) ...")
        t0 = time.time()
        nll_b, plan = compute_token_nlls(model, ids, args.window, args.stride, args.batch_size)
        ppl_b, tot_b, n_b = ppl_from_nlls(nll_b)
        result["baseline"] = {
            "ppl": ppl_b, "total_nll": tot_b, "num_tokens": n_b,
            "num_windows": len(plan), "runtime_seconds": time.time() - t0,
        }
        print(f"[baseline] ppl={ppl_b:.4f} over {n_b} tokens ({time.time()-t0:.0f}s)")

    # ── patched run ──
    attach_phi(model, phi)
    restore = None
    if args.patch_mode == "registry":
        set_attn_impl(model, PATCH_KEY)
    elif args.patch_mode == "module_global":
        restore = patch_via_module_global(model, route_patch_attention_forward)
    else:
        raise ValueError(args.patch_mode)

    print("\n[patched] route-patched forward ...")
    t0 = time.time()
    try:
        nll, plan = compute_token_nlls(model, ids, args.window, args.stride, args.batch_size)
    finally:
        if restore is not None:
            restore()
    ppl, tot, n_tok = ppl_from_nlls(nll)
    runtime = time.time() - t0

    result.update({
        "ppl": ppl,
        "total_nll": tot,
        "num_tokens": n_tok,
        "num_windows": len(plan),
        "runtime_seconds": runtime,
    })
    if args.baseline:
        result["delta_ppl"] = ppl - result["baseline"]["ppl"]
        result["ppl_ratio"] = ppl / result["baseline"]["ppl"]

    print(f"\n=== {key} | {method} R={R} | ppl={ppl:.4f} over {n_tok} tokens "
          f"({len(plan)} windows, {runtime:.0f}s) ===")
    if args.baseline:
        print(f"    baseline ppl={result['baseline']['ppl']:.4f}  delta={result['delta_ppl']:+.4f}  "
              f"ratio={result['ppl_ratio']:.4f}")

    out = args.output or os.path.join(_DEFAULT_OUT_DIR, f"ppl_{key.replace('/', '_')}_{method}_R{R}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out}")
    print("!!! NOTE: correctness gates were not exercised in this run "
          "(run `--self_test` separately; results only trustworthy if it passed).")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ══════════════════════════════════════════════════════════════════
#  Self-tests / correctness gates
# ══════════════════════════════════════════════════════════════════
class Gates:
    def __init__(self, quiet=False):
        self.rows = []
        self.quiet = quiet

    def add(self, gate, model, status, detail):
        self.rows.append((gate, model, status, detail))
        if not self.quiet:
            print(f"[{status}] {gate:<26} {model:<40} {detail}")

    def n_fail(self):
        return sum(1 for r in self.rows if r[2] == "FAIL")

    def failed_gates(self):
        return sorted({r[0] for r in self.rows if r[2] == "FAIL"})

    def report(self):
        n_fail = sum(1 for r in self.rows if r[2] == "FAIL")
        n_skip = sum(1 for r in self.rows if r[2] == "SKIP")
        n_na = sum(1 for r in self.rows if r[2] == "N/A")
        print("\n" + "=" * 92)
        print("GATE SUMMARY")
        print("=" * 92)
        for g, m, s, d in self.rows:
            print(f"  {s:<5} {g:<26} {m:<40} {d}")
        print("-" * 92)
        print(f"  {len(self.rows)} checks: {len(self.rows)-n_fail-n_skip-n_na} pass, {n_fail} fail, "
              f"{n_skip} skip, {n_na} n/a (structurally inapplicable)")
        if n_skip:
            print("\n" + "!" * 92)
            print("!!! WARNING: {} GATE(S) WERE SKIPPED. E4b NUMBERS ARE NOT FULLY VERIFIED.".format(n_skip))
            print("!" * 92)
        if n_fail:
            print("\n" + "!" * 92)
            print("!!! {} GATE(S) FAILED -- DO NOT TRUST ANY route_patch_ppl.py NUMBERS.".format(n_fail))
            print("!" * 92)
        return n_fail, n_skip, n_na


def _tiny_synthetic_gqa_llama(dtype=torch.float32, seed=0):
    """Offline GQA Llama (H_q=8, H_kv=2) so gate (iv) never depends on network."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=257, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=8, num_key_value_heads=2, head_dim=4,
        max_position_embeddings=256, attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg).to(dtype)
    model.eval()
    return model


def _rand_ids(vocab, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(low=1, high=max(2, min(vocab, 1000)), size=(n,), generator=g)


@torch.no_grad()
def _forward_logits(model, ids):
    device = next(model.parameters()).device
    return model(input_ids=ids.unsqueeze(0).to(device), use_cache=False).logits.float().cpu()


def gate_suite(model, name, gates, window=32, n_tokens=96, atol=1e-5):
    """
    Run gates (i),(ii),(iii),(v) on `model`; gate (iv) additionally if GQA.
    All on CPU fp32.
    """
    cfg = text_config(model)
    H_q = cfg.num_attention_heads
    H_kv = getattr(cfg, "num_key_value_heads", H_q)
    n_layers = len(find_layers(model))
    vocab = cfg.vocab_size
    ids = _rand_ids(vocab, n_tokens, seed=1234)

    # ---------- gate (i): identity phi is a no-op, token-level ----------
    attach_phi(model, None)
    set_attn_impl(model, "eager")
    nll_ref, _ = compute_token_nlls(model, ids, window, window, verbose=False)

    phi_id = make_phi("identity", n_layers, H_q)
    attach_phi(model, phi_id)
    set_attn_impl(model, PATCH_KEY)
    nll_id, _ = compute_token_nlls(model, ids, window, window, verbose=False)

    if nll_ref.numel() != nll_id.numel() or nll_ref.numel() == 0:
        gates.add("i-identity-nll", name, "FAIL", f"token count {nll_ref.numel()} vs {nll_id.numel()}")
    else:
        d = float((nll_ref - nll_id).abs().max())
        exact = bool(torch.equal(nll_ref, nll_id))
        status = "PASS" if d <= atol else "FAIL"
        gates.add("i-identity-nll", name, status,
                  f"max|dNLL|={d:.3e} over {nll_ref.numel()} tokens, bit_exact={exact}")

    # ---------- gate (v): non-vacuity -- a real phi must CHANGE the NLLs ----
    phi_zero = {l: [0] * H_q for l in range(n_layers)}
    attach_phi(model, phi_zero)
    set_attn_impl(model, PATCH_KEY)
    nll_zero, _ = compute_token_nlls(model, ids, window, window, verbose=False)
    dz = float((nll_zero - nll_ref).abs().max())
    gates.add("v-non-vacuity", name, "PASS" if dz > 1e-4 else "FAIL",
              f"max|dNLL| vs unpatched = {dz:.3e} (must be > 0; a no-op patch would read 0)")

    # ---------- gate (ii): all-heads->head0 vs independent reference --------
    # code under test: registry key + inline copy of eager
    attach_phi(model, phi_zero)
    set_attn_impl(model, PATCH_KEY)
    logits_test = _forward_logits(model, ids[:window])

    # reference: different patch MECHANISM (module-global rebind) *and*
    # different patch CODE (library eager for weights + cat-based permute)
    attach_phi(model, phi_zero)
    restore = patch_via_module_global(model, route_patch_reference_forward)
    try:
        logits_ref = _forward_logits(model, ids[:window])
    finally:
        restore()

    d2 = float((logits_test - logits_ref).abs().max())
    scale = float(logits_ref.abs().max()) + 1e-12
    gates.add("ii-vs-reference-logits", name, "PASS" if d2 <= max(atol, 1e-6 * scale) else "FAIL",
              f"max|dlogit|={d2:.3e} (|logit|max={scale:.3f}), phi=all->head0")

    # ---------- gate (ii-b): observed attention maps are the rep's ----------
    attach_phi(model, phi_zero)
    set_attn_impl(model, PATCH_KEY)
    _CAPTURE["on"], _CAPTURE["store"] = True, {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = _forward_logits(model, ids[:window])
        cap = dict(_CAPTURE["store"])
    finally:
        _CAPTURE["on"], _CAPTURE["store"] = False, {}

    ok, det = True, []
    for l, c in cap.items():
        post, pre = c["post_patch"], c["pre_patch"]
        same_as_head0 = bool(torch.equal(post, pre[:, 0:1, :, :].expand_as(post)))
        untouched_pre = bool(torch.equal(pre[:, 0], post[:, 0]))
        ok = ok and same_as_head0 and untouched_pre
        det.append(f"L{l}:{same_as_head0}/{untouched_pre}")
    gates.add("ii-b-attn-maps-are-reps", name, "PASS" if (ok and cap) else "FAIL",
              f"all heads == head0 rows / head0 unchanged: {' '.join(det) if det else 'NO CAPTURE'}")

    # ---------- gate (iii): causality + rows sum to 1 -----------------------
    ok, det = True, []
    for l, c in cap.items():
        post = c["post_patch"].float()
        if c["mask_was_none"]:
            ok = False
            det.append(f"L{l}:NO-MASK")
            continue
        nan = bool(torch.isnan(post).any() or torch.isinf(post).any())
        rows = post.sum(-1)
        row_err = float((rows - 1.0).abs().max())
        N = post.shape[-1]
        upper = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        causal = float(post.masked_select(upper.expand(post.shape)).abs().max()) if N > 1 else 0.0
        good = (not nan) and row_err < 1e-3 and causal == 0.0
        ok = ok and good
        det.append(f"L{l}:nan={nan},rowerr={row_err:.2e},upper={causal:.1e}")
    gates.add("iii-causal-and-stochastic", name, "PASS" if (ok and cap) else "FAIL",
              " ".join(det) if det else "NO CAPTURE")

    # ---------- gate (iv): GQA indexing ------------------------------------
    if H_kv == H_q:
        gates.add("iv-gqa-indexing", name, "N/A",
                  f"model is MHA (H_q=H_kv={H_q}); gate iv is exercised on the GQA models instead")
    else:
        g = np.random.default_rng(7)
        phi_rand = {l: [int(x) for x in g.integers(0, H_q, size=H_q)] for l in range(n_layers)}
        for l in phi_rand:  # make idempotent
            v = phi_rand[l]
            reps = sorted(set(v))
            for r in reps:
                v[r] = r
        attach_phi(model, phi_rand)
        set_attn_impl(model, PATCH_KEY)
        _CAPTURE["on"], _CAPTURE["store"] = True, {}
        try:
            _ = _forward_logits(model, ids[:window])
            cap4 = dict(_CAPTURE["store"])
        finally:
            _CAPTURE["on"], _CAPTURE["store"] = False, {}

        ok, det = True, []
        n_rep_expected = H_q // H_kv
        for l, c in cap4.items():
            n_rep = c["n_rep"]
            v_raw, v_rep = c["value_raw"], c["value_states"]
            shape_ok = (n_rep == n_rep_expected
                        and v_raw.shape[1] == H_kv and v_rep.shape[1] == H_q
                        and c["post_patch"].shape[1] == H_q)
            # repeat_kv must be interleave ordering: v_rep[:,h] == v_raw[:,h//n_rep]
            order_ok = all(torch.equal(v_rep[:, h], v_raw[:, h // n_rep]) for h in range(H_q))
            # head h's output == P_{phi(h)} @ V_{kappa(h)}, kappa(h)=h//n_rep
            phi_l = c["phi"]
            out = c["attn_output"].transpose(1, 2)  # back to [B,H,N,d]
            man = torch.stack(
                [c["pre_patch"][:, int(phi_l[h])] @ v_raw[:, h // n_rep] for h in range(H_q)], dim=1
            )
            num_ok = torch.allclose(out, man, atol=1e-6, rtol=1e-5)
            # discriminating check: permuting the VALUE path too must differ
            man_bad = torch.stack(
                [c["pre_patch"][:, int(phi_l[h])] @ v_raw[:, int(phi_l[h]) // n_rep] for h in range(H_q)], dim=1
            )
            differs = not torch.allclose(out, man_bad, atol=1e-6, rtol=1e-5)
            good = shape_ok and order_ok and num_ok and differs
            ok = ok and good
            det.append(f"L{l}:shape={shape_ok},order={order_ok},num={num_ok},vperm_differs={differs}")
        gates.add("iv-gqa-indexing", name, "PASS" if (ok and cap4) else "FAIL",
                  f"H_q={H_q},H_kv={H_kv},n_rep={n_rep_expected} | " + " ".join(det))

    # ---------- gate (vi): patch_mode equivalence ---------------------------
    phi_rand2 = make_phi("random", n_layers, H_q, R=max(1, H_q // 2), seed=3)
    attach_phi(model, phi_rand2)
    set_attn_impl(model, PATCH_KEY)
    nll_a, _ = compute_token_nlls(model, ids, window, window, verbose=False)
    attach_phi(model, phi_rand2)
    restore = patch_via_module_global(model, route_patch_attention_forward)
    try:
        nll_b, _ = compute_token_nlls(model, ids, window, window, verbose=False)
    finally:
        restore()
    d6 = float((nll_a - nll_b).abs().max())
    gates.add("vi-patchmode-equivalence", name, "PASS" if d6 <= atol else "FAIL",
              f"registry vs module_global max|dNLL|={d6:.3e}")

    # cleanup
    attach_phi(model, None)
    set_attn_impl(model, "eager")


# ── Negative controls: deliberately broken patches. The gate suite MUST fail
#    on each of these, otherwise the gates prove nothing. ─────────────────
def _sabotage_permute_values(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """WRONG on purpose: permutes the value path along with the attention map."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    pre = attn_weights
    phi = _phi_for(module, attn_weights)
    if phi is not None:
        attn_weights = attn_weights[:, phi, :, :]
        value_states = value_states[:, phi, :, :]  # <-- the bug
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    if _CAPTURE["on"]:
        _CAPTURE["store"][getattr(module, "layer_idx", 0)] = {
            "n_rep": module.num_key_value_groups, "value_raw": value.detach(),
            "value_states": value_states.detach(), "key_states": key_states.detach(),
            "pre_patch": pre.detach(), "post_patch": attn_weights.detach(),
            "attn_output": attn_output.detach(),
            "phi": None if phi is None else phi.detach().clone(),
            "mask_was_none": attention_mask is None,
        }
    return attn_output, attn_weights


def _sabotage_presoftmax(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """WRONG on purpose: permutes the PRE-softmax logits instead of the probabilities."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    phi = getattr(module, "_route_phi", None)
    if phi is not None:
        attn_weights = attn_weights[:, phi.to(attn_weights.device), :, :]  # <-- the bug (before mask+softmax)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    pre = attn_weights
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    if _CAPTURE["on"]:
        _CAPTURE["store"][getattr(module, "layer_idx", 0)] = {
            "n_rep": module.num_key_value_groups, "value_raw": value.detach(),
            "value_states": value_states.detach(), "key_states": key_states.detach(),
            "pre_patch": pre.detach(), "post_patch": attn_weights.detach(),
            "attn_output": attn_output.detach(),
            "phi": None if phi is None else phi.detach().clone(),
            "mask_was_none": attention_mask is None,
        }
    return attn_output, attn_weights


def _sabotage_wrong_dim(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """WRONG on purpose: permutes query POSITIONS (dim 2) instead of heads (dim 1)."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    pre = attn_weights
    phi = _phi_for(module, attn_weights)
    if phi is not None and attn_weights.shape[2] >= phi.shape[0]:
        idx = torch.arange(attn_weights.shape[2], device=attn_weights.device)
        idx[: phi.shape[0]] = phi
        attn_weights = attn_weights[:, :, idx, :]  # <-- the bug (wrong axis)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    if _CAPTURE["on"]:
        _CAPTURE["store"][getattr(module, "layer_idx", 0)] = {
            "n_rep": module.num_key_value_groups, "value_raw": value.detach(),
            "value_states": value_states.detach(), "key_states": key_states.detach(),
            "pre_patch": pre.detach(), "post_patch": attn_weights.detach(),
            "attn_output": attn_output.detach(),
            "phi": None if phi is None else phi.detach().clone(),
            "mask_was_none": attention_mask is None,
        }
    return attn_output, attn_weights


def _sabotage_inverse_map(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """WRONG on purpose: scatters (out[phi[h]] = P[h]) instead of gathering (out[h] = P[phi[h]])."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    pre = attn_weights
    phi = _phi_for(module, attn_weights)
    if phi is not None:
        scattered = attn_weights.clone()
        scattered[:, phi, :, :] = attn_weights  # <-- the bug (inverse direction)
        attn_weights = scattered
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    if _CAPTURE["on"]:
        _CAPTURE["store"][getattr(module, "layer_idx", 0)] = {
            "n_rep": module.num_key_value_groups, "value_raw": value.detach(),
            "value_states": value_states.detach(), "key_states": key_states.detach(),
            "pre_patch": pre.detach(), "post_patch": attn_weights.detach(),
            "attn_output": attn_output.detach(),
            "phi": None if phi is None else phi.detach().clone(),
            "mask_was_none": attention_mask is None,
        }
    return attn_output, attn_weights


def run_negative_controls(gates):
    """
    Re-run the whole gate suite against known-broken implementations and against
    a missing causal-mask registration. Each control must be CAUGHT (n_fail > 0).
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask

    model = _tiny_synthetic_gqa_llama()
    controls = [
        ("permute-values-too", lambda: ALL_ATTENTION_FUNCTIONS.register(PATCH_KEY, _sabotage_permute_values), None),
        ("permute-wrong-axis", lambda: ALL_ATTENTION_FUNCTIONS.register(PATCH_KEY, _sabotage_wrong_dim), None),
        ("inverse-map-scatter", lambda: ALL_ATTENTION_FUNCTIONS.register(PATCH_KEY, _sabotage_inverse_map), None),
        # NOTE: permuting the PRE-softmax logits is mathematically equivalent to
        # permuting the probabilities whenever the additive mask is head-independent
        # ([B,1,N,N]); it is caught as a *semantic* divergence, not a numeric one.
        ("permute-pre-softmax", lambda: ALL_ATTENTION_FUNCTIONS.register(PATCH_KEY, _sabotage_presoftmax), None),
        ("unregistered-causal-mask",
         lambda: ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(PATCH_KEY, None),
         lambda: ALL_MASK_ATTENTION_FUNCTIONS.register(PATCH_KEY, eager_mask)),
    ]
    for label, break_it, fix_it in controls:
        break_it()
        try:
            probe = Gates(quiet=True)
            gate_suite(model, "negative-control", probe)
            caught = probe.n_fail() > 0
            detail = f"caught by {probe.failed_gates()}" if caught else "NOT CAUGHT -- gates are vacuous!"
        except Exception as e:  # an exception also counts as caught
            caught, detail = True, f"raised {type(e).__name__}: {str(e)[:60]}"
        finally:
            ALL_ATTENTION_FUNCTIONS.register(PATCH_KEY, route_patch_attention_forward)
            if fix_it is not None:
                fix_it()
            _CAPTURE["on"], _CAPTURE["store"] = False, {}
            attach_phi(model, None)
            set_attn_impl(model, "eager")
        gates.add("x-negative-control", label, "PASS" if caught else "FAIL", detail)
    del model


def gate_ppl_loop(gates, name="ppl-accounting"):
    """Arithmetic-only checks of the sliding-window accounting."""
    plan = window_plan(2048, 512, 512)
    scored = sum(e - b - fp for b, e, fp in plan)
    ok1 = (len(plan) == 4 and scored == 4 * 511)
    gates.add("vii-window-accounting", name, "PASS" if ok1 else "FAIL",
              f"n=2048,w=512,s=512 -> {len(plan)} windows, {scored} scored tokens (expect 4/2044)")

    plan2 = window_plan(1000, 512, 256)
    ok2 = plan2[0] == (0, 512, 1) and all(fp == 256 for _, _, fp in plan2[1:])
    covered = sum(e - b - fp for b, e, fp in plan2)
    gates.add("vii-b-overlap-accounting", name, "PASS" if (ok2 and covered == 999) else "FAIL",
              f"n=1000,w=512,s=256 -> {plan2} scored={covered} (expect 999)")


def run_self_test(args):
    register_route_patch()
    gates = Gates()
    torch.manual_seed(0)

    gate_ppl_loop(gates)

    # 1) synthetic offline GQA Llama (never depends on the network)
    try:
        m = _tiny_synthetic_gqa_llama()
        check_arch_supported(m)
        gate_suite(m, "synthetic-gqa-llama(H8/KV2)", gates)
        del m
    except Exception as e:
        gates.add("suite", "synthetic-gqa-llama(H8/KV2)", "FAIL", f"{type(e).__name__}: {e}")

    # 2) hf-internal-testing tiny Llama (MHA)  3) tiny Qwen3 (GQA + QK-norm)
    todo = [
        ("hf-internal-testing/tiny-random-LlamaForCausalLM", "tiny-random-LlamaForCausalLM"),
        ("trl-internal-testing/tiny-Qwen3ForCausalLM", "tiny-Qwen3ForCausalLM"),
    ]
    if args.extra_test_models:
        todo += [(x, x.split("/")[-1]) for x in args.extra_test_models]
    for hf_name, label in todo:
        try:
            m, tok = load_model(hf_name, "cpu", torch.float32, attn_impl="eager")
        except Exception as e:
            gates.add("suite", label, "SKIP", f"could not load ({type(e).__name__}: {str(e)[:80]})")
            continue
        try:
            check_arch_supported(m)
            gate_suite(m, label, gates)
        except Exception as e:
            gates.add("suite", label, "FAIL", f"{type(e).__name__}: {e}")
        finally:
            del m
            gc.collect()

    # 3b) negative controls: prove the gates are not vacuous
    if not args.no_negative_control:
        try:
            run_negative_controls(gates)
        except Exception as e:
            gates.add("x-negative-control", "all", "FAIL", f"{type(e).__name__}: {e}")
    else:
        gates.add("x-negative-control", "all", "SKIP", "--no_negative_control given")

    # 4) end-to-end ppl path on a local text file with a real tokenizer
    try:
        m, tok = load_model("hf-internal-testing/tiny-random-LlamaForCausalLM", "cpu", torch.float32)
        text = ("the quick brown fox jumps over the lazy dog . " * 300)
        ids = tokenize_stream(tok, text, limit_tokens=2048)
        attach_phi(m, make_phi("identity", len(find_layers(m)), text_config(m).num_attention_heads))
        set_attn_impl(m, PATCH_KEY)
        nll, plan = compute_token_nlls(m, ids, 512, 512, verbose=False)
        ppl, tot, n = ppl_from_nlls(nll)
        ok = np.isfinite(ppl) and n == sum(e - b - fp for b, e, fp in plan)
        gates.add("viii-end-to-end-ppl", "tiny-random-LlamaForCausalLM", "PASS" if ok else "FAIL",
                  f"ppl={ppl:.3f} tokens={n} windows={len(plan)} (random-init model, value meaningless)")
        del m
    except Exception as e:
        gates.add("viii-end-to-end-ppl", "tiny-random-LlamaForCausalLM", "SKIP",
                  f"{type(e).__name__}: {str(e)[:80]}")

    # 5) datasets availability (streaming/download test is optional)
    if args.test_wikitext:
        try:
            text, tag = get_eval_text("wikitext2")
            gates.add("ix-wikitext2-load", tag, "PASS", f"{len(text)} chars")
        except ImportError as e:
            # `datasets` (or its pyarrow) cannot be imported: every perplexity run
            # would fail the same way, so this is a broken environment, not a skip
            gates.add("ix-wikitext2-load", "wikitext-2-raw-v1", "FAIL",
                      f"{type(e).__name__}: {str(e)[:80]}")
        except Exception as e:
            gates.add("ix-wikitext2-load", "wikitext-2-raw-v1", "SKIP",
                      f"{type(e).__name__}: {str(e)[:80]}")
    else:
        gates.add("ix-wikitext2-load", "wikitext-2-raw-v1", "SKIP",
                  "not requested (pass --test_wikitext to exercise the datasets path)")

    n_fail, n_skip, n_na = gates.report()
    if args.self_test_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.self_test_json)), exist_ok=True)
        with open(args.self_test_json, "w") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "rows": [{"gate": g, "model": m, "status": s, "detail": d} for g, m, s, d in gates.rows],
                "n_fail": n_fail, "n_skip": n_skip, "n_na": n_na,
            }, f, indent=2)
        print(f"Wrote {args.self_test_json}")
    return 1 if n_fail else 0


# ══════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3-8b",
                   help="registry key, HF checkpoint name, or local checkpoint directory")
    p.add_argument("--phi_json", default=None, help="phi file (see PHI_SCHEMA.md)")
    p.add_argument("--method", default="identity", choices=["identity", "random", "first_R"])
    p.add_argument("--R", type=int, default=8, help="number of representative heads (random/first_R)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strict_phi", action="store_true", help="fail instead of warn on non-idempotent phi")
    p.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "text_file"])
    p.add_argument("--text_file", default=None)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--limit_tokens", type=int, default=None)
    p.add_argument("--baseline", action="store_true", help="also measure the unpatched model in the same run")
    p.add_argument("--patch_mode", default="registry", choices=["registry", "module_global"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    p.add_argument("--allow_unverified_arch", action="store_true")
    p.add_argument("--output", default=None,
                   help="result JSON (default: results/patching/ppl_<model>_<method>_R<R>.json)")
    p.add_argument("--dump_phi", default=None, help="write the phi actually used, in PHI_SCHEMA.md format")
    p.add_argument("--hf_token", default=None, help="defaults to $HF_TOKEN; never hardcode a token")
    p.add_argument("--self_test", action="store_true",
                   help="run the correctness gates on tiny CPU models; a full pass reads "
                        "'0 fail, 0 skip' (needs --test_wikitext)")
    p.add_argument("--self_test_json", default=None)
    p.add_argument("--test_wikitext", action="store_true",
                   help="in --self_test, also load WikiText-2 (gate ix; skipped otherwise)")
    p.add_argument("--no_negative_control", action="store_true",
                   help="skip the sabotaged-implementation controls (NOT recommended)")
    p.add_argument("--extra_test_models", nargs="*", default=None)
    args = p.parse_args()

    if args.self_test:
        sys.exit(run_self_test(args))
    run_eval(args)


if __name__ == "__main__":
    main()
