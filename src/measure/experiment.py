#!/usr/bin/env python3
"""
Routing-message spectrum measurement for multi-head attention (rebuttal edition).

Extension of experiments/scripts/experiment.py used for the NeurIPS 2026
submission.  The measurement math is unchanged:

  1. G_P^C[h,h'] = sum_s <P_h(X^s), P_h'(X^s)>_F   from eager attention maps
  2. G_M[h,h']   = <M_h, M_h'>_F, M_h = W^V_kappa(h) W^O_h, from dequantized weights
  3. spectrum    = eigenvalues of G_P^{1/2} G_M G_P^{1/2}  (H_q x H_q Gram eigenproblem)

New in this version (all opt-in; defaults reproduce the published protocol
c4 / S=200 / N=512 / quant auto):

  --dump_grams DIR   save per-prompt G_P contributions + G_M + provenance to .npz
  --dataset          c4 | wikitext | code
  --quant            auto | fp16 | bf16 | nf4 | fp32
  --seed             seed for any future sampling (selection stays first-K)
  --model_id         arbitrary HF id escape hatch (config introspected)
  --pool_scan_limit  raise the calibration-pool scan cap
  NaN/Inf accounting on attention maps and Grams *before* any nan_to_num,
  per-layer nonzero-energy assertion, and a 'corrupted' flag per layer.

Usage:
    python experiment.py --models qwen3-8b llama3-70b \
                         --num_prompts 200 --seq_len 512 \
                         --dataset c4 --quant auto \
                         --dump_grams ./npz --output results.json
"""

import argparse
import hashlib
import json
import gc
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# ── Model registry ────────────────────────────────────────────────
MODELS = {
    # ── Qwen3 family ──
    "qwen3-0.6b": {"hf_name": "Qwen/Qwen3-0.6B", "needs_token": False},
    "qwen3-1.7b": {"hf_name": "Qwen/Qwen3-1.7B", "needs_token": False},
    "qwen3-4b": {"hf_name": "Qwen/Qwen3-4B", "needs_token": False},
    "qwen3-8b": {"hf_name": "Qwen/Qwen3-8B", "needs_token": False},
    "qwen3-14b": {"hf_name": "Qwen/Qwen3-14B", "needs_token": False},
    "qwen3-32b": {"hf_name": "Qwen/Qwen3-32B", "needs_token": False},
    # ── Llama 3 family ──
    "llama3.2-1b": {"hf_name": "meta-llama/Llama-3.2-1B", "needs_token": True},
    "llama3.2-3b": {"hf_name": "meta-llama/Llama-3.2-3B", "needs_token": True},
    "llama3-8b": {"hf_name": "meta-llama/Meta-Llama-3-8B", "needs_token": True},
    "llama3-70b": {"hf_name": "meta-llama/Meta-Llama-3-70B", "needs_token": True},
    # ── Gemma 3 family ──
    "gemma3-1b": {"hf_name": "google/gemma-3-1b-pt", "needs_token": False},
    "gemma2-2b": {"hf_name": "google/gemma-2-2b", "needs_token": False},
    "gemma2-9b": {"hf_name": "google/gemma-2-9b", "needs_token": False},
    "gemma2-27b": {"hf_name": "google/gemma-2-27b", "needs_token": False},
    # ── Mistral family (latest versions) ──
    "mistral-7b-v03": {"hf_name": "mistralai/Mistral-7B-v0.3", "needs_token": False},
    "mistral-nemo-12b": {"hf_name": "mistralai/Mistral-Nemo-Base-2407", "needs_token": False},
    "mistral-small-24b": {"hf_name": "mistralai/Mistral-Small-3.1-24B-Base-2503", "needs_token": False},
}

# The 14 checkpoints reported in Table 1 (same order as plot_results_v2.MODEL_ORDER).
TABLE1_MODELS = [
    "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b",
    "llama3.2-1b", "llama3.2-3b", "llama3-8b", "llama3-70b",
    "gemma2-2b", "gemma2-9b",
    "mistral-7b-v03", "mistral-nemo-12b",
]

# ── Calibration corpora ───────────────────────────────────────────
# 'splits' is tried in order: the first split supplies prompts until it is
# exhausted, then the next one continues.  Selection inside a split is always
# "first K documents with raw length > min_chars" (deterministic, no sampling).
DATASETS = {
    "c4": {
        "hf_name": "allenai/c4",
        "config": "en",
        "splits": ["validation"],
        "field": "text",
        # original protocol: stop scanning after num_prompts * 3 documents
        "scan_multiplier": 3,
        "default_scan_limit": None,
    },
    "wikitext": {
        "hf_name": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "splits": ["test", "validation", "train"],
        "field": "text",
        # wikitext rows are paragraphs: only ~322/4358 test rows exceed 1024
        # chars and only 9 exceed 2048, so the pool must be scanned much deeper
        # and overflows into validation/train for large N.
        "scan_multiplier": None,
        "default_scan_limit": 400000,
    },
    "code": {
        "hf_name": "codeparrot/codeparrot-clean-valid",
        "config": None,
        "splits": ["train"],  # the *-valid repo ships its data as split 'train'
        "field": "content",
        "scan_multiplier": None,
        "default_scan_limit": 400000,
    },
}

QUANT_CHOICES = ["auto", "fp16", "bf16", "nf4", "fp32"]

_DTYPE_OF = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


# ── Small helpers ─────────────────────────────────────────────────
def sanitize_key(name):
    """Filename-safe model key."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def library_versions():
    vers = {"python": sys.version.split()[0], "torch": torch.__version__}
    for mod in ("transformers", "numpy", "datasets", "bitsandbytes", "accelerate", "huggingface_hub"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            vers[mod] = None
    vers["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        vers["cuda"] = torch.version.cuda
        vers["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return vers


def script_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        return None


def register_model_id(model_id, needs_token=False):
    """Escape hatch: register an arbitrary HF id as a model key."""
    key = sanitize_key(model_id)
    MODELS[key] = {"hf_name": model_id, "needs_token": needs_token}
    return key


def resolve_revision(hf_name, config, token):
    """Resolved repo commit hash for provenance."""
    rev = getattr(config, "_commit_hash", None)
    if rev:
        return rev, "config._commit_hash"
    try:
        from huggingface_hub import HfApi
        sha = HfApi().model_info(hf_name, token=token).sha
        return sha, "HfApi.model_info"
    except Exception as e:  # offline / gated / no hub
        return None, "unavailable:%s" % type(e).__name__


def warn_block(title, lines):
    bar = "!" * 72
    print("\n" + bar)
    print("!! " + title)
    for ln in lines:
        print("!!   " + str(ln))
    print(bar + "\n", flush=True)


# ── Model loading ─────────────────────────────────────────────────
def resolve_quant(quant, est_params_b):
    """Return (quant_used, note). 'auto' = original rule: >=15B params -> NF4."""
    has_cuda = torch.cuda.is_available()
    note = ""
    if quant == "auto":
        used = "nf4" if est_params_b >= 15 else "fp16"
        if not has_cuda:
            note = "no CUDA: '%s' downgraded to fp32 for CPU run" % used
            used = "fp32"
        return used, note
    if quant == "nf4" and not has_cuda:
        raise RuntimeError("--quant nf4 requires a CUDA device (bitsandbytes)")
    if quant == "fp16" and not has_cuda:
        return "fp32", "no CUDA: fp16 downgraded to fp32 (CPU fp16 matmul unsupported)"
    return quant, note


def load_model_and_tokenizer(model_key, hf_token=None, quant="auto", device_map="auto"):
    """Load model, auto-detect head config, honour the requested precision.

    Returns (model, tokenizer, load_info).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig

    info = MODELS[model_key]
    hf_name = info["hf_name"]
    token = hf_token if info.get("needs_token") else None

    # Auto-detect head configuration from model config
    # Some models (e.g., Gemma 3) nest the text config inside a composite config
    config = AutoConfig.from_pretrained(hf_name, token=token)
    text_cfg = getattr(config, "text_config", config)  # use text_config if present
    num_q_heads = getattr(text_cfg, "num_attention_heads", None) or info.get("num_query_heads")
    num_kv_heads = getattr(text_cfg, "num_key_value_heads", None) or info.get("num_kv_heads") or num_q_heads
    num_layers = getattr(text_cfg, "num_hidden_layers", None)
    hidden_size = getattr(text_cfg, "hidden_size", None)
    # Estimate total params from config for quantization decision
    est_params_b = (hidden_size * hidden_size * 4 * num_layers) / 1e9 if hidden_size and num_layers else 0
    print(f"Config: Q={num_q_heads}, KV={num_kv_heads}, layers={num_layers}, hidden={hidden_size}, ~{est_params_b:.1f}B params")

    revision, revision_source = resolve_revision(hf_name, config, token)
    print(f"Revision: {revision} ({revision_source})")

    # Store detected config back into info for downstream use
    info["num_query_heads"] = num_q_heads
    info["num_kv_heads"] = num_kv_heads

    print(f"Loading tokenizer for {hf_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_used, quant_note = resolve_quant(quant, est_params_b)
    if quant_note:
        print(f"NOTE: {quant_note}")
    effective_device_map = device_map if torch.cuda.is_available() else None

    if quant_used == "nf4":
        print(f"Loading {hf_name} with int4 (NF4) quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            quantization_config=quant_config,
            device_map=effective_device_map,
            token=token,
            attn_implementation="eager",  # need full attention weights
        )
        torch_dtype = "float16(compute)"
    else:
        dtype = _DTYPE_OF[quant_used]
        print(f"Loading {hf_name} in {quant_used}...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            torch_dtype=dtype,
            device_map=effective_device_map,
            token=token,
            attn_implementation="eager",
        )
        torch_dtype = str(dtype).replace("torch.", "")

    model.eval()

    load_info = {
        "hf_name": hf_name,
        "revision": revision,
        "revision_source": revision_source,
        "num_query_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "est_params_b": float(est_params_b),
        "quant_used": quant_used,
        "quant_note": quant_note,
        "torch_dtype": torch_dtype,
        "device_map": effective_device_map,
        "attn_implementation": "eager",
    }
    return model, tokenizer, load_info


# ── Calibration prompts ───────────────────────────────────────────
def get_calibration_prompts(num_prompts, seq_len, dataset="c4", pool_scan_limit=None,
                            min_chars=None, cache_dir=None):
    """Deterministic first-K qualifying documents (raw length > min_chars).

    min_chars defaults to 2 * seq_len (the original protocol: 1024 chars for N=512).
    With cache_dir set, the selected pool is written once and reused by later runs
    (identical prompts across models, and no repeated deep streaming scans:
    wikitext at N=1024 has to scan ~121k documents to find 200 long enough).
    Returns (texts, meta, stats).
    """
    from datasets import load_dataset

    spec = DATASETS[dataset]
    field = spec["field"]
    if min_chars is None:
        min_chars = seq_len * 2

    if pool_scan_limit is None:
        if spec["scan_multiplier"] is not None:
            pool_scan_limit = num_prompts * spec["scan_multiplier"]  # oversample to handle short texts
        else:
            pool_scan_limit = spec["default_scan_limit"]

    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / (
            f"{dataset}__N{seq_len}__S{num_prompts}__min{min_chars}__scan{pool_scan_limit}.json")
        if cache_path.exists():
            with open(cache_path) as f:
                blob = json.load(f)
            stats = blob["stats"]
            same = (stats["dataset"] == dataset and stats["min_chars"] == min_chars
                    and stats["pool_scan_limit"] == pool_scan_limit
                    and stats["num_prompts_requested"] == num_prompts)
            if same:
                stats["cache_hit"] = True
                stats["cache_path"] = str(cache_path)
                print(f"Loaded {len(blob['texts'])} calibration prompts from cache {cache_path}")
                return blob["texts"], blob["meta"], stats
            print(f"Ignoring stale prompt cache {cache_path}")

    print(f"Loading {num_prompts} calibration prompts from {dataset} "
          f"({spec['hf_name']} / {spec['config']} / {spec['splits']}), "
          f"min_chars={min_chars}, scan_limit={pool_scan_limit}...")

    texts, meta = [], []
    scanned_total = 0
    per_split_scanned, per_split_kept = {}, {}

    for split in spec["splits"]:
        if len(texts) >= num_prompts or scanned_total >= pool_scan_limit:
            break
        load_kwargs = dict(split=split, streaming=True)
        if spec["config"] is not None:
            ds = load_dataset(spec["hf_name"], spec["config"], **load_kwargs)
        else:
            ds = load_dataset(spec["hf_name"], **load_kwargs)

        per_split_scanned[split] = 0
        per_split_kept[split] = 0
        for i, example in enumerate(ds):
            if scanned_total >= pool_scan_limit:
                break
            scanned_total += 1
            per_split_scanned[split] += 1
            text = example[field]
            if len(text) > min_chars:  # rough char filter
                texts.append(text)
                meta.append({
                    "index": len(texts) - 1,
                    "split": split,
                    "source_index": i,
                    "raw_chars": len(text),
                    "preview": text[:80],
                })
                per_split_kept[split] += 1
            if len(texts) >= num_prompts:
                break

    if len(texts) < num_prompts:
        print(f"Warning: only got {len(texts)} prompts (wanted {num_prompts})")

    stats = {
        "dataset": dataset,
        "hf_name": spec["hf_name"],
        "config": spec["config"],
        "field": field,
        "splits_available": spec["splits"],
        "splits_used": [s for s, k in per_split_kept.items() if k > 0],
        "docs_scanned": scanned_total,
        "docs_scanned_per_split": per_split_scanned,
        "docs_kept_per_split": per_split_kept,
        "min_chars": min_chars,
        "pool_scan_limit": pool_scan_limit,
        "num_prompts_requested": num_prompts,
        "num_prompts_actual": len(texts),
        "cache_hit": False,
        "cache_path": str(cache_path) if cache_path else None,
    }
    print(f"  kept {len(texts)} / scanned {scanned_total}  per-split kept: {per_split_kept}")

    texts, meta = texts[:num_prompts], meta[:num_prompts]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"stats": stats, "meta": meta, "texts": texts}, f)
        print(f"  prompt pool cached to {cache_path}")
    return texts, meta, stats


# ── G_M: message-map Gram ─────────────────────────────────────────
def dequantize_weight(linear_layer):
    """Extract the full float weight from a possibly quantized linear layer."""
    if hasattr(linear_layer, 'weight') and hasattr(linear_layer.weight, 'quant_state'):
        # BitsAndBytes 4-bit quantized
        import bitsandbytes as bnb
        return bnb.functional.dequantize_4bit(
            linear_layer.weight.data, linear_layer.weight.quant_state
        ).float()
    else:
        return linear_layer.weight.detach().float()


def find_layers(model):
    """Locate the transformer block list (models nest it differently)."""
    for path in ['model.layers', 'model.model.layers', 'language_model.model.layers']:
        obj = model
        try:
            for part in path.split('.'):
                obj = getattr(obj, part)
            if hasattr(obj, '__len__') and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    raise ValueError(f"Cannot find transformer layers in {type(model).__name__}")


def get_message_gram(model, model_key):
    """
    Compute G_M[h,h'] = <M_h, M_h'>_F for each layer.
    M_h = W_h^V W_h^O is the value-output message map.
    Returns (gram_M, diag): layer_idx -> G_M (H x H numpy array), plus NaN/Inf counts.
    """
    info = MODELS[model_key]
    num_q_heads = info["num_query_heads"]
    num_kv_heads = info["num_kv_heads"]
    heads_per_group = num_q_heads // num_kv_heads

    gram_M = {}
    diag = {}

    layers_list = find_layers(model)

    for layer_idx, layer in enumerate(layers_list):
        attn = layer.self_attn

        # Extract weight matrices, handling int4 quantization
        W_v = dequantize_weight(attn.v_proj)  # (num_kv_heads*head_dim, hidden_dim)
        W_o = dequantize_weight(attn.o_proj)  # (hidden_dim, num_q_heads*head_dim)

        hidden_dim = W_o.shape[0]
        head_dim = W_v.shape[0] // num_kv_heads

        # Reshape V weights per KV head: (num_kv_heads, head_dim, hidden_dim)
        W_v_heads = W_v.view(num_kv_heads, head_dim, hidden_dim)

        # Reshape O weights per query head: (num_q_heads, hidden_dim, head_dim)
        # W_o is (hidden_dim, num_q_heads * head_dim)
        W_o_heads = W_o.view(hidden_dim, num_q_heads, head_dim).permute(1, 0, 2)
        # Now W_o_heads[h] is (hidden_dim, head_dim)

        # M_h = W_v_{kv_group(h)} @ W_o_h^T  but we need it as a map on tokens:
        # For token x (1 x hidden_dim): x @ W_v^T @ W_o = x @ M_h
        # M_h = W_v_{group}^T @ W_o_h  (hidden_dim x hidden_dim)
        # Actually: Att_h(X) = P_h(X) @ X @ W_v_{group}^T @ W_o_h
        # So M_h = W_v_{group}^T @ W_o_h where:
        #   W_v_{group}^T is (hidden_dim, head_dim) -- the V projection for the KV group
        #   W_o_h is (head_dim, hidden_dim) -- the O slice for query head h
        # Thus M_h = W_v_{group}^T @ W_o_h has shape (hidden_dim, hidden_dim)

        # Compute G_M[h,h'] = <M_h, M_h'>_F on GPU using batched ops.
        # M_h = wv_kv(h).T @ wo_h.T, so:
        # G_M[h,h'] = <wv_h @ wv_h'.T, wo_h.T @ wo_h'>_F
        #
        # Strategy: for each head h, batch-compute against ALL h' on GPU.
        # Per head: one (head_dim, hidden) @ (Q, hidden, head_dim) batched matmul
        # This is O(Q * head_dim * hidden * head_dim) = tiny on GPU.

        # Expand V weights to per-query-head: (Q, head_dim, hidden_dim)
        kv_indices = torch.arange(num_q_heads) // heads_per_group
        wv_exp = W_v_heads[kv_indices]  # (Q, head_dim, hidden_dim)

        # Transpose O to get wo_T: (Q, head_dim, hidden_dim)
        wo_T = W_o_heads.permute(0, 2, 1)  # (Q, head_dim, hidden_dim)

        G = torch.zeros(num_q_heads, num_q_heads, device=W_v.device)

        for h in range(num_q_heads):
            # C_V_batch[h'] = wv_exp[h] @ wv_exp[h'].T for all h'
            # (head_dim, hidden) @ (Q, hidden, head_dim) -> (Q, head_dim, head_dim)
            C_V = torch.matmul(
                wv_exp[h].unsqueeze(0),       # (1, head_dim, hidden)
                wv_exp.transpose(-1, -2)       # (Q, hidden, head_dim)
            )  # (Q, head_dim, head_dim)

            # C_O_batch[h'] = wo_T[h] @ wo_T[h'].T for all h'
            C_O = torch.matmul(
                wo_T[h].unsqueeze(0),          # (1, head_dim, hidden)
                wo_T.transpose(-1, -2)          # (Q, hidden, head_dim)
            )  # (Q, head_dim, head_dim)

            # G_M[h, :] = Frobenius inner product per h'
            G[h, :] = (C_V * C_O).sum(dim=(-1, -2))

        G = G.cpu().numpy()

        # NaN/Inf accounting BEFORE any cleaning
        diag[layer_idx] = {
            "gm_nan": int(np.isnan(G).sum()),
            "gm_inf": int(np.isinf(G).sum()),
            "gm_trace": float(np.trace(np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0))),
            "weight_nan_v": int(torch.isnan(W_v).sum().item()),
            "weight_nan_o": int(torch.isnan(W_o).sum().item()),
        }

        # Free GPU memory from dequantized weights
        del W_v, W_o, W_v_heads, W_o_heads, wv_exp, wo_T
        torch.cuda.empty_cache()
        gram_M[layer_idx] = G

        if layer_idx % 10 == 0:
            print(f"  G_M computed for layer {layer_idx}")

    return gram_M, diag


# ── G_P: attention Gram ───────────────────────────────────────────
def measure_attention_gram(model, tokenizer, prompts, seq_len, model_key, collect_per_prompt=False):
    """
    Accumulate G_P^C[h,h'] = sum_s <P_h(X^s), P_h'(X^s)>_F over calibration prompts.

    Returns (gram_P, per_prompt, diag):
      gram_P     : dict layer_idx -> G_P (H x H, float64 accumulator)
      per_prompt : float32 array [S, L, H, H] with the single-prompt contribution
                   <P_h(X^s), P_h'(X^s)>_F, or None if collect_per_prompt is False
      diag       : per-layer NaN/Inf counts + per-prompt token counts
    """
    info = MODELS[model_key]
    num_q_heads = info["num_query_heads"]
    num_kv_heads = info["num_kv_heads"]
    layers_list = find_layers(model)
    num_layers = len(layers_list)

    # Initialize accumulators
    gram_P = {l: np.zeros((num_q_heads, num_q_heads)) for l in range(num_layers)}

    per_prompt = None
    if collect_per_prompt:
        nbytes = len(prompts) * num_layers * num_q_heads * num_q_heads * 4
        print(f"  allocating per-prompt G_P buffer: "
              f"[{len(prompts)}, {num_layers}, {num_q_heads}, {num_q_heads}] float32 "
              f"= {nbytes / 1e6:.0f} MB")
        per_prompt = np.zeros((len(prompts), num_layers, num_q_heads, num_q_heads), dtype=np.float32)

    diag = {
        "attn_nan_per_layer": {l: 0 for l in range(num_layers)},
        "attn_inf_per_layer": {l: 0 for l in range(num_layers)},
        "gram_nan_per_layer": {l: 0 for l in range(num_layers)},
        "gram_inf_per_layer": {l: 0 for l in range(num_layers)},
        "missing_per_layer": {l: 0 for l in range(num_layers)},
        "num_tokens": [],
        "heads_returned": None,
    }

    # Hook to capture attention weights
    attn_weights_store = {}

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights_store[layer_idx] = output[1].detach()
        return hook_fn

    # Register hooks
    hooks = []
    for layer_idx, layer in enumerate(layers_list):
        h = layer.self_attn.register_forward_hook(make_hook(layer_idx), with_kwargs=True)
        hooks.append(h)

    print(f"Running {len(prompts)} forward passes for attention Gram accumulation...")
    for s, text in enumerate(prompts):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        diag["num_tokens"].append(int(input_ids.shape[1]))

        attn_weights_store.clear()

        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
            )

        # Accumulate Gram matrices
        for layer_idx in range(num_layers):
            if layer_idx not in attn_weights_store:
                diag["missing_per_layer"][layer_idx] += 1
                continue

            # attn_w: (1, num_heads, seq_len, seq_len)
            attn_w = attn_weights_store[layer_idx][0].float()  # (H, N, N)
            H_actual = attn_w.shape[0]

            # NaN/Inf accounting on the raw attention maps (before any cleaning)
            n_nan = int(torch.isnan(attn_w).sum().item())
            n_inf = int(torch.isinf(attn_w).sum().item())
            diag["attn_nan_per_layer"][layer_idx] += n_nan
            diag["attn_inf_per_layer"][layer_idx] += n_inf

            # If GQA, attention weights may be num_kv_heads not num_q_heads
            # Need to expand to full query head count
            if H_actual == num_kv_heads and num_q_heads != num_kv_heads:
                heads_per_group = num_q_heads // num_kv_heads
                attn_w = attn_w.repeat_interleave(heads_per_group, dim=0)
            elif H_actual != num_q_heads:
                # Some models return per-query-head attention
                pass  # use as-is if H_actual == num_q_heads

            # Flatten each head's attention matrix: (H, N*N)
            H_use = attn_w.shape[0]
            N_actual = attn_w.shape[1]
            P_flat = attn_w.reshape(H_use, -1)  # (H, N^2)

            # G_P += P_flat @ P_flat^T
            G_batch = (P_flat @ P_flat.T).cpu().numpy()
            gram_P[layer_idx][:H_use, :H_use] += G_batch

            if per_prompt is not None:
                per_prompt[s, layer_idx, :H_use, :H_use] = G_batch.astype(np.float32)

            if not np.isfinite(G_batch).all():
                diag["gram_nan_per_layer"][layer_idx] += int(np.isnan(G_batch).sum())
                diag["gram_inf_per_layer"][layer_idx] += int(np.isinf(G_batch).sum())

            if diag["heads_returned"] is None:
                diag["heads_returned"] = int(H_actual)

        if (s + 1) % 20 == 0:
            print(f"  Processed {s+1}/{len(prompts)} prompts")

    # Remove hooks
    for h in hooks:
        h.remove()

    return gram_P, per_prompt, diag


# ── Spectrum ──────────────────────────────────────────────────────
def compute_spectrum(gram_P, gram_M, gp_diag=None, gm_diag=None):
    """
    Compute the routing-message spectrum per layer.
    Squared singular values of Psi_C = eigenvalues of G_P^{1/2} G_M G_P^{1/2}.

    NaN/Inf are counted BEFORE nan_to_num and reported per layer; a layer with
    zero total energy or any NaN/Inf is flagged 'corrupted' instead of being
    silently zeroed.
    """
    results = {}

    for layer_idx in sorted(gram_P.keys()):
        GP = gram_P[layer_idx].copy()
        GM = gram_M[layer_idx].copy()

        # ── NaN/Inf accounting BEFORE any cleaning ──
        gp_nan = int(np.isnan(GP).sum())
        gp_inf = int(np.isinf(GP).sum())
        gm_nan = int(np.isnan(GM).sum())
        gm_inf = int(np.isinf(GM).sum())

        # Clean NaN/Inf before any linear algebra
        GP = np.nan_to_num(GP, nan=0.0, posinf=0.0, neginf=0.0)
        GM = np.nan_to_num(GM, nan=0.0, posinf=0.0, neginf=0.0)
        # Symmetrize (numerical noise can break symmetry)
        GP = 0.5 * (GP + GP.T)
        GM = 0.5 * (GM + GM.T)

        gp_trace = float(np.trace(GP))
        gm_trace = float(np.trace(GM))

        # Compute squared singular values of Psi_C = eigenvalues of G_P G_M
        # Use SVD of the product as a numerically robust alternative
        try:
            # Method 1: symmetric form G_P^{1/2} G_M G_P^{1/2}
            eigvals_P, eigvecs_P = np.linalg.eigh(GP)
            eigvals_P = np.maximum(eigvals_P, 0)
            sqrt_P = eigvecs_P @ np.diag(np.sqrt(eigvals_P)) @ eigvecs_P.T
            sym_product = sqrt_P @ GM @ sqrt_P
            sigma_sq = np.linalg.eigvalsh(sym_product)
            sigma_sq = np.maximum(sigma_sq, 0)
        except np.linalg.LinAlgError:
            # Fallback: use SVD of G_P @ G_M directly
            try:
                sigma_sq = np.linalg.svd(GP @ GM, compute_uv=False) ** 1  # singular values = sqrt of eigenvalues... no
                # Actually eigenvalues of G_P G_M = squared singular values of Psi
                # Use: nonzero eigenvalues of AB = nonzero eigenvalues of BA
                # So eigenvalues of G_P G_M are real and nonneg (since both PSD)
                product = GP @ GM
                eigs = np.linalg.eigvals(product)
                sigma_sq = np.maximum(np.real(eigs), 0)
            except np.linalg.LinAlgError:
                # Last resort: just use diagonal approximation
                sigma_sq = np.array([GP[i, i] * GM[i, i] for i in range(GP.shape[0])])
                sigma_sq = np.maximum(sigma_sq, 0)

        sigma_sq = np.sort(sigma_sq)[::-1]  # descending

        # Also compute G_P-only and G_M-only spectra (with full NaN protection)
        def safe_eigvalsh(M):
            M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
            M = 0.5 * (M + M.T)
            try:
                return np.sort(np.maximum(np.linalg.eigvalsh(M), 0))[::-1]
            except np.linalg.LinAlgError:
                try:
                    return np.sort(np.maximum(np.real(np.linalg.eigvals(M)), 0))[::-1]
                except np.linalg.LinAlgError:
                    return np.sort(np.abs(np.diag(M)))[::-1]

        eigvals_P_only = safe_eigvalsh(GP)
        eigvals_M_only = safe_eigvalsh(GM)

        sigma = np.sqrt(sigma_sq)
        total_energy = sigma_sq.sum()

        # Effective rank
        if total_energy > 0:
            r_eff = (sigma.sum() ** 2) / total_energy
        else:
            r_eff = 0.0

        # Tail energy T_R for various R
        tail_energy = {}
        for R in [1, 2, 4, 8, 16, 32]:
            if R < len(sigma_sq):
                tail = sigma_sq[R:].sum() / total_energy if total_energy > 0 else 0.0
            else:
                tail = 0.0
            tail_energy[str(R)] = float(tail)

        # ── corruption diagnostics (never silently zero) ──
        attn_nan = int(gp_diag["attn_nan_per_layer"].get(layer_idx, 0)) if gp_diag else 0
        attn_inf = int(gp_diag["attn_inf_per_layer"].get(layer_idx, 0)) if gp_diag else 0
        gram_nan_stream = int(gp_diag["gram_nan_per_layer"].get(layer_idx, 0)) if gp_diag else 0
        gram_inf_stream = int(gp_diag["gram_inf_per_layer"].get(layer_idx, 0)) if gp_diag else 0
        missing = int(gp_diag["missing_per_layer"].get(layer_idx, 0)) if gp_diag else 0
        weight_nan = 0
        if gm_diag and layer_idx in gm_diag:
            weight_nan = int(gm_diag[layer_idx]["weight_nan_v"] + gm_diag[layer_idx]["weight_nan_o"])

        nan_total = gp_nan + gm_nan + attn_nan + gram_nan_stream + weight_nan
        inf_total = gp_inf + gm_inf + attn_inf + gram_inf_stream

        reasons = []
        if nan_total > 0:
            reasons.append(f"NaN count {nan_total}")
        if inf_total > 0:
            reasons.append(f"Inf count {inf_total}")
        if not (gp_trace > 0):
            reasons.append(f"G_P trace not positive ({gp_trace:g})")
        if not (gm_trace > 0):
            reasons.append(f"G_M trace not positive ({gm_trace:g})")
        if not (total_energy > 0):
            reasons.append(f"zero total energy ({total_energy:g})")
        if missing > 0:
            reasons.append(f"{missing} prompts returned no attention map")

        results[layer_idx] = {
            "sigma_squared": sigma_sq.tolist(),
            "sigma": sigma.tolist(),
            "r_eff": float(r_eff),
            "tail_energy": tail_energy,
            "total_energy": float(total_energy),
            "gp_eigenvalues": eigvals_P_only.tolist(),
            "gm_eigenvalues": eigvals_M_only.tolist(),
            # ── diagnostics ──
            "gp_trace": gp_trace,
            "gm_trace": gm_trace,
            "nan_count": nan_total,
            "inf_count": inf_total,
            "nan_breakdown": {
                "attn_maps": attn_nan, "gram_P_stream": gram_nan_stream,
                "gram_P_total": gp_nan, "gram_M": gm_nan, "weights": weight_nan,
            },
            "inf_breakdown": {
                "attn_maps": attn_inf, "gram_P_stream": gram_inf_stream,
                "gram_P_total": gp_inf, "gram_M": gm_inf,
            },
            "missing_attention_prompts": missing,
            "corrupted": bool(reasons),
            "corruption_reasons": reasons,
        }

    return results


# ── Gram dump ─────────────────────────────────────────────────────
def dump_stem(model_key, dataset, seq_len, num_prompts, quant_used):
    return f"{sanitize_key(model_key)}__{dataset}__N{seq_len}__S{num_prompts}__{quant_used}"


def save_grams(dump_dir, stem, gram_P, gram_M, per_prompt, prompt_meta, config_payload, spectrum):
    """Write one .npz with G_M, per-prompt G_P contributions and full provenance."""
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    path = dump_dir / (stem + ".npz")

    layer_ids = sorted(gram_P.keys())
    G_M = np.stack([gram_M[l] for l in layer_ids]).astype(np.float32)
    G_P_total = np.stack([gram_P[l] for l in layer_ids]).astype(np.float64)

    payload = {
        "G_M": G_M,                                   # [L, H, H] float32
        "G_P_total": G_P_total,                       # [L, H, H] float64 (accumulator)
        "layer_index": np.array(layer_ids, dtype=np.int32),
        "prompt_index": np.array([m["index"] for m in prompt_meta], dtype=np.int32),
        "prompt_source_index": np.array([m["source_index"] for m in prompt_meta], dtype=np.int32),
        "prompt_split": np.array([m["split"] for m in prompt_meta], dtype=object).astype("U32"),
        "prompt_raw_chars": np.array([m["raw_chars"] for m in prompt_meta], dtype=np.int32),
        "prompt_preview": np.array([m["preview"] for m in prompt_meta], dtype=object).astype("U80"),
        "prompt_num_tokens": np.array(config_payload.get("prompt_num_tokens", []), dtype=np.int32),
        "layer_corrupted": np.array([bool(spectrum[l]["corrupted"]) for l in layer_ids], dtype=bool),
        "config_json": np.array(json.dumps(config_payload, indent=2)),
    }
    if per_prompt is not None:
        payload["G_P_per_prompt"] = per_prompt          # [S, L, H, H] float32

    np.savez(path, **payload)
    size_mb = path.stat().st_size / 1e6
    print(f"Grams dumped to {path} ({size_mb:.1f} MB)")
    return str(path), size_mb


# ── Per-model pipeline ────────────────────────────────────────────
def run_model(model_key, prompts, prompt_meta, seq_len, hf_token=None, args=None,
              prompt_stats=None):
    """Full pipeline for one model."""
    print(f"\n{'='*60}")
    print(f"Model: {model_key}")
    print(f"{'='*60}")

    t0 = time.time()

    # Load model
    model, tokenizer, load_info = load_model_and_tokenizer(
        model_key, hf_token, quant=args.quant, device_map=args.device_map
    )

    # Compute G_M from weights
    print("\nComputing message Gram G_M...")
    gram_M, gm_diag = get_message_gram(model, model_key)

    # Compute G_P from attention maps
    print("\nComputing attention Gram G_P...")
    gram_P, per_prompt, gp_diag = measure_attention_gram(
        model, tokenizer, prompts, seq_len, model_key,
        collect_per_prompt=bool(args.dump_grams),
    )

    # Compute spectrum
    print("\nComputing routing-message spectrum...")
    spectrum = compute_spectrum(gram_P, gram_M, gp_diag=gp_diag, gm_diag=gm_diag)

    # Summary
    info = MODELS[model_key]
    print(f"\n--- Summary for {model_key} ---")
    print(f"Query heads: {info['num_query_heads']}, KV heads: {info['num_kv_heads']}")
    print(f"Layers: {len(spectrum)}")
    for layer_idx in [0, len(spectrum) // 4, len(spectrum) // 2, 3 * len(spectrum) // 4, len(spectrum) - 1]:
        if layer_idx in spectrum:
            s = spectrum[layer_idx]
            print(f"  Layer {layer_idx:3d}: r_eff={s['r_eff']:.1f}, "
                  f"T_1={s['tail_energy'].get('1', 0):.4f}, "
                  f"T_4={s['tail_energy'].get('4', 0):.4f}, "
                  f"T_8={s['tail_energy'].get('8', 0):.4f}")

    corrupted_layers = [l for l in sorted(spectrum) if spectrum[l]["corrupted"]]
    total_nan = sum(spectrum[l]["nan_count"] for l in spectrum)
    total_inf = sum(spectrum[l]["inf_count"] for l in spectrum)
    if corrupted_layers:
        lines = [
            f"model={model_key}  quant={load_info['quant_used']}  dtype={load_info['torch_dtype']}",
            f"dataset={args.dataset}  N={seq_len}  S={len(prompts)}",
            f"{len(corrupted_layers)}/{len(spectrum)} layers flagged corrupted: {corrupted_layers[:20]}"
            + (" ..." if len(corrupted_layers) > 20 else ""),
            f"total NaN={total_nan}  total Inf={total_inf}",
        ]
        for l in corrupted_layers[:5]:
            lines.append(f"layer {l}: {spectrum[l]['corruption_reasons']}")
        lines.append("Values are NOT trustworthy. For Gemma-2 style fp16 overflow, re-run with --quant bf16.")
        warn_block("CORRUPTED MEASUREMENT DETECTED", lines)
    else:
        print("Integrity: OK (no NaN/Inf, all layers have positive G_P and G_M energy)")

    elapsed = time.time() - t0
    print(f"\nTotal time for {model_key}: {elapsed:.0f}s")

    result = {
        "model": model_key,
        "hf_name": info["hf_name"],
        "num_query_heads": info["num_query_heads"],
        "num_kv_heads": info["num_kv_heads"],
        "num_layers": len(spectrum),
        "num_prompts": len(prompts),
        "seq_len": seq_len,
        "elapsed_seconds": elapsed,
        # ── rebuttal-edition provenance ──
        "dataset": args.dataset,
        "dataset_info": prompt_stats,
        "quant_requested": args.quant,
        "quant_used": load_info["quant_used"],
        "quant_note": load_info["quant_note"],
        "torch_dtype": load_info["torch_dtype"],
        "device_map": load_info["device_map"],
        "revision": load_info["revision"],
        "revision_source": load_info["revision_source"],
        "est_params_b": load_info["est_params_b"],
        "hidden_size": load_info["hidden_size"],
        "seed": args.seed,
        "heads_returned_by_library": gp_diag["heads_returned"],
        "prompt_num_tokens": gp_diag["num_tokens"],
        "prompts": prompt_meta,
        "library_versions": library_versions(),
        "script_sha256": script_sha256(),
        "timestamp_utc": utcnow(),
        "corrupted_layers": corrupted_layers,
        "any_corrupted": bool(corrupted_layers),
        "total_nan_count": int(total_nan),
        "total_inf_count": int(total_inf),
        "layers": {str(k): v for k, v in spectrum.items()},
    }

    # Optional Gram dump
    if args.dump_grams:
        stem = dump_stem(model_key, args.dataset, seq_len, args.num_prompts, load_info["quant_used"])
        config_payload = {k: v for k, v in result.items() if k != "layers"}
        path, size_mb = save_grams(args.dump_grams, stem, gram_P, gram_M, per_prompt,
                                   prompt_meta, config_payload, spectrum)
        result["dump_path"] = path
        result["dump_size_mb"] = size_mb
        result["dump_stem"] = stem

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen3-8b"],
                        choices=list(MODELS.keys()))
    parser.add_argument("--model_id", nargs="+", default=None,
                        help="Arbitrary HF ids (escape hatch); head config is introspected.")
    parser.add_argument("--model_id_needs_token", action="store_true",
                        help="Pass HF_TOKEN to the --model_id checkpoints (gated repos).")
    parser.add_argument("--num_prompts", type=int, default=200)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Prefer the HF_TOKEN environment variable; never hardcode a token.")
    parser.add_argument("--dataset", type=str, default="c4", choices=list(DATASETS.keys()))
    parser.add_argument("--quant", type=str, default="auto", choices=QUANT_CHOICES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump_grams", type=str, default=None,
                        help="Directory for per-model .npz Gram dumps.")
    parser.add_argument("--pool_scan_limit", type=int, default=None,
                        help="Max documents scanned when building the calibration pool.")
    parser.add_argument("--min_chars", type=int, default=None,
                        help="Raw-length filter (default 2 * seq_len, the original protocol).")
    parser.add_argument("--prompt_cache", type=str, default=None,
                        help="Directory for cached calibration pools (identical prompts across "
                             "models; avoids re-scanning wikitext for every run).")
    parser.add_argument("--device_map", type=str, default="auto")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_keys = list(args.models) if args.models else []
    if args.model_id:
        # --model_id replaces the default --models value unless --models was given explicitly
        explicit_models = any(a.startswith("--models") for a in sys.argv)
        if not explicit_models:
            model_keys = []
        for mid in args.model_id:
            model_keys.append(register_model_id(mid, needs_token=args.model_id_needs_token))

    print("Library versions: " + json.dumps(library_versions()))

    # Load calibration prompts once
    prompts, prompt_meta, prompt_stats = get_calibration_prompts(
        args.num_prompts, args.seq_len, dataset=args.dataset,
        pool_scan_limit=args.pool_scan_limit, min_chars=args.min_chars,
        cache_dir=args.prompt_cache,
    )
    print(f"Loaded {len(prompts)} calibration prompts")
    if len(prompts) < args.num_prompts:
        warn_block("CALIBRATION POOL SHORTFALL", [
            f"requested {args.num_prompts}, got {len(prompts)}",
            f"dataset={args.dataset} min_chars={prompt_stats['min_chars']} "
            f"scan_limit={prompt_stats['pool_scan_limit']}",
            "raise --pool_scan_limit or lower --min_chars",
        ])

    all_results = {}
    for model_key in model_keys:
        result = run_model(model_key, prompts, prompt_meta, args.seq_len,
                           hf_token=hf_token, args=args, prompt_stats=prompt_stats)
        all_results[model_key] = result

        # Save intermediate results
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Intermediate results saved to {args.output}")

    any_corrupted = any(r["any_corrupted"] for r in all_results.values())
    print(f"\n{'='*60}")
    print(f"All done. Final results in {args.output}")
    if any_corrupted:
        print("WARNING: at least one model has corrupted layers (see 'corrupted_layers').")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
