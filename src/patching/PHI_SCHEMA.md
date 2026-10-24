# phi JSON schema (v1): shared contract between `export_phi.py` and `route_patch_ppl.py`

`phi` is a **per-layer head-merge map**. For decoder layer `l` and query head
`h`, `phi_l(h)` names the *representative* head whose post-softmax attention
matrix head `h` will reuse:

```
patched:    Att_h(X) = P_{phi_l(h)}(X) @ X @ W^V_{kappa(h)} @ W^O_h
unpatched:  Att_h(X) = P_h(X)          @ X @ W^V_{kappa(h)} @ W^O_h
```

Only the routing matrix `P` is shared. The value/output path of head `h`
(`W^V_{kappa(h)} W^O_h`, with `kappa(h) = h // (H_q // H_kv)` under GQA) is
**never** touched, and never permuted.

`export_phi.py` (the E4a -> E4b bridge, built on `analyze_merge.py`'s own map
constructors) **writes** these files; `route_patch_ppl.py` (E4b)
**reads and applies** them. Neither script infers a phi from the other's
internals; the file is the whole interface.

## File format

Example: the header and first layer of the shipped
`data/patching/phis/phi_Qwen3-8B_greedy_R8.json`, as written by `export_phi.py`:

```json
{
  "schema_version": "1",
  "model": "Qwen/Qwen3-8B",
  "num_query_heads": 32,
  "num_kv_heads": 8,
  "num_layers": 36,
  "method": "greedy",
  "R": 8,
  "meta": {
    "source": "export_phi.py",
    "dump": "qwen3-8b__c4__N512__S400__fp16.npz",
    "sel_range": "0:200",
    "seed": 0,
    "dataset": "c4",
    "seq_len": 512
  },
  "layers": {
    "0": [27, 1, 2, 27, 4, 29, 4, 4, 29, 2, 4, 4, 12, 27, 27, 2,
          17, 17, 18, 17, 4, 29, 18, 4, 4, 29, 18, 27, 18, 29, 4, 4],
    "1": [...],
    "35": [...]
  }
}
```

### Fields

| key | required | type | meaning |
|---|---|---|---|
| `layers` | **yes** | object | decoder-layer index (as a **string**) -> list of `H_q` ints |
| `schema_version` | no | string | `"1"` |
| `model` | no | string | HF checkpoint the phi was derived from. If present and it disagrees with the model being evaluated, `route_patch_ppl.py` prints a loud warning (it does not abort, so a phi can be transferred deliberately). |
| `num_query_heads` | no | int | If present it **must** equal the model's `H_q`, else the run aborts. |
| `num_kv_heads`, `num_layers` | no | int | informational |
| `method`, `R` | no | any | provenance; copied verbatim into the E4b result JSON |
| `meta` | no | object | free-form provenance |

### Constraints on `layers[l]`

1. Length is exactly `H_q` (query heads, **not** KV heads).
2. Every entry is an integer in `[0, H_q)`.
3. Indices are **query-head indices after `repeat_kv`**, the same index space
   as `attn_weights[:, h, :, :]` in the eager attention forward. They are *not*
   KV-group indices.
4. **Idempotence**: `phi[phi[h]] == phi[h]`. Representatives must map to
   themselves. Violations are a warning by default and an abort under
   `--strict_phi`. (A non-idempotent map is still applied literally. The map
   is a gather, never a chain: head `h` uses `P_{phi[h]}`, full stop.)
5. Layer keys are JSON strings of ints in `[0, num_layers)`. Layers **absent**
   from `layers` are treated as identity, with a printed warning. Emitting all
   layers is strongly preferred.

### Number of representatives

`R` is descriptive metadata only; the harness derives the real compression from
the map itself and records it in the output JSON as
`phi_stats.distinct_reps_per_layer` (plus mean/min/max). A budget of `R`
representatives per layer means `len(set(layers[l])) == R`.

## Built-in generators (baselines, no file needed)

`route_patch_ppl.py --method ...` produces these on the fly; all are idempotent:

| method | definition |
|---|---|
| `identity` | `phi[h] = h`. Must reproduce the unpatched model exactly (gate i). |
| `first_R` | heads `0..R-1` are representatives; head `h >= R` uses `h % R`. |
| `random` | per layer, `R` representatives drawn without replacement with `np.random.default_rng(seed*100003 + layer)`; every other head is assigned a uniformly random representative; representatives are forced to be fixed points. |

`--dump_phi out.json` writes the generated map in exactly this schema, so a
random/first_R baseline can be re-applied byte-identically later, and so the
maps of `export_phi.py` can be diffed against a baseline of the same `R`.

## Producer: `export_phi.py` (E4a -> E4b bridge)

`analyze_merge.py` reports merge *error* per method and rank and records the
representative sets, but does not itself write phi maps. `export_phi.py`
recomputes the maps with `analyze_merge`'s own constructors (`greedy_path`,
`eigen_path`, `similarity_path`, `random_phis`) from the same Gram dump and
writes this schema:

```bash
python src/analysis/export_phi.py \
    --npz results/measure/npz/qwen3-8b__c4__N512__S400__fp16.npz \
    --methods greedy eigen similarity --ranks 4 8 16 --sel_range 0:200 \
    --model Qwen/Qwen3-8B --out_dir results/patching/phis_check \
    --check_json results/analysis/e4a_merge.json   # <- verifies the reps agree
```

The consistency check writes to its own folder (`phis_check`): file names
depend only on checkpoint, method and `R`, so a check run that writes to
`results/patching/phis` replaces the maps exported there by the Appendix I.3
command of the README.

`--check_json` walks
`results["models"][i]["analyses"][j]["layers"][k]["ranks"][R]["reps"][method]`
of `e4a_merge.json`, using only the analysis selected on the same prompts: the
held-out analysis (selected on prompts 0-199, the `--fit-range` of
`analyze_merge.py`) for `--sel_range 0:200`, the in-sample analysis (all prompts)
for `--sel_range all`. It fails (exit 1) if any exported representative set
disagrees with the one `analyze_merge.py` reported, or if nothing could be
compared. The random maps are not part of this check, because `analyze_merge.py`
records no representatives for its random draws.

## Producer snippet (the format in a few lines of Python)

```python
import json

def write_phi(path, hf_name, H_q, H_kv, n_layers, phi_by_layer, method, R, meta=None):
    """phi_by_layer: {int layer -> list[int] of length H_q}"""
    for l, v in phi_by_layer.items():
        assert len(v) == H_q, (l, len(v))
        assert all(0 <= int(x) < H_q for x in v), l
        assert all(v[v[h]] == v[h] for h in range(H_q)), f"layer {l} phi not idempotent"
    with open(path, "w") as f:
        json.dump({
            "schema_version": "1",
            "model": hf_name,
            "num_query_heads": int(H_q),
            "num_kv_heads": int(H_kv),
            "num_layers": int(n_layers),
            "method": method,
            "R": R,
            "meta": meta or {},
            "layers": {str(l): [int(x) for x in phi_by_layer[l]] for l in sorted(phi_by_layer)},
        }, f, indent=2)
```

## Consumer contract (what E4b guarantees)

* The map is applied to **post-softmax** probabilities, after `repeat_kv`, on
  query-head axis 1 of `[B, H_q, N, N]`, as a gather:
  `attn_weights = attn_weights[:, phi, :, :]`.
* Value/output states are never permuted; GQA `kappa(h) = h // n_rep` ordering
  is asserted numerically by gate (iv).
* The identity map reproduces the unpatched model bit-for-bit (gate i), so
  `--method identity` is a legitimate baseline for a phi run.
