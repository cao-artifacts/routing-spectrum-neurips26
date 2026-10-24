# Robustness measurements (Appendix H)

Spectrum measurements made during the author-response period for the
robustness studies of Appendix H (labelled E1 to E3 in the code), and the
held-out, bootstrap and cross-domain results computed from their Gram dumps.
The measurements need GPUs and the (partly gated) model weights; the Gram
dumps behind `e1_heldout.*` are not shipped (see below).

## Hardware and software

- One NVIDIA H100 80GB HBM3 per run, except the Llama-3-70B matched pair
  (`llama3-70b__c4__N512__S200__{fp16,nf4}`), which ran on a host with two
  H100 80GB GPUs (`device_map=auto`); its fp16 arm needs both.
- Recorded in every cell (`library_versions`): Python 3.11.10,
  PyTorch 2.4.1+cu124 (CUDA 12.4), Transformers 4.57.6, Datasets 5.0.0,
  bitsandbytes 0.45.5, Accelerate 1.14.0, NumPy 1.26.3, huggingface_hub 0.36.2.
- Every cell records the checkpoint commit it used (`revision`; one commit per
  model across all cells) and the SHA-256 of the measurement script
  (`script_sha256`), which equals the SHA-256 of `src/measure/experiment.py`.

## Files

`cells/<model>__<dataset>__N<N>__S<S>__<precision>.json`: one measurement per
file, written by `src/measure/experiment.py` (151 files):

| Cells | Count | Used for |
|---|---:|---|
| `*__c4__N512__S400__auto` | 14 | H.1 measurement: 400 C4 prompts; prompts 0-199 are the Table 1 calibration set, prompts 200-399 the held-out set |
| `*__{c4,wikitext,code}__N{256,512,1024}__S200__auto` | 126 | H.2 grid (14 models x 3 corpora x 3 lengths) |
| `qwen3-8b__code__N{256,512,1024}__S200__bf16` | 3 | H.2: bf16 reruns of the three Qwen3-8B code cells |
| `{qwen3-14b,mistral-nemo-12b,qwen3-32b,llama3-70b}__c4__N512__S200__{fp16,nf4}` | 8 | H.4 matched precision pairs |

The three fp16 (`auto`) Qwen3-8B code cells overflowed: the NaN/Inf sentinels
flagged 33 of 36 layers (`any_corrupted: true`). They are kept as the record of
that event; `src/analysis/summarize_grids.py` uses the bf16 reruns instead.

`auto` precision resolves to NF4 only for Llama-3-70B; in these runs Qwen3-32B
is measured in fp16, unlike Table 1, which used NF4 for it (the H.4 pair
measures the difference). Attention maps are computed in the load dtype (fp16,
or bf16 for the three reruns); value and output weights are cast to
float32, after dequantization for NF4, for G_M.

Per-layer fields are those of `../table1/` plus NaN/Inf counts, a `corrupted`
flag with reasons, and the traces of G_P and G_M. Per-cell provenance:
`dataset_info` (corpus, split order, documents scanned and kept per split,
length filter, scan limit, prompt-cache file name), `prompts` (index, split,
position in the stream and character length of every prompt),
`prompt_num_tokens`, `quant_requested`, `quant_used`, `torch_dtype`,
`revision`, `library_versions`, `script_sha256`, `timestamp_utc`,
`elapsed_seconds` and the file name of the Gram dump.

`e1_heldout.json`, `e1_heldout.md`: output of `src/analysis/analyze_heldout.py`
on the C4 S=400 Gram dumps (H.1 and H.3): held-out split (fit on prompts
0-199, evaluate on 200-399, R in {1, 4, 8}), bootstrap (500 resamples of 200
of the 400 prompts, seed 0) and cross-domain transfer (fit on C4, score on the
N=512 WikiText-103 and CodeParrot dumps of the same model; the bf16 dump for
Qwen3-8B code). The JSON has per-layer values, the Markdown file the
layer-median tables.

## Calibration pools

Selection is deterministic: the first documents in stream order whose raw
length exceeds 2N characters, identical for every model (cached pool).

- C4 (`allenai/c4`, `en`, validation): the scan stops after 3S documents. At
  N = 1024 only 173 of the first 600 documents qualify, so those 14 cells have
  S = 173.
- WikiText-103 (`wikitext-103-raw-v1`): test, then validation, then train. At
  N = 1024 the 200 prompts are 9 test, 2 validation and 189 train documents.
- CodeParrot (`codeparrot/codeparrot-clean-valid`, field `content`).

## Redactions

The source files were changed in two ways only: the 80-character text preview
of every prompt was removed, so no corpus text is shipped, and absolute paths
of the measurement machines (`dataset_info.cache_path`, `dump_path`, and
`path` in `e1_heldout.json`) were reduced to file names.

Not shipped: the per-prompt Gram dumps (151 `.npz` files, about 9 GB, which
also contain the prompt previews) and the cached prompt pools (full corpus
text). `src/measure/run_batch.py` regenerates both.

## Re-runs

Each cell is a single-run snapshot (one measurement, no repetitions averaged).
Re-measurements can differ slightly with other GPUs, library versions, or
upstream dataset and checkpoint revisions. On the same hardware and software the
measurement repeated exactly: the four settings measured twice, once as an H.2
cell (`auto`) and once as an H.4 arm (`qwen3-14b`, `mistral-nemo-12b` and
`qwen3-32b` in fp16, `llama3-70b` in NF4), have identical spectra. The H.2 and
H.4 tables regenerate on CPU from these cells
(`python src/analysis/summarize_grids.py`); the regenerated Markdown tables are
byte-identical to the ones used for the paper.
