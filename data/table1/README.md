# Table 1 measurements

Per-layer routing-message spectra of the 14 GQA checkpoints behind Table 1,
Figures 1 and 2 (Section 6) and the per-layer range figure in Appendix D.
These are the measurement files of the original submission, unchanged.
They cannot be regenerated without GPUs and the (partly gated) model weights;
`src/measure/run_table1.sh` re-measures them.

## How they were measured

- Hardware: one NVIDIA H100 80GB GPU.
- Software: PyTorch 2.4 and Transformers 4.57 (patch versions and model
  revisions were not recorded in these files).
- Calibration set: `allenai/c4`, English validation split, streamed; the first
  200 documents with more than 1024 characters among the first 600 streamed,
  tokenized with each model's own tokenizer and truncated to N = 512 tokens
  (no padding).
- Attention maps come from eager attention (`output_attentions=True`) in fp16.
  Qwen3-32B and Llama-3-70B were loaded in 4-bit NF4 and the other twelve
  models in fp16; value and output weights are cast to float32, after
  dequantization for NF4, before forming the message Gram G_M.

## Files

| File | Model keys |
|---|---|
| `results_small.json` | `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, `qwen3-14b`, plus the pilot `mistral-7b` |
| `results.json` | `qwen3-8b` |
| `results_part3.json` | `qwen3-32b` |
| `results_llama32.json` | `llama3.2-1b`, `llama3.2-3b` |
| `results_round2.json` | `llama3-8b` |
| `results_part2.json` | `llama3-70b` |
| `results_round2b.json` | `gemma2-2b`, `gemma2-9b`, plus the pilot `gemma2-27b` |
| `results_round2c.json` | `mistral-7b-v03`, `mistral-nemo-12b` |

Each file maps a model key to: `hf_name`, `num_query_heads`, `num_kv_heads`,
`num_layers`, `num_prompts`, `seq_len`, `elapsed_seconds` (wall time from the
start of model loading to the finished spectrum, as in Appendix G) and `layers`. For every layer index, `layers` holds
`sigma_squared` and `sigma` (squared and plain singular values of the stacked
routing-message operator, descending), `r_eff`, `tail_energy` (T_R for
R in {1, 2, 4, 8, 16, 32}; defined as 0 when R >= H_q, so Gemma-2-2B's
T_8 = 0 is definitional), `total_energy`, and the separate eigenvalues of
G_P and G_M (`gp_eigenvalues`, `gm_eigenvalues`).

Two pilot runs are not part of the paper and are skipped by
`src/figures/plot_figures.py`: `mistral-7b` (Mistral-7B-v0.1, superseded by
v0.3) and `gemma2-27b`, whose fp16 run overflowed so that 30 of its 46 layers
have zero energy.

## Re-runs

Each file is a single-run snapshot: one measurement per model, no repetitions
averaged. Re-runs can differ slightly with other GPUs, library versions or
upstream checkpoint and dataset revisions. The independent re-measurement made
during the author-response period at the same setting and precision
(`../robustness/cells/*__c4__N512__S200__*.json`; NF4 arm for Qwen3-32B)
reproduces every per-layer effective rank in these files to within 1e-6
relative.
