# Attention Is More Than You Need: Spectral Redundancy in Multi-Head Routing

Code and data for the paper

> Charles Cao (University of Tennessee, Knoxville).
> *Attention Is More Than You Need: Spectral Redundancy in Multi-Head Routing.*
> Advances in Neural Information Processing Systems 39 (NeurIPS 2026), poster.

This repository is an archived research artifact for the paper above. It is not
actively maintained.

## Summary

Recent direct-sum hardness results show that evaluating every attention head
separately is essentially optimal. The paper proves that the key-value sharing
of MQA and GQA does not remove this barrier in the worst case: under standard
fine-grained complexity assumptions, even one-layer MQA with a single shared
key-value head needs one full quadratic routing computation per query head. To
see how far trained models are from that worst case, it measures a
calibration-stacked, signed routing-message spectrum on 14 open-weight GQA
checkpoints (Qwen3, Llama 3, Gemma 2, Mistral; 0.6B to 70B parameters). This
spectrum diagnoses a signed calibration relaxation: on C4 calibration prompts at
N = 512, eight directions capture more than 90% of the signed calibration
energy in every model, and across a 126-setting grid of corpora and sequence
lengths they capture at least 87.9% (more than 90% in 121 settings). The
robustness studies show that the fitted message subspace is stable on held-out
prompts and under bootstrap resampling and transfers across corpora at a small
cost, and that the layer-median statistics barely move between fp16 and NF4
weights. In a route-merging study within the PatternMerge family, the joint
routing-message criterion serves as a ranking signal for choosing
representative routes. The spectrum shows where route sharing is worth testing;
it is not a compressor, it does not establish a softmax-realizable compressed
attention module, and no speedup is measured or claimed. The theoretical
results have no code; this repository covers the measurements.

## Contents

| Path | Contents |
|---|---|
| `src/measure/experiment.py` | Measurement pipeline: per-layer route Gram G_P (eager attention maps over the calibration prompts) and message Gram G_M (value-output maps), the H_q x H_q Gram eigenproblem, effective rank r_eff and tail energies T_R; explicit precision flag `--quant`, optional per-prompt Gram dumps, NaN/Inf sentinels, checkpoint-revision capture |
| `src/measure/run_table1.sh` | Table 1 run with explicit per-model precision |
| `src/measure/run_batch.py` | Resumable driver for the Appendix H measurement matrix |
| `src/measure/verify_dump.py` | Integrity check of a Gram dump against its result JSON |
| `src/measure/selftest.sh` | Pre-flight check of the measurement and patching code on tiny models (CPU; downloads the tiny models and WikiText samples) |
| `src/analysis/spectral_common.py` | Shared NumPy linear algebra and the Gram-dump format |
| `src/analysis/analyze_heldout.py` | H.1 held-out split and bootstrap, H.3 cross-domain transfer |
| `src/analysis/summarize_grids.py` | H.2 domain x length grid and H.4 precision pairs, from per-setting JSONs |
| `src/analysis/analyze_merge.py` | I.1 merge-map constructors and baselines, I.2 calibration-level comparison |
| `src/analysis/merge_tables.py` | I.2 tables for every merge width R from the merge JSON |
| `src/analysis/export_phi.py` | Exports the merge maps applied in I.3 and I.4 |
| `src/analysis/test_analyzers.py`, `make_synthetic_dump.py` | Property tests of the closed forms; synthetic Gram dumps for testing |
| `src/patching/route_patch_ppl.py` | I.3 and I.4 route-patching perplexity harness with a 30-check self-test |
| `src/patching/PHI_SCHEMA.md` | File format of the merge maps |
| `src/figures/plot_figures.py` | Table 1 (LaTeX) and the spectrum figures |
| `data/table1/` | Table 1 measurements (14 models) |
| `data/robustness/` | Appendix H: 151 per-setting spectra and the held-out, bootstrap and transfer results |
| `data/merge/` | Appendix I.2: merge comparison for every R |
| `data/patching/` | Appendix I.3 and I.4: merge maps, 26 perplexity runs, gate-suite run |
| `requirements.txt`, `constraints.txt` | Pinned dependencies (see "Setup and tested versions") |
| `LICENSE`, `LICENSES/Apache-2.0.txt` | MIT License; Apache License 2.0 for the code copied from Hugging Face Transformers (see "License") |

Every folder under `data/` has a README with the hardware, the file contents and
any redactions. Run every command from the repository root. The scripts write to
`results/` (git-ignored), with one exception: `src/measure/experiment.py` run
without `--output` writes `./results.json` (also git-ignored); all commands
below pass `--output`. Model weights and datasets are downloaded to the Hugging
Face cache (`~/.cache/huggingface` unless `HF_HOME` or `HF_HUB_CACHE` points
elsewhere).

## Setup and tested versions

Python 3.11 (as on the GPU hosts) or 3.12; the CPU checks below pass with both.
Python 3.13 and newer do not work: numpy 1.26.3 and torch 2.4.1 have no wheels
for them.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
export HF_TOKEN=...   # read access to the gated checkpoints (see "Models and data")
```

`requirements.txt` pins the direct dependencies, including pyarrow (pyarrow
26.0.0 refuses to import with NumPy 1.26) and sentencepiece and protobuf (the
Mistral-7B-v0.3 tokenizer needs both). `constraints.txt` pins every other
package to the environment in which the CPU checks below were run. From PyPI,
`torch==2.4.1` is the CUDA 12.1 build; the measurements used the CUDA 12.4 build
(`pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124`
before the `pip install -r` line). On a machine without a GPU, install the CPU
build first (`pip install torch==2.4.1 --index-url
https://download.pytorch.org/whl/cpu`); it saves about 2.6 GB of downloads (the
CUDA build of torch and its CUDA libraries). The four checks that only read the shipped data (`test_analyzers.py`,
`plot_figures.py`, `summarize_grids.py`, `merge_tables.py`) need only
`pip install numpy==1.26.3 matplotlib==3.10.8`.

Tested versions:

- GPU measurements of Appendix H (recorded in every result file): Python
  3.11.10, PyTorch 2.4.1 (CUDA 12.4 build), Transformers 4.57.6, Datasets 5.0.0,
  Accelerate 1.14.0, bitsandbytes 0.45.5, huggingface_hub 0.36.2, NumPy 1.26.3.
  Route patching of Appendix I (recorded in its result files): PyTorch 2.4.1
  (CUDA 12.4 build) and Transformers 4.57.6. The versions of pyarrow,
  sentencepiece and protobuf on the GPU hosts were not recorded.
- Table 1 measurements: PyTorch 2.4 and Transformers 4.57 (patch versions not
  recorded).
- CPU checks of this repository (every command under "Quick checks on CPU", in
  fresh environments installed from `requirements.txt` after the CPU build of
  PyTorch): Python 3.11.15 and 3.12.3, PyTorch 2.4.1+cpu, Transformers 4.57.6,
  Datasets 5.0.0, pyarrow 25.0.1, sentencepiece 0.2.2, protobuf 7.36.2, NumPy
  1.26.3, Matplotlib 3.10.8; the complete list is `constraints.txt`.

The route-patching harness relies on Transformers 4.57 attention internals
(`ALL_ATTENTION_FUNCTIONS`, `ALL_MASK_ATTENTION_FUNCTIONS`); other major
versions are untested.

## Hardware

- Table 1 and all Appendix H measurements: one NVIDIA H100 80GB per run, except
  the Llama-3-70B matched pair of H.4, which ran on a host with two H100 80GB
  GPUs (the fp16 arm needs both; NF4 alone fits on one GPU, as in Table 1, H.1
  and H.2).
- Route patching (I.4): one H100 80GB, fp16.
- Disk: the 14 checkpoints take about 344 GB in fp16 in the Hugging Face cache;
  the Gram dumps of the full Appendix H matrix take about 9 GB.
  `run_batch.py --purge_hf_cache` deletes each checkpoint after its last run. It
  removes the checkpoint's whole folder (`models--<org>--<name>`) from the cache,
  including files that were there before the run and other revisions, so use it
  only with a cache of its own (`HF_HOME` or `HF_HUB_CACHE`).
- Everything that reads the shipped data runs on a CPU, in seconds.

## Measurement protocol

What `src/measure/experiment.py` computes for each model and calibration pool:

- Prompts: the first documents in stream order whose raw text has more than 2N
  characters (for C4, at most 3S documents are scanned), tokenized with the
  model's own tokenizer and truncated to at most N tokens, without padding;
  `--prompt_cache` gives every model byte-identical prompts.
- Attention maps: the model is loaded with `attn_implementation="eager"` and
  run on one prompt at a time with `output_attentions=True`; a forward hook on
  every self-attention module captures the post-softmax maps (causal mask
  included). Maps returned at key-value-head granularity are repeated to
  query-head granularity.
- Route Gram: G_P[h, h'] = sum over prompts s of <P_h(X^s), P_h'(X^s)>_F,
  accumulated in float64; `--dump_grams` also saves every per-prompt term.
- Message Gram: G_M[h, h'] = <M_h, M_h'>_F with M_h = W^V_kappa(h) W^O_h,
  computed from head-dimension factors (the hidden x hidden maps are never
  formed) with the weights cast to float32, after dequantization for NF4.
- Spectrum: the squared singular values of the calibration-stacked operator are
  the eigenvalues of G_P^(1/2) G_M G_P^(1/2), an H_q x H_q problem. Per layer,
  r_eff = (sum_k sigma_k)^2 / sum_k sigma_k^2 and
  T_R = sum_{k>R} sigma_k^2 / sum_k sigma_k^2 (0 when R >= H_q); tables report
  medians over layers.
- Precision: `--quant fp16|bf16|nf4|fp32` sets the load precision explicitly
  (NF4 is bitsandbytes 4-bit with fp16 compute). `--quant auto` keeps the
  original size heuristic, which among the 14 models selects NF4 only for
  Llama-3-70B.
- Integrity: non-finite values in attention maps, weights and Gram matrices are
  counted before any cleaning, and a layer with non-finite values or zero energy
  is flagged as corrupted.

## Quick checks on CPU

No GPU, no gated weights:

```bash
python src/analysis/test_analyzers.py                  # 16 property tests of the closed forms
python src/figures/plot_figures.py                     # Table 1 and figures from data/table1/
python src/analysis/summarize_grids.py                 # H.2 and H.4 tables from data/robustness/cells/
python src/analysis/merge_tables.py                    # I.2 tables for every R from data/merge/
python src/patching/route_patch_ppl.py --self_test --test_wikitext   # route-patching gates (downloads two tiny models and WikiText-2)
bash src/measure/selftest.sh                           # tiny-model measurement, dump check, gates (needs network)
```

The gate suite passes only if its summary line reads 0 fail and 0 skip (29 pass
and 1 n/a: one tiny model has no grouped key-value heads). It also exits with
code 0 when gates were skipped, for example without network access, so read the
summary line. `selftest.sh` checks this itself and exits with 1 if any step
fails or any gate is skipped.

## Reproducing the paper

The analyses of H.1, H.3, I.2 and the merge-map export read per-prompt Gram
dumps, which are not shipped (about 9 GB). Their outputs are in `data/`; to
re-derive them, first run the GPU measurements below. Table 1, the figures, H.2,
H.4 and the all-R merge tables regenerate on a CPU from the shipped data.

The LaTeX tables of Appendices H and I were typeset from these outputs by a
script of the paper sources, which is not part of this repository: H.1 and H.3
from `e1_heldout.json`, H.2 and H.4 from `e2_grid.json` and `e3_quant.json`
(`summarize_grids.py`), I.2 from `e4a_merge.json` (`merge_tables.py` prints the
same numbers for every R), and I.4 from the 26 perplexity files, whose `ppl`
values `data/patching/e4b_ppl.md` lists. The commands below regenerate the
numbers, not the LaTeX layout.

### Table 1 and Section 6 (Figures 1 and 2, Appendix D figure)

From the shipped measurements:

```bash
python src/figures/plot_figures.py
# results/figures/results_table.tex: Table 1 values
# Figure 1: fig2_bar_chart, fig1_reff_per_layer, fig3_scale_scatter, fig6_family_scaling
# Figure 2: fig5_sv_grid, fig4_heatmap; Appendix D: fig7_reff_range
```

The published figures were rendered with the DejaVu Serif fallback font; where
Times New Roman is installed, `plot_figures.py` uses it, so glyphs and text
widths differ from the paper while the data do not. `fig6_family_scaling` (the
bottom-right panel of Figure 1) was drawn for the paper with plotting
code that was not preserved; the function here reconstructs that panel and
draws the bars from the unrounded layer medians;
the published panel used the one-decimal values of Table 1, so bar heights
differ by less than 0.05.

Re-measurement (GPU; C4 validation, S = 200 prompts, N = 512, precision fixed
per model: NF4 for Qwen3-32B and Llama-3-70B, fp16 for the other twelve):

```bash
bash src/measure/run_table1.sh            # or a subset: bash src/measure/run_table1.sh qwen3-0.6b
python src/figures/plot_figures.py --results_dir results/measure/table1 --outdir results/figures_remeasured
```

Do not use `--quant auto` for Table 1: its size heuristic loads Qwen3-32B in
fp16.

### Appendix H: robustness of the spectral diagnostic

GPU measurements (one process per setting, resumable; JSONs in
`results/measure/json/`, Gram dumps in `results/measure/npz/`):

```bash
python src/measure/run_batch.py --only e1      # H.1: 14 models, C4, N=512, S=400 (prompts 0-199 = Table 1 set)
python src/measure/run_batch.py --only e2      # H.2: 14 models x {c4, wikitext, code} x N in {256, 512, 1024}, S=200
# Qwen3-8B on code overflows in fp16 (the NaN sentinels flag it); re-measure those three cells in bf16:
for N in 256 512 1024; do
  python src/measure/experiment.py --models qwen3-8b --dataset code --seq_len $N --num_prompts 200 \
    --quant bf16 --pool_scan_limit 400000 --prompt_cache results/measure/prompts \
    --dump_grams results/measure/npz --output results/measure/json/qwen3-8b__code__N${N}__S200__bf16.json
done
python src/measure/run_batch.py --only e3      # H.4: Qwen3-14B, Mistral-Nemo-12B, Qwen3-32B in fp16 and NF4
python src/measure/run_batch.py --only e3_70b  # H.4: Llama-3-70B in fp16 (two H100) and NF4
python src/measure/run_batch.py --only all --dry_run   # prints the full matrix (148 runs)
```

`--quant auto` (used by `run_batch.py` for H.1 and H.2) resolves to NF4 only
for Llama-3-70B, so Qwen3-32B is measured in fp16 in H.1 to H.3 and in the
merge study, as in the shipped data.

**H.1 Held-out prompts and prompt resampling** and **H.3 Cross-domain transfer**
(CPU, several minutes; defaults: fit on prompts 0-199, evaluate on 200-399,
R in {1, 4, 8}, 500 bootstrap resamples of 200 prompts, seed 0). The transfer
to code uses the bf16 dumps of Qwen3-8B, so move the overflowed fp16 dumps out
of the way first:

```bash
mkdir -p results/measure/npz_overflowed
mv results/measure/npz/qwen3-8b__code__N*__S200__fp16.npz results/measure/npz_overflowed/
python src/analysis/analyze_heldout.py \
  --npz results/measure/npz/*__c4__N512__S400__*.npz \
  --cross results/measure/npz/*__code__N512__S200__*.npz results/measure/npz/*__wikitext__N512__S200__*.npz
# -> results/analysis/e1_heldout.{json,md}; shipped: data/robustness/e1_heldout.{json,md}
```

The rows come out in the order in which the shell expands the `*` patterns,
with the code dumps listed before the WikiText dumps, as above. The shipped
files follow the expansion order of an en_US.UTF-8 locale; in a C or POSIX
locale (for example `LC_ALL=C`), Llama-3-70B and Llama-3-8B come before the
Llama-3.2 models and Qwen3-1.7B before Qwen3-14B, with the same numbers.
Re-derived from the original dumps in an en_US.UTF-8 locale, the Markdown file
is identical to the shipped one. The JSON differs in its `path`
fields (full paths instead of file names) and timings; with NumPy 2.4.2 every
number is identical, and with NumPy 1.26.3 (as pinned here) numbers differ by
floating-point rounding (relative differences up to about 1e-9).

**H.2 Domain and sequence-length grid** and **H.4 Precision: fp16 versus NF4**:

```bash
python src/analysis/summarize_grids.py                                 # from the shipped cells
python src/analysis/summarize_grids.py --cells results/measure/json    # from a re-measurement
# -> results/analysis/e2_grid.{json,md} (H.2) and e3_quant.{json,md} (H.4)
```

### Appendix I: route-merging study

**I.1 Methods and metrics.** The merge maps (greedy and eigenbasis-loading
spectrum-guided selection, routing-only similarity, Gram similarity, and two
random controls) and the closed-form calibration error are defined in
`src/analysis/analyze_merge.py` and `src/analysis/spectral_common.py`;
`python src/analysis/analyze_merge.py --self-test` checks the closed forms
against explicit matrices.

**I.2 Calibration-level merge quality** (CPU, about 10 minutes for the 14
models; R in {1, 2, 4, 8, 16} with R < H_q, 1000 random draws per control,
seed 0; in-sample and held-out variants):

```bash
python src/analysis/analyze_merge.py --npz results/measure/npz/*__c4__N512__S400__*.npz
# -> results/analysis/e4a_merge.{json,md}; shipped: data/merge/e4a_merge.{json,md}
python src/analysis/merge_tables.py                                         # all-R tables from the shipped JSON
python src/analysis/merge_tables.py --merge_json results/analysis/e4a_merge.json
```

**I.3 Route patching protocol.** Each query head reads the post-softmax
attention rows of its representative head in every layer and keeps its own
value and output path. Gate suite (run it before any perplexity run; it has
passed only if the summary line reads 0 fail and 0 skip):

```bash
python src/patching/route_patch_ppl.py --self_test --test_wikitext --self_test_json results/patching/gates.json
```

Merge maps, selected on prompts 0-199 of the C4 dumps (the shipped maps are in
`data/patching/phis/`, so this step can be skipped):

```bash
python src/analysis/export_phi.py --npz results/measure/npz/qwen3-8b__c4__N512__S400__fp16.npz \
  --methods greedy eigen similarity random --ranks 4 8 16 --sel_range 0:200 --seed 0 \
  --model Qwen/Qwen3-8B --out_dir results/patching/phis
python src/analysis/export_phi.py --npz results/measure/npz/llama3-8b__c4__N512__S400__fp16.npz \
  --methods greedy eigen similarity random --ranks 4 8 16 --sel_range 0:200 --seed 0 \
  --model meta-llama/Meta-Llama-3-8B --out_dir results/patching/phis
```

Gemma 2 is excluded from route patching: its attention-logit softcapping
differs from the patched Llama-style attention, and the harness refuses such
architectures.

**I.4 Downstream perplexity** (GPU; WikiText-2 raw test, window = stride = 512,
fp16):

```bash
PHIS=data/patching/phis   # or results/patching/phis after re-exporting
for pair in qwen3-8b:Qwen3-8B llama3-8b:Meta-Llama-3-8B; do
  key=${pair%%:*}; ckpt=${pair##*:}
  python src/patching/route_patch_ppl.py --model $key --method identity \
    --output results/patching/ppl_${key}_identity.json
  for method in greedy eigen similarity random; do
    for R in 4 8 16; do
      python src/patching/route_patch_ppl.py --model $key --strict_phi \
        --phi_json $PHIS/phi_${ckpt}_${method}_R${R}.json \
        --output results/patching/ppl_${key}_${method}_R${R}.json
    done
  done
done
# shipped: data/patching/ppl/*.json, summary in data/patching/e4b_ppl.md
```

## Models and data

| Key | Hugging Face checkpoint | H_q / H_kv / layers | Table 1 precision | Revision (recorded in the Appendix H runs) |
|---|---|---|---|---|
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | 16 / 8 / 28 | fp16 | `c1899de289a04d12100db370d81485cdf75e47ca` |
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | 16 / 8 / 28 | fp16 | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| `qwen3-4b` | `Qwen/Qwen3-4B` | 32 / 8 / 36 | fp16 | `1cfa9a7208912126459214e8b04321603b3df60c` |
| `qwen3-8b` | `Qwen/Qwen3-8B` | 32 / 8 / 36 | fp16 | `b968826d9c46dd6066d109eabc6255188de91218` |
| `qwen3-14b` | `Qwen/Qwen3-14B` | 40 / 8 / 40 | fp16 | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| `qwen3-32b` | `Qwen/Qwen3-32B` | 64 / 8 / 64 | NF4 | `9216db5781bf21249d130ec9da846c4624c16137` |
| `llama3.2-1b` | `meta-llama/Llama-3.2-1B` | 32 / 8 / 16 | fp16 | `4e20de362430cd3b72f300e6b0f18e50e7166e08` |
| `llama3.2-3b` | `meta-llama/Llama-3.2-3B` | 24 / 8 / 28 | fp16 | `13afe5124825b4f3751f836b40dafda64c1ed062` |
| `llama3-8b` | `meta-llama/Meta-Llama-3-8B` | 32 / 8 / 32 | fp16 | `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920` |
| `llama3-70b` | `meta-llama/Meta-Llama-3-70B` | 64 / 8 / 80 | NF4 | `c82494877ce7f6d7d317c56ec081328e382c72fe` |
| `gemma2-2b` | `google/gemma-2-2b` | 8 / 4 / 26 | fp16 | `c5ebcd40d208330abc697524c919956e692655cf` |
| `gemma2-9b` | `google/gemma-2-9b` | 16 / 8 / 42 | fp16 | `33c193028431c2fde6c6e51f29e6f17b60cbfac6` |
| `mistral-7b-v03` | `mistralai/Mistral-7B-v0.3` | 32 / 8 / 32 | fp16 | `caa1feb0e54d415e2df31207e5f4e273e33509b1` |
| `mistral-nemo-12b` | `mistralai/Mistral-Nemo-Base-2407` | 32 / 8 / 40 | fp16 | `a4477a2f977929a969745b69bbd62e03043551a5` |

Model weights are not included; they are downloaded from the Hugging Face Hub
at run time and remain under their own licenses and terms of use (as stated on
the model cards):

| Checkpoints | License | Access |
|---|---|---|
| the six Qwen3 models, Mistral-7B-v0.3, Mistral-Nemo-Base-2407 | Apache-2.0 | open |
| Meta-Llama-3-8B, Meta-Llama-3-70B | Meta Llama 3 Community License | gated |
| Llama-3.2-1B, Llama-3.2-3B | Llama 3.2 Community License | gated |
| gemma-2-2b, gemma-2-9b | Gemma Terms of Use | gated |

The Table 1 files also contain two pilot runs that the paper does not report
(see `data/table1/README.md`): `google/gemma-2-27b` (Gemma Terms of Use,
gated) and `mistralai/Mistral-7B-v0.1` (Apache-2.0, open).

For the gated checkpoints, accept the license on each model page and export
`HF_TOKEN` for an account with access. The scripts read the token from the
environment; never put it on a command line or in a file.

Revisions: the scripts load the default branch of each checkpoint and have no
revision option. The revisions in the table are recorded in all 151 Appendix H
files (`revision`); the Gram dumps behind H.1, H.3, I.2 and I.3 come from those
runs. The Table 1 files and the I.4 perplexity files record no revision. The
I.4 runs loaded the default branch less than two hours after the last Appendix H
measurement, and in October 2026 the default branch of every checkpoint above
still pointed to the listed commit. To pin a revision, download it and pass the
snapshot directory in place of the model key:

```bash
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218   # prints the snapshot directory
python src/measure/experiment.py --model_id <snapshot directory> --dataset c4 --num_prompts 200 --seq_len 512 \
  --quant fp16 --output results/measure/pinned/qwen3-8b.json
python src/patching/route_patch_ppl.py --model <snapshot directory> --method identity \
  --output results/patching/pinned/ppl_qwen3-8b_identity.json
```

`experiment.py` then records the commit taken from the snapshot path, and keys
the result by the sanitized directory path instead of the model key.
`run_batch.py` and `run_table1.sh` always use the default branch.

Datasets are also downloaded (streamed) at run time and are governed by their
own licenses; no corpus text is included in this repository:

| Dataset | Hugging Face id, configuration, split | License (dataset card) | Used in |
|---|---|---|---|
| C4 | `allenai/c4`, `en`, validation | ODC-BY, with the Common Crawl terms of use | Table 1, Appendices H and I |
| WikiText-103 | `Salesforce/wikitext`, `wikitext-103-raw-v1` (test, then validation, then train) | CC BY-SA 3.0 and GFDL | H.2, H.3 |
| CodeParrot | `codeparrot/codeparrot-clean-valid` | no dataset-level license; each file records its repository's license | H.2, H.3 |
| WikiText-2 | `Salesforce/wikitext`, `wikitext-2-raw-v1`, test | CC BY-SA 3.0 and GFDL | I.4 |

`route_patch_ppl.py` requests WikiText-2 under the legacy id `wikitext`, which
the Hub resolves to `Salesforce/wikitext`, and falls back to
`Salesforce/wikitext` if that fails.

## Scope and limitations

- The Gram dumps (about 9 GB; they also contain short corpus excerpts) and the
  cached prompt pools (corpus text) are not included, so H.1, H.3, I.2 and the
  merge-map export can be re-derived only after re-measuring on GPUs; the
  shipped files are their outputs.
- Precision differs between Table 1 and Appendix H: Table 1 uses NF4 for
  Qwen3-32B, the Appendix H runs use fp16 for it (`--quant auto`); the H.4 pair
  bounds the difference. The three Qwen3-8B code cells were measured in bf16
  after an fp16 overflow, so for them the domain effect and the dtype change
  cannot be separated.
- Calibration pools: C4 at N = 1024 yields 173 prompts instead of 200, and
  WikiText-103 at N = 1024 draws 189 of its 200 prompts from the train split
  (see `data/robustness/README.md`).
- The downstream route-patching check (I.3, I.4) covers two 8B models and
  WikiText-2 perplexity only; the calibration-level comparison (I.2) covers all
  14 models. The gate suite ran on reference models, not on the two 8B
  checkpoints. At R = 4 every method breaks down (perplexities in the hundreds to
  tens of thousands), and even at R = 16 the patched perplexity stays well above
  the unpatched one (Qwen3-8B: 20.7 versus 12.87 for the greedy maps). The
  criterion ranks representative routes; it is not a compression method, and no
  runtime is measured.
- Not included: fitted softmax routes, runtime or prefill benchmarks,
  weighted-covariance spectra, and downstream tasks other than WikiText-2
  perplexity.
- Every shipped measurement file is a single-run snapshot (no repetitions were
  averaged). Re-runs can differ slightly with other GPUs, library versions, or
  dataset and checkpoint revisions. The Table 1 files do not record checkpoint
  revisions; the later Appendix H runs at the same setting and precision
  reproduce every per-layer effective rank of Table 1 to within 1e-6 relative.

## Provenance of the code

- `src/measure/experiment.py` is byte-identical to the script that wrote every
  file in `data/robustness/cells/`: its SHA-256
  (`f3e030a7188e4f1b9c4286d47595ca74eed905259b2502074f08b527b9fd5aaa`) is the
  `script_sha256` recorded there. It extends the Table 1 measurement script
  with opt-in options (corpus choice, explicit precision, Gram dumps, NaN/Inf
  sentinels, revision capture) and leaves the measurement math unchanged. Its
  header comments are therefore kept as they were: they name files of the
  original repository (`experiments/scripts/experiment.py`, `plot_results_v2`),
  and their statement that the defaults reproduce the published protocol does
  not hold for Qwen3-32B. Table 1 reports Qwen3-32B in NF4 (its per-layer
  effective ranks match the NF4 arm of the H.4 pair to within 1e-8 relative and
  differ from the fp16 arm by up to 5.5%), whereas the default `--quant auto`,
  like the size rule of the submitted Table 1 script, selects fp16 for it;
  `run_table1.sh` passes `--quant nf4` explicitly.
- The other scripts are the versions used for the paper, with edits that change
  no computed number: default input and output locations moved to `data/` and
  `results/`; script paths in `selftest.sh` adapted to the `src/` layout;
  command-line options for input and output directories (`summarize_grids.py`,
  `plot_figures.py`); creation of missing output directories
  (`route_patch_ppl.py`); stricter checks (`selftest.sh` writes each run to a
  new folder and fails on any failed step or skipped gate, the WikiText-2 gate
  of `route_patch_ppl.py` counts an import error as a failure, `export_phi.py
  --check_json` compares only with the analysis selected on the same prompts,
  and `summarize_grids.py` reports missing precision pairs instead of stopping);
  and updated usage text and comments.
- `fig6_family_scaling` in `src/figures/plot_figures.py` was rewritten for this
  repository: the script that drew the published panel was not kept, and the
  submitted plotting script drew a line chart instead. The function now draws
  the published layout (bars grouped by family with value labels and the
  H_kv = 8 line); with the DejaVu Serif font its output matches the published
  panel up to the rounding noted above.
- New helpers: `src/measure/run_table1.sh` (Table 1 with explicit precision)
  and `src/analysis/merge_tables.py` (all-R tables).
- Shipped data are unchanged except that prompt text previews were removed and
  absolute paths of the measurement machines were reduced to file names.

## Citation

```bibtex
@inproceedings{cao2026attention,
  title     = {Attention Is More Than You Need: Spectral Redundancy in Multi-Head Routing},
  author    = {Charles Cao},
  booktitle = {Advances in Neural Information Processing Systems 39 (NeurIPS 2026)},
  year      = {2026}
}
```

## Acknowledgments

This material is based upon work supported in part by the U.S. National Science
Foundation under Award Nos. 2613350 and 2520347, and by the USDA National
Institute of Food and Agriculture under award no. 2023-67021-40613. Any
opinions, findings, and conclusions or recommendations expressed in this
material are those of the author and do not necessarily reflect the views of
the National Science Foundation. Any opinions, findings, conclusions, or
recommendations expressed in this material are those of the author and should
not be construed to represent any official USDA or U.S. Government
determination or policy.

## License

The code and the measurement data in this repository are released under the
MIT License (see `LICENSE`), with one exception: `repeat_kv` and the attention
arithmetic of `route_patch_attention_forward` in
`src/patching/route_patch_ppl.py` (repeated in its negative-control functions)
are copied from Hugging Face Transformers 4.57
(`src/transformers/models/llama/modeling_llama.py`, Copyright 2022 EleutherAI
and the HuggingFace Inc. team) and remain under the Apache License, Version 2.0
(see `LICENSES/Apache-2.0.txt`). Model weights and datasets are not part of this
repository and keep their own licenses.
