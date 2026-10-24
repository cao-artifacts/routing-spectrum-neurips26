# Route-patching runs (Appendix I.3 and I.4)

Route-patched WikiText-2 perplexity of Qwen3-8B and Meta-Llama-3-8B (labelled
E4b in the code). In a patched model every query head reads the post-softmax
attention rows of its representative head, in every layer, while keeping its
own value and output path.

## Files

`phis/phi_<checkpoint>_<method>_R<R>.json`: the 24 per-layer merge maps that
were applied (2 checkpoints x methods `greedy`, `eigen`, `similarity`,
`random` x R in {4, 8, 16}). They were selected on prompts 0-199 of the C4,
N=512, S=400 Gram dumps by `src/analysis/export_phi.py` with
`--sel_range 0:200 --seed 0`; the format is described in
`src/patching/PHI_SCHEMA.md`. The representative sets of the `greedy`, `eigen`
and `similarity` maps equal the held-out representatives recorded in
`../merge/e4a_merge.json` (612 of 612 layer maps), and `export_phi.py`
regenerates the `random` maps exactly.

`ppl/ppl_<model>_<method>[_R<R>].json`: 26 runs (2 models x (12 maps +
identity)) of `src/patching/route_patch_ppl.py`: WikiText-2 raw test split,
window = stride = 512 tokens (585 windows and 298,493 scored tokens for
Qwen3-8B, 565 and 288,512 for Llama-3-8B), fp16, eager attention. `identity`
applies the identity map and is the unpatched baseline. Each file records the
map statistics, perplexity, total NLL, token and window counts, runtime and
library versions. Every file carries `"gates_verified": false` by design: the
harness cannot prove inside a perplexity run that the gate suite was run.
`phi_file` was reduced to the map's file name; nothing else was changed.

`gates_gpu.json`: the correctness suite
(`route_patch_ppl.py --self_test --test_wikitext`) run on the same GPU machine
and software before the perplexity runs: 30 checks, 29 pass, 1 not applicable
(GQA indexing on an MHA model), 0 fail, 0 skipped. Like every run of the suite,
it ran on that host's CPU in float32, so it does not exercise the fp16 CUDA code
path of the perplexity runs (the file name only records which host it ran on).
The checks run on a synthetic GQA Llama and on tiny random-weight Llama and
Qwen3 models: bit-exact identity-map NLL, agreement with an independent
reference implementation, causal and row-stochastic patched maps, GQA head
indexing, equality of the two patch mechanisms, window accounting, and five
deliberately broken implementations that must each be caught. The suite was not
re-run on the two 8B checkpoints.

`e4b_ppl.md`: summary table of the 26 runs.

## Hardware and re-runs

One NVIDIA H100 80GB; PyTorch 2.4.1+cu124 and Transformers 4.57.6 (recorded in
every file). Each perplexity file is a single-run snapshot (one run per model
and map, no repetitions averaged). Re-runs can differ slightly with other GPUs
(fp16 numerics), library versions, or checkpoint and dataset revisions; the
very large perplexities at R = 4 are the most sensitive. The files record no
checkpoint revision (see "Models and data" in the top-level README).
