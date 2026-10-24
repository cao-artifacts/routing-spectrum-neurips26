# Route-merging comparison (Appendix I.1 and I.2)

Calibration-level comparison of PatternMerge maps (labelled E4a in the code),
for every merge width R. The appendix prints R in {4, 8}; these files carry all
R in {1, 2, 4, 8, 16} with R < H_q.

## Files

`e4a_merge.json`: output of `src/analysis/analyze_merge.py` on the C4, N=512,
S=400 Gram dumps of the 14 checkpoints (the measurement described in
`../robustness/README.md`). Per model, two variants: `in_sample` (map selected
and scored on all 400 prompts) and `heldout` (selected on prompts 0-199, scored
on prompts 200-399). Per layer and R:

- `oracle`: rank-R tail energy, a lower bound for any merge map;
- `methods`: calibration error fraction tr(C^T G_P C G_M) / tr(G_P G_M) of
  `greedy` and `eigen` (spectrum-guided), `similarity` (average-linkage
  clustering on the cosine of G_P columns, no G_M), `similarity_gram`
  (clustering on the normalized G_P), and the mean and best of the random
  controls;
- `random` (uniform representatives, uniform assignment) and `random_nearest`
  (uniform representatives, nearest-representative assignment): 1000 seeded
  draws each, with mean, min, percentiles and the fraction of draws at least as
  good as `greedy`;
- `reps`: the representative heads chosen by each method.

Per-(model, R) summaries add layer medians and means, wins, losses and ties of
`greedy` and `eigen` against `similarity`, and layer counts against both random
controls. The file also records the property tests run before the analysis.
`path` was reduced to the dump file name; nothing else was changed.

`e4a_merge.md`: the layer-median tables, the win/loss counts of `greedy`
against `similarity` and the counts of `greedy` against the uniform random
control, for every R. It does not list the best nearest-random draw or the layer
counts of `greedy` against `similarity_gram` and against the nearest-random
control, nor those of `eigen` against `similarity`.

`python src/analysis/merge_tables.py` renders the complete per-R tables from
`e4a_merge.json` (both random controls, both similarity baselines, per-model
and pooled layer counts) into `results/analysis/merge_tables_all_R.{csv,md}`.
For R in {1, 2, 16}, which the appendix does not print, `e4a_merge.md` has the
medians and the greedy-versus-similarity counts; the other counts come only from
`merge_tables.py`.

The error here is the Frobenius (Gaussian operator) calibration metric; the
NP-hardness statement for PatternMerge in the paper uses a different metric.

## Hardware and re-runs

The Gram dumps were measured on one NVIDIA H100 80GB per run (see
`../robustness/README.md`); each is a single-run snapshot (one measurement per
model, no repetitions averaged). The analysis itself is pure NumPy on CPU and is
deterministic given the dumps (seed 0). A re-measurement with other GPUs,
library versions or checkpoint revisions produces slightly different Grams, so
re-derived numbers can differ slightly. The dumps are not shipped (about 9 GB).
