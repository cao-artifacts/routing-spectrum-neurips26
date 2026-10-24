#!/usr/bin/env bash
# Table 1 / Section 6: routing-message spectra of the 14 GQA checkpoints on C4
# validation (S=200 prompts, N=512 tokens), one model per process.
#
# Precision is passed explicitly per model and matches the Table 1 run:
# Qwen3-32B and Llama-3-70B in 4-bit NF4, the other twelve models in fp16.
# Do not use `--quant auto` for Table 1: its size heuristic loads Qwen3-32B in fp16.
#
#   bash src/measure/run_table1.sh                        # all 14 models
#   bash src/measure/run_table1.sh qwen3-0.6b gemma2-2b   # a subset
#   bash src/measure/run_table1.sh --help
#
# Environment: PYTHON (default: python), OUT (default: <repo>/results/measure/table1).
# Gated checkpoints (meta-llama/*, google/gemma-2-*) need HF_TOKEN in the environment.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python}"
OUT="${OUT:-$ROOT/results/measure/table1}"
ALL_MODELS=(qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b qwen3-14b qwen3-32b
            llama3.2-1b llama3.2-3b llama3-8b llama3-70b
            gemma2-2b gemma2-9b mistral-7b-v03 mistral-nemo-12b)

usage() {
  echo "usage: bash src/measure/run_table1.sh [model_key ...]"
  echo "Re-measures Table 1 (C4 validation, S=200, N=512) with the Table 1 precision per model."
  echo "model keys (default: all 14): ${ALL_MODELS[*]}"
  echo "environment: PYTHON (default: python), OUT (default: results/measure/table1), HF_TOKEN"
}

precision_of() {
  case "$1" in
    qwen3-32b|llama3-70b) echo nf4 ;;
    qwen3-0.6b|qwen3-1.7b|qwen3-4b|qwen3-8b|qwen3-14b) echo fp16 ;;
    llama3.2-1b|llama3.2-3b|llama3-8b) echo fp16 ;;
    gemma2-2b|gemma2-9b|mistral-7b-v03|mistral-nemo-12b) echo fp16 ;;
    *) return 1 ;;
  esac
}

if [ "$#" -gt 0 ]; then
  case "$1" in
    -h|--help) usage; exit 0 ;;
  esac
  MODELS=("$@")
else
  MODELS=("${ALL_MODELS[@]}")
fi

# Check every key before anything is written.
for m in "${MODELS[@]}"; do
  if ! precision_of "$m" > /dev/null; then
    echo "unknown model key: $m" >&2
    usage >&2
    exit 1
  fi
done

mkdir -p "$OUT"
for m in "${MODELS[@]}"; do
  q="$(precision_of "$m")"
  echo "=== $m (--quant $q)"
  "$PY" "$HERE/experiment.py" --models "$m" --dataset c4 --num_prompts 200 --seq_len 512 \
      --quant "$q" --prompt_cache "$ROOT/results/measure/prompts" \
      --output "$OUT/results_$m.json"
done
echo "Table 1 JSONs written to $OUT"
echo "Render Table 1 and the figures with: python src/figures/plot_figures.py --results_dir $OUT"
