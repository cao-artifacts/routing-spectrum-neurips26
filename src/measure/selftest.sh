#!/usr/bin/env bash
# Pre-flight check of the measurement and route-patching code on tiny models.
# CPU only, no GPU and no gated checkpoints; needs network access for the tiny
# test models and the WikiText samples. Run it (also on a GPU host before the
# measurements) from the repository root:
#
#   bash src/measure/selftest.sh [python] [workdir]
#
# Every run writes into a new subfolder of workdir (default: results/selftest/),
# so no file from an earlier run is checked. Exit code 0 only if every step
# passes, including all route-patching gates with 0 fail and 0 skip.
set -u -o pipefail
PY="${1:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${2:-$HERE/../../results/selftest}"
mkdir -p "$WORK"
RUN="$(mktemp -d "$WORK/run.XXXXXX")" || exit 1
mkdir -p "$RUN/npz" "$RUN/json"
echo "selftest outputs: $RUN"
rc=0
failed=""
fail() { rc=1; failed="$failed $1"; }

echo "=== (a) import check ==="
"$PY" -c "
import importlib.util, sys
for name in ['experiment','run_batch','verify_dump']:
    spec = importlib.util.spec_from_file_location(name, '$HERE/' + name + '.py')
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    print('import OK:', name)
import experiment as E
print('MODELS:', len(E.MODELS), 'TABLE1:', len(E.TABLE1_MODELS), 'DATASETS:', list(E.DATASETS))
" || fail a

echo
echo "=== (b) tiny end-to-end run (wikitext, S=3, N=64, dump) ==="
"$PY" "$HERE/experiment.py" \
  --model_id hf-internal-testing/tiny-random-LlamaForCausalLM \
  --dataset wikitext --num_prompts 3 --seq_len 64 \
  --dump_grams "$RUN/npz" --output "$RUN/json/tiny.json" 2>&1 | tail -25 || fail b

echo
echo "=== (b2) npz verification ==="
shopt -s nullglob
dumps=("$RUN"/npz/*.npz)
shopt -u nullglob
if [ "${#dumps[@]}" -eq 0 ]; then
  echo "no Gram dump was written by (b)"
  fail b2
else
  "$PY" "$HERE/verify_dump.py" "${dumps[@]}" --json "$RUN/json/tiny.json" || fail b2
fi

echo
echo "=== (b3) JSON sanity (finite spectrum, no corruption flags) ==="
"$PY" -c "
import json, math, sys
d = json.load(open('$RUN/json/tiny.json'))
r = list(d.values())[0]
vals = [v for L in r['layers'].values() for v in L['sigma'] + L['sigma_squared'] + [L['r_eff']]]
print('layers=%d  values checked=%d  all finite=%s' % (r['num_layers'], len(vals), all(map(math.isfinite, vals))))
print('any_corrupted=%s  total_nan=%d  total_inf=%d  quant_used=%s  revision=%s'
      % (r['any_corrupted'], r['total_nan_count'], r['total_inf_count'], r['quant_used'], r['revision']))
sys.exit(0 if all(map(math.isfinite, vals)) and not r['any_corrupted'] else 1)
" || fail b3

echo
echo "=== (b4) route-patching gate suite (must end with 0 fail and 0 skip) ==="
GATES="$RUN/json/route_patch_gates.json"
"$PY" "$HERE/../patching/route_patch_ppl.py" --self_test --test_wikitext --self_test_json "$GATES" || fail b4
"$PY" -c "
import json, sys
d = json.load(open(sys.argv[1]))
print('gate suite: %d checks, %d fail, %d skip, %d n/a' % (len(d['rows']), d['n_fail'], d['n_skip'], d['n_na']))
sys.exit(0 if d['n_fail'] == 0 and d['n_skip'] == 0 else 1)
" "$GATES" || fail b4-gates

echo
echo "=== (c) batch matrix dry run ==="
"$PY" "$HERE/run_batch.py" --only all --dry_run --out_dir "$RUN/batch_dry_run" | tail -8 || fail c

echo
echo "=== selftest exit code: $rc${failed:+ (failed:$failed)} ==="
exit $rc
