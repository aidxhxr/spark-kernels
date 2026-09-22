#!/usr/bin/env bash
# Run every bench_* binary in the build dir and collect JSON-lines results into results/.
#   ./scripts/run_all_benches.sh [build_dir] [flags passed to every bench, e.g. --iters=200]
# Each bench validates its kernel against a reference and exits non-zero on mismatch, so
# this script stops at the first failing kernel. The GPU state the rows were measured in
# (clocks, power limit, throttle reasons, display; see gpu_env.sh) is written to
# results/env.txt first, so a results directory always says what produced it.
set -euo pipefail

BUILD_DIR="${1:-build}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

echo "=== GPU state -> $RESULTS/env.txt" >&2
"$ROOT/scripts/gpu_env.sh" | tee "$RESULTS/env.txt" >&2
echo >&2

shopt -s nullglob
benches=("$BUILD_DIR"/bench_*)
if [ ${#benches[@]} -eq 0 ]; then
  echo "no bench_* binaries in $BUILD_DIR — run 'make build' first" >&2
  exit 1
fi

for bin in "${benches[@]}"; do
  [ -x "$bin" ] || continue
  name="$(basename "$bin")"
  kernel="${name#bench_}"
  out="$RESULTS/$kernel.json"
  echo "=== $name -> $out" >&2
  # stdout = JSON lines (captured), stderr = human table (passed through)
  "$bin" "${@:2}" > "$out"
  echo >&2
done

echo "all benchmarks finished; results in $RESULTS/" >&2
echo "next: python3 scripts/make_results_table.py && python3 scripts/roofline.py" >&2
