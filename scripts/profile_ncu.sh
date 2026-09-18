#!/usr/bin/env bash
# Profile the headline kernels with Nsight Compute and dump the "details" page as text.
#   ./scripts/profile_ncu.sh [build_dir]
#
# Permissions: on DGX OS, ncu needs access to GPU performance counters. Either run with
# sudo, or allow all users once:
#   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/ncu.conf
#   sudo update-initramfs -u && sudo reboot
set -euo pipefail

BUILD_DIR="${1:-build}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

NCU="${NCU:-ncu}"
command -v "$NCU" >/dev/null || { echo "ncu not found; it ships with the CUDA 13 toolkit (/usr/local/cuda/bin)" >&2; exit 1; }

# name | binary + args | kernel-name regex (our kernels all live in namespace spark)
profile() {
  local name="$1"; shift
  local regex="$1"; shift
  echo "=== ncu: $name" >&2
  "$NCU" --set full \
         --kernel-name "regex:$regex" \
         --launch-skip 0 --launch-count 1 \
         -f -o "$RESULTS/ncu_$name" \
         "$@" >/dev/null
  "$NCU" --import "$RESULTS/ncu_$name.ncu-rep" --page details > "$RESULTS/ncu_$name.txt"
  echo "    -> $RESULTS/ncu_$name.ncu-rep, $RESULTS/ncu_$name.txt" >&2
}

profile hgemm_v2  'hgemm'   "$BUILD_DIR/bench_hgemm"   --variant=2 --m=4096 --n=4096 --k=4096 --iters=1 --warmup=0
profile hgemm_v0  'hgemm'   "$BUILD_DIR/bench_hgemm"   --variant=0 --m=4096 --n=4096 --k=4096 --iters=1 --warmup=0
profile sgemm_v3  'sgemm'   "$BUILD_DIR/bench_sgemm"   --variant=3 --m=4096 --n=4096 --k=4096 --iters=1 --warmup=0
profile rmsnorm   'rmsnorm' "$BUILD_DIR/bench_rmsnorm" --iters=1 --warmup=0
profile softmax   'softmax' "$BUILD_DIR/bench_softmax" --iters=1 --warmup=0

echo "open the .ncu-rep files in the Nsight Compute GUI, or grep the .txt dumps for" >&2
echo "'Achieved Occupancy', 'DRAM Throughput', 'Shared Memory Bank Conflicts'." >&2
