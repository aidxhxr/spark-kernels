#!/usr/bin/env bash
# Print the state of the GPU that decides whether a benchmark run can be trusted: driver and
# toolkit, clocks against their maxima, P-state, power limit, throttle reasons, and whether a
# display hangs off the card (docs/RTX5090.md, "Measuring on a GeForce card").
#   ./scripts/gpu_env.sh            # run_all.sh tees this into results/env.txt
# Fields a driver does not know print "n/a" instead of failing the run; no UUID or serial is
# queried, so the output is safe to commit next to the results.
set -uo pipefail

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found: no GPU state recorded" >&2
  exit 0
fi

FIELDS=(name driver_version vbios_version compute_cap pstate
        power.limit power.max_limit power.draw temperature.gpu
        clocks.gr clocks.max.gr clocks.mem clocks.max.mem
        clocks_event_reasons.active
        pcie.link.gen.current pcie.link.width.current
        display_active memory.used memory.total)

# First GPU's value for one field; empty if this driver rejects the field.
query() {
  local out
  out="$(nvidia-smi --query-gpu="$1" --format=csv,noheader 2>/dev/null)" || return 0
  echo "${out%%$'\n'*}"
}

echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvcc_release="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
echo "nvcc: ${nvcc_release:-n/a}"
for f in "${FIELDS[@]}"; do
  v="$(query "$f")"
  echo "$f: ${v:-n/a}"
done

# The two conditions that most often spoil a run on the 5090.
if [ "$(query display_active)" = "Enabled" ]; then
  echo "WARNING: a display is attached to this GPU; the compositor competes for time slices and VRAM bandwidth" >&2
fi
reasons="$(query clocks_event_reasons.active)"
case "$reasons" in
  ""|0x0000000000000000|0x0000000000000001) ;;  # none, or idle
  *) echo "WARNING: clocks are being limited (clocks_event_reasons.active=$reasons); see nvidia-smi -q -d PERFORMANCE" >&2 ;;
esac
