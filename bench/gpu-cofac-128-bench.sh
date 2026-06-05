#!/usr/bin/env bash
#
# gpu-cofac-128-bench.sh -- validate + benchmark the 128-bit GPU ECM path.
#
# The 64-bit GPU path only covers cofactors < 2^62, so it never engages at the
# stock c120 mfb (52/54). This script forces a heavy-cofactoring, large-cofactor
# regime by running the c120 poly with mfb1=90 / lpb1=31 (a 3-large-prime side):
# side-1 leftover cofactors reach ~90 bits, which (a) exercise the 2-limb
# factor_batch_128 (validated bit-exact in bench/gpu-mont128.cu) and (b) make CPU
# cofactoring the dominant cost -- the regime where GPU offload can pay off.
#
# Reports, CPU-only vs batch (CADO_GPU_ECM=batch):
#   - correctness: relations are a valid SUPERSET (lost must be 0)
#   - the 128-bit path actually fires (split>0 with MINBITS=62)
#   - throughput: rel/wall (off is the bar; in this regime batch can beat it)
#
# USAGE: bench/gpu-cofac-128-bench.sh [las] [random-sample N]
set -eu
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LAS="${1:-$HERE/build-gpu/sieve/las}"; NS="${2:-40}"
POLY="$HERE/tests/sieve/c120.poly"; FB="${FB:-/tmp/cado-bench/c120.fb1}"
MAKEFB="$(dirname "$LAS")/makefb"; OUT="$(mktemp -d /tmp/cado-gpu-128.XXXXXX)"
A=(-poly "$POLY" -fb1 "$FB" -lim0 2500000 -lim1 3400000 -lpb0 27 -lpb1 31
   -mfb0 52 -mfb1 90 -I 12 -q0 600000 -q1 1200000 -sqside 1 -t 1
   -random-sample "$NS" -seed 1)
[ -s "$FB" ] || "$MAKEFB" -poly "$POLY" -lim 3400000 -maxbits 12 -side 1 -t "$(nproc)" -out "$FB" >/dev/null 2>&1

run() { local lbl="$1"; shift; local t0 t1 w rel fac
  t0=$(date +%s.%N)
  env "$@" "$LAS" "${A[@]}" >"$OUT/$lbl.out" 2>"$OUT/$lbl.err" || true
  t1=$(date +%s.%N); w=$(echo "$t1-$t0"|bc)
  rel=$(grep -c '^[^#]' "$OUT/$lbl.out" || true)
  fac=$(grep -m1 'Total cpu time' "$OUT/$lbl.out" | sed -E 's/.*factor ([0-9.]+) .*/\1/' || echo "?")
  grep '^[^#]' "$OUT/$lbl.out" | sort > "$OUT/$lbl.rel"
  printf "%-12s wall=%6.1fs factor=%-7ss rel=%-6s rel/wall=%-5.0f " "$lbl" "$w" "$fac" "$rel" "$(echo "$rel/$w"|bc -l)"
  grep -i 'GPU ECM cofac hook' "$OUT/$lbl.err" | sed 's/# GPU ECM cofac hook: //' || echo "(cpu-only)"
}

echo "### c120 poly, mfb1=90 lpb1=31 (heavy + 128-bit regime), random-sample $NS, -t1"
run off    CADO_GPU_ECM=
run batch  CADO_GPU_ECM=batch CADO_GPU_MINBITS=0
echo "### correctness: batch vs off  +$(comm -13 "$OUT/off.rel" "$OUT/batch.rel"|wc -l) extra, -$(comm -23 "$OUT/off.rel" "$OUT/batch.rel"|wc -l) lost (lost must be 0)"
echo "# artifacts in $OUT"
