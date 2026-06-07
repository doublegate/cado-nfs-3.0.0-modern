#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
set -e
cd "$REPO"
N=142100302046804900094961337970933307673197699104573925821488286818433988882065923457630446247
P=./build/$(hostname)/polyselect/polyselect
echo "N digits: ${#N}"
run() {
  local tag=$1; shift
  /usr/bin/time -v "$@" $P -N $N -d 5 -P 250000 -t 4 -admin 1 -admax 3000 \
     >/tmp/cado-nfs-bench/hd_$tag.txt 2>/tmp/cado-nfs-bench/hd_$tag.time
  local w; w=$(grep -oE "took \(WCT\) [0-9.]+s" /tmp/cado-nfs-bench/hd_$tag.txt | tail -1)
  echo "$tag: $w  polys=$(grep -cE '^Y0:' /tmp/cado-nfs-bench/hd_$tag.txt)"
  grep -iE 'no CUDA|falling back' /tmp/cado-nfs-bench/hd_$tag.time | head -1 || true
}
run CPU
run GPU env CADO_GPU_POLYSELECT=1
grep -E "^(Y0|Y1|c[0-9]):" /tmp/cado-nfs-bench/hd_CPU.txt | sort > /tmp/cado-nfs-bench/hd_cpu_p.txt
grep -E "^(Y0|Y1|c[0-9]):" /tmp/cado-nfs-bench/hd_GPU.txt | sort > /tmp/cado-nfs-bench/hd_gpu_p.txt
if diff -q /tmp/cado-nfs-bench/hd_cpu_p.txt /tmp/cado-nfs-bench/hd_gpu_p.txt >/dev/null; then
  echo "correctness: IDENTICAL poly set ($(wc -l < /tmp/cado-nfs-bench/hd_cpu_p.txt) lines)"
else
  echo "correctness: DIFF"; diff /tmp/cado-nfs-bench/hd_cpu_p.txt /tmp/cado-nfs-bench/hd_gpu_p.txt | head
fi
