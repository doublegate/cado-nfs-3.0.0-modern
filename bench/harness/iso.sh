#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
set -e
cd "$REPO"
N=142100302046804900094961337970933307673197699104573925821488286818433988882065923457630446247
P=./build/$(hostname)/polyselect/polyselect
# Huge P (root-finding over ~150k primes), tiny admax (few ad-values => proots
# dominates the wall). 1 thread to remove team-scheduling noise.
ARGS="-N $N -d 5 -P 2000000 -t 1 -admin 1 -admax 60"
run() {
  local tag=$1; shift
  "$@" /usr/bin/time -v $P $ARGS >/tmp/cado-nfs-bench/iso_$tag.txt 2>/tmp/cado-nfs-bench/iso_$tag.time
  local cpu wct; cpu=$(grep -oE "total phase took [0-9.]+s" /tmp/cado-nfs-bench/iso_$tag.txt | tail -1)
  wct=$(grep -oE "took \(WCT\) [0-9.]+s" /tmp/cado-nfs-bench/iso_$tag.txt | tail -1)
  echo "$tag: $cpu | $wct"
}
run CPU
run GPU env CADO_GPU_POLYSELECT=1
