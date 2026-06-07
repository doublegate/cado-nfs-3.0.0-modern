#!/usr/bin/env bash
# Extract per-phase CPU-seconds + headline cpu/elapsed from each sweep log.
for sz in c60 c70 c80 c90; do
  log=/tmp/cado-nfs-bench/$sz.log
  [ -f "$log" ] || continue
  python3 - "$sz" "$log" <<'PY'
import re, sys
sz, log = sys.argv[1], sys.argv[2]
t = open(log, encoding='utf-8', errors='replace').read()
def last(pat):
    m = re.findall(pat, t)
    return float(m[-1]) if m else 0.0
cpu, elapsed = 0.0, 0.0
m = re.search(r'entire Complete Factorization\s+([\d.]+)/([\d.]+)', t)
if m: cpu, elapsed = float(m.group(1)), float(m.group(2))
poly = last(r'Polynomial Selection \(size optimized\): Total time: ([\d.]+)') \
     + last(r'Polynomial Selection \(root optimized\): Total time: ([\d.]+)')
sieve = last(r'Lattice Sieving: Total time: ([\d.]+)s')
filt = (last(r'time for dup1: ([\d.]+)/') + last(r'time for dup2: ([\d.]+)/')
        + last(r'time for purge: ([\d.]+)/') + last(r'time for merge: ([\d.]+)/')
        + last(r'time for replay: ([\d.]+)/'))
la = last(r'time for bwc: ([\d.]+)/')
sqrt = last(r'time for sqrt: ([\d.]+)/')
par = cpu/elapsed if elapsed else 0
print(f"{sz}: cpu={cpu:.1f} elapsed={elapsed:.1f} parallel={par:.1f}x | "
      f"poly={poly:.1f} sieve={sieve:.1f} filter={filt:.1f} LA={la:.1f} sqrt={sqrt:.1f}")
PY
done
