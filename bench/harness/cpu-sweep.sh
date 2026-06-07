#!/usr/bin/env bash
set -u
PY=cado-nfs.venv/bin/python3
declare -A N=( [c60]=218874463111634589510199972681714178136600659532376772034259
 [c70]=1303194040226516848020750679655954294201842794411873471660917321538889
 [c80]=13719034522081971984611388445022948804646613410389445937337686473743642720735633
 [c90]=298486368711190085093354660667346905640598024729851236663480292985820275742836529265711663 )
for sz in c60 c70 c80 c90; do
  log=/tmp/cado-nfs-bench/sweep-$sz.log
  t0=$(date +%s.%N)
  $PY ./cado-nfs.py "${N[$sz]}" server.ssl=no -t 20 >"$log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")
  cpu=$(grep -oE "Total cpu/elapsed time for entire .*[0-9.]+/[0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+/[0-9.]+" | tail -1)
  facs=$(tail -1 "$log")
  echo "$sz rc=$rc wall=${wall}s cpu/elapsed=$cpu factors=$facs"
done
echo "=== per-phase (cpu seconds) ==="
for sz in c60 c70 c80 c90; do
  log=/tmp/cado-nfs-bench/sweep-$sz.log
  echo "-- $sz --"
  grep -oE "Total cpu time for (polyselect|las|makefb|dup1|dup2|purge|merge|replay|bwc|sqrt|sm)[^,]*" "$log" 2>/dev/null | tail -20
  grep -iE "Lattice Sieving: Total time|Linear Algebra: Total cpu|Square Root: Total cpu|Polynomial Selection.*Total|Filtering.*Total cpu" "$log" | tail -12
done
