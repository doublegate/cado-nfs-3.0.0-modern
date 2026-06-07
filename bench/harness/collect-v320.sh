#!/usr/bin/env bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
set -u
cd "$REPO"
B=/tmp/cado-nfs-bench; SDE=/opt/intel-sde/sde64
R=$B/results-v320.txt; : > "$R"
log(){ echo "===== $1 =====" >> "$R"; }
# --- C1: adaptive SpMV ---
log "C1 gpu-spmv-bench (adaptive vec16)"
nvcc -arch=sm_86 -O3 -Xcompiler -pthread bench/gpu-spmv-bench.cu -o $B/spmv 2>>"$R" && $B/spmv >> "$R" 2>&1
# --- A2: Edwards vs ladder ECM ---
log "A2 gpu-ecm-edwards (mixed-rep)"
nvcc -arch=sm_86 -O3 bench/gpu-ecm-edwards.cu -o $B/edw 2>>"$R" && timeout 120 $B/edw >> "$R" 2>&1
# --- C3: batch-smooth ---
log "C3 gpu-batch-smooth"
nvcc -arch=sm_86 -O3 bench/gpu-batch-smooth.cu -lgmp -o $B/bsm 2>>"$R" && timeout 120 $B/bsm >> "$R" 2>&1
# --- C4: sieve-scatter ---
log "C4 gpu-sieve-scatter"
nvcc -arch=sm_86 -O3 -Xcompiler -fopenmp bench/gpu-sieve-scatter.cu -o $B/scat 2>>"$R" && timeout 120 $B/scat >> "$R" 2>&1
# --- B2: VPCLMULQDQ mul2/3/4 (SDE) ---
log "B2 vpclmul-muln (SDE)"
gcc -O2 -mavx512f -mvpclmulqdq bench/vpclmul-muln.c -o $B/vmuln 2>>"$R" && $SDE -future -- $B/vmuln >> "$R" 2>&1
# --- B3: IFMA GF(p) (SDE) ---
log "B3 ifma-gfp (SDE)"
gcc -O2 -mavx512f -mavx512ifma bench/ifma-gfp.c -lgmp -o $B/ifgfp 2>>"$R" && $SDE -future -- $B/ifgfp >> "$R" 2>&1
# --- B1: AVX-512 batched modinv (SDE) ---
log "B1 avx512-modinv (SDE)"
gcc -O2 -mavx512f -mavx512cd bench/avx512-modinv.c -lgmp -o $B/amod 2>>"$R" && $SDE -future -- $B/amod >> "$R" 2>&1
echo "ALL_COLLECTED" >> "$R"
