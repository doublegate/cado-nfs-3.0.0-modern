#!/usr/bin/env bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
set -u
R="$REPO"
cd /tmp/cado-nfs-bench
echo "######## GPU pre-factoring ECM (CPU vs GPU) ########"
nvcc -arch=sm_86 -O3 -Xcompiler -pthread "$R/bench/gpu-prefactor-bench.cu" -lgmp -o gpb 2>&1 | tail -2 && ./gpb
echo; echo "######## GPU BWC SpMV scaling sweep (b64) ########"
head -163 "$R/bench/gpu-spmv-bench.cu" > sweep.cu
cat >> sweep.cu <<'M'
int main(){
    printf("GF(2) BWC SpMV scaling sweep (b64, ~30 nnz/row) -- RTX 3090 vs full CPU\n");
    int fails=0;
    fails += run<1>("c100~", 1000000, 30, 60);
    fails += run<1>("c110~", 2000000, 30, 60);
    fails += run<1>("c115~", 4000000, 30, 50);
    fails += run<1>("c120~", 8000000, 30, 40);
    printf("%s\n", fails==0?"ALL PASS":"FAILURES");
    return fails!=0;
}
M
nvcc -arch=sm_86 -O3 -Xcompiler -pthread sweep.cu -o sweep 2>&1 | tail -2 && ./sweep
echo; echo "######## AVX-512 VPCLMULQDQ mul1 (SDE) ########"
SDE=/opt/intel-sde/sde64 bash "$R/bench/vpclmul-validate.sh" 2>&1 | tail -2
echo; echo "######## AVX-512 IFMA GF(p) modmul (SDE) ########"
SDE=/opt/intel-sde/sde64 bash "$R/bench/ifma-validate.sh" 2>&1 | tail -2
echo "######## DONE ########"
