#!/usr/bin/env bash
#
# gpu-research-v340-validate.sh — compile and run the two v3.4.0-modern GPU
# research benches:
#
#   * C5+ (gpu-ropt-threshold-c5plus.cu): the conditional-launch threshold for the
#     GPU root-sieve. Sweeps the sieve-line length, checks the kernel is bit-exact
#     vs the CPU int16 sieve at every size, and verifies the launch heuristic routes
#     each size to the measured-faster path (CPU below the crossover, GPU above it).
#
#   * C6+ (gpu-lingen-ntt-crt-c6plus.cu): the multi-modular CRT wrapper around the
#     GPU GF(p) lingen NTT — reduce mod K NTT primes, multiply on the GPU, CRT-
#     reconstruct, and check the result is bit-exact vs an __int128 integer
#     convolution (and after reduction mod a target prime).
#
# Both are honest, gated research items (see docs/{gpu-polyselect-ropt-c5,
# gpu-lingen-ntt-c6}.md): they validate mechanism/correctness, not a single-machine
# speed win. Needs nvcc + an NVIDIA GPU (the reference box has an RTX 3090, sm_86).
#
#     bench/gpu-research-v340-validate.sh
#
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
ARCH="${CUDA_ARCH:-sm_86}"
TMP="${TMPDIR:-/tmp}"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "# NOTE: no nvcc found; skipping (build on a CUDA host)."
    exit 0
fi

build_and_run() {
    local src="$1" bin="$2" label="$3"
    echo "# === $label ==="
    echo "# compiling $src (-arch=$ARCH)"
    nvcc -arch="$ARCH" -O3 "$HERE/$src" -o "$TMP/$bin"
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
        echo "# NOTE: built OK but no GPU visible; run it on a host with an NVIDIA GPU."
        return 0
    fi
    "$TMP/$bin"
    echo
}

echo "# GPU present: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none')"
echo
build_and_run gpu-ropt-threshold-c5plus.cu   gpu-ropt-threshold  "C5+ root-sieve launch threshold"
build_and_run gpu-lingen-ntt-crt-c6plus.cu   gpu-lingen-ntt-crt  "C6+ lingen NTT multi-modular CRT"
