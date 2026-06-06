#!/usr/bin/env bash
#
# ifma-validate.sh — compile the AVX-512 IFMA batched Montgomery modmul kernel
# and bit-exactly validate it against GMP under Intel SDE (emulating a CPU with
# AVX-512-IFMA, since the dev box is Comet Lake = no IFMA). Track 1.4.
#
# Install SDE first (e.g. `paru -S intel-sde`), then:
#     bench/ifma-validate.sh                 # auto-detects sde64
#     SDE=/path/to/sde64 bench/ifma-validate.sh
#
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/ifma-modmul.c"
BIN="${TMPDIR:-/tmp}/ifma-modmul"
SDE="${SDE:-$(command -v sde64 || command -v sde \
    || ls /opt/intel-sde/sde64 2>/dev/null \
    || ls /opt/sde*/sde64 2>/dev/null | head -1 || true)}"

echo "# compiling $SRC with -mavx512f -mavx512ifma"
${CC:-gcc} -O2 -mavx512f -mavx512ifma -Wall "$SRC" -lgmp -o "$BIN"

if [ -z "$SDE" ]; then
    echo "# NOTE: Intel SDE not found (set \$SDE or install it)."
    echo "# Binary built OK; it cannot run on a non-IFMA host (would SIGILL)."
    echo "# Verify the IFMA instruction is present instead:"
    objdump -d "$BIN" | grep -m2 -iE "vpmadd52" || echo "  (no vpmadd52 found — unexpected)"
    exit 0
fi

echo "# running under: $SDE -future"
"$SDE" -future -- "$BIN"
