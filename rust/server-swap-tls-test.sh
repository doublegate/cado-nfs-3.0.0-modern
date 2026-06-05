#!/usr/bin/env bash
# In-process server swap, over TLS: same as server-swap-test.sh but with HTTPS
# left on (the default server.ssl=yes). The shim generates the self-signed cert
# the Python server would have used, serves it from the Rust binary (--cert/
# --key), and reports its SHA1 via get_cert_sha1() so the stock Python clients
# pin it. Validates that TLS works end-to-end through the Rust server.
#
# Run from the repo root:  bash rust/server-swap-tls-test.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
N="${1:-90377629292003121684002147101760858109247336549001090677693}"  # 59-digit
PY="${PY:-cado-nfs.venv/bin/python3}"
export CADO_RUST_WU_SERVER="$ROOT/rust/target/release/cado-wu-server-rs"
[ -x "$CADO_RUST_WU_SERVER" ] || { echo "build first: (cd rust && cargo build --release)"; exit 1; }
T="$(mktemp -d /tmp/cado-swap-tls.XXXXXX)"; LOG="$T/cado.log"

echo "## CADO_RUST_WU_SERVER=$CADO_RUST_WU_SERVER"
echo "## running cado-nfs.py with the Rust server swapped in, TLS ON (Python clients) ..."
# note: NO server.ssl=no -- ssl defaults to yes, so the shim gets a cafile.
"$PY" ./cado-nfs.py "$N" -t 2 >"$LOG" 2>&1
RC=$?

echo "## cado-nfs.py exit code: $RC"
echo "## shim launched the Rust server with TLS?"
grep -m1 "Launching Rust work-unit server" "$LOG" | grep -q -- "--cert" \
   && grep -m1 "Launching Rust work-unit server" "$LOG" | sed 's/^/   /' \
   || echo "   (NO --cert in launch line -- TLS did NOT engage)"
echo "## server reported an https URL?"
grep -m1 "SERVER_URL https://" "$LOG" | sed 's/^/   /' || echo "   (no https SERVER_URL found)"
echo "## Rust server activity (assignments / results):"
python3 -c "
import sys
a=r=0
for l in open('$LOG',encoding='utf-8',errors='replace'):
    if 'rust-wu-server:' in l and 'assigned wu' in l: a+=1
    if 'rust-wu-server:' in l and 'recorded result' in l: r+=1
print(f'   {a} work-units assigned, {r} results recorded by the Rust server')
"
echo "## factors:"; grep -E '^[0-9]+ [0-9]+( [0-9]+)*$' "$LOG" | tail -1 | sed 's/^/   /'
if [ "$RC" -eq 0 ] && grep -q "SERVER_URL https://" "$LOG"; then
    echo "## PASS: TLS factorization completed with the Rust server in-process"
else
    echo "## FAIL"; tail -20 "$LOG"
fi
exit "$RC"
