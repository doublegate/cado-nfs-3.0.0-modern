#!/usr/bin/env bash
# Live interop test for the Rust work-unit client: start a real cado-nfs.py
# server (plain HTTP), then have cado-nfs-client-rs fetch one work-unit, run it,
# and upload the result -- proving it speaks the stock Python api_server protocol.
#
# Run from the repo root:  bash rust/interop-test.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
N="${1:-90377629292003121684002147101760858109247336549001090677693}"   # 59-digit
PY="${PY:-cado-nfs.venv/bin/python3}"                                    # needs flask
CLIENT="rust/target/release/cado-nfs-client-rs"
TMP="$(mktemp -d /tmp/cado-rust-interop.XXXXXX)"
LOG="$TMP/server.log"
mkdir -p "$TMP/dl" "$TMP/work"

[ -x "$CLIENT" ] || { echo "build first: (cd rust && cargo build --release)"; exit 1; }

echo "## starting cado-nfs.py (server.ssl=no) ..."
"$PY" ./cado-nfs.py "$N" server.ssl=no -t 2 >"$LOG" 2>&1 &
CADO_PID=$!
trap 'kill $CADO_PID 2>/dev/null' EXIT

URL=""
for _ in $(seq 1 90); do
  URL=$(grep -oE 'http://[a-zA-Z0-9_.:-]+' "$LOG" 2>/dev/null | head -1)
  [ -n "$URL" ] && break
  kill -0 $CADO_PID 2>/dev/null || { echo "## cado-nfs.py exited early"; tail -5 "$LOG"; exit 1; }
  sleep 1
done
echo "## server URL: ${URL:-NONE}"
[ -z "$URL" ] && { tail -15 "$LOG"; exit 1; }

echo "## running Rust client (--single) ..."
"$CLIENT" --server "$URL" --single --downloadretry 2 \
    --dldir "$TMP/dl" --workdir "$TMP/work" --clientid rust-interop-test 2>&1 | sed 's/^/[client] /'
RC=${PIPESTATUS[0]}
echo "## rust client exit code: $RC"
[ "$RC" -eq 0 ] && echo "## PASS: client fetched, ran, and uploaded a work-unit" \
               || echo "## FAIL"
exit "$RC"
