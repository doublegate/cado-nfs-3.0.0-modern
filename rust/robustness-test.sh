#!/usr/bin/env bash
# Robustness tests for the Rust orchestration layer:
#   1. server: stale-ASSIGNED reassignment after --wutimeout
#   2. server: 410 after POST /control action=finish
#   3. client: --server failover (bad server then good)
#   4. server TLS + client --certsha1 cert-fingerprint pinning
# Run from the repo root:  bash rust/robustness-test.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
SERVER="rust/target/release/cado-wu-server-rs"
CLIENT="rust/target/release/cado-nfs-client-rs"
for b in "$SERVER" "$CLIENT"; do [ -x "$b" ] || { echo "build first"; exit 1; }; done
T="$(mktemp -d /tmp/cado-robust.XXXXXX)"; cd "$T"
mkdir -p filedir
PASS=0; FAIL=0
check(){ if [ "$2" = "$3" ]; then echo "  PASS: $1"; PASS=$((PASS+1)); else echo "  FAIL: $1 (got '$2' want '$3')"; FAIL=$((FAIL+1)); fi; }

seed(){ python3 - "$1" "$2" <<'PY'
import sqlite3,json,sys
db,n=sys.argv[1],int(sys.argv[2]); c=sqlite3.connect(db)
c.executescript("""CREATE TABLE IF NOT EXISTS workunits(wurowid INTEGER PRIMARY KEY ASC UNIQUE NOT NULL,
 wuid VARCHAR(512) UNIQUE NOT NULL, submitter VARCHAR(512), status INTEGER NOT NULL, wu TEXT NOT NULL,
 timecreated TEXT,timeassigned TEXT,assignedclient TEXT,timeresult TEXT,resultclient TEXT,
 errorcode INTEGER,failedcommand INTEGER,timeverified TEXT,retryof INTEGER,priority INTEGER);
CREATE TABLE IF NOT EXISTS files(filesrowid INTEGER PRIMARY KEY ASC UNIQUE NOT NULL,filename TEXT,
 path VARCHAR(512) UNIQUE NOT NULL,type TEXT,command INTEGER,wurowid INTEGER);""")
for i in range(n):
    wu={"id":f"wu{i}","commands":["/bin/echo hi"],"files":{"STDOUT0":{"filename":f"wu{i}.out","upload":True}}}
    c.execute("INSERT INTO workunits(wuid,status,wu,timecreated) VALUES(?,?,?,?)",(f"wu{i}",0,json.dumps(wu),"0"))
c.commit();c.close()
PY
}
st(){ python3 -c "import sqlite3,sys;print(sqlite3.connect(sys.argv[1]).execute('SELECT status FROM workunits WHERE wuid=?',(sys.argv[2],)).fetchone()[0])" "$1" "$2"; }
url(){ for _ in $(seq 1 40); do u=$(grep -oE 'SERVER_URL \S+' "$1" 2>/dev/null|awk '{print $2}'|head -1); [ -n "$u" ]&&{ echo "$u"; return; }; sleep 0.2; done; }

echo "### 1+2: timeout reassignment + 410"
seed wu1.db 1
"$ROOT/$SERVER" --db wu1.db --filedir filedir --uploaddir up1 --wutimeout 2 --port 0 >s1.log 2>&1 & S1=$!
U1=$(url s1.log); echo "  server: $U1"
curl -s --request GET --data 'clientid=c1' "$U1/workunit" >/dev/null
check "assign -> ASSIGNED(1)" "$(st wu1.db wu0)" 1
sleep 6
check "stale reclaimed -> AVAILABLE(0)" "$(st wu1.db wu0)" 0
curl -s --request POST --data 'action=finish' "$U1/control" >/dev/null
code=$(curl -s -o /dev/null -w '%{http_code}' --request GET --data 'clientid=c1' "$U1/workunit")
check "after finish -> 410" "$code" 410
kill $S1 2>/dev/null

echo "### 3: client failover (bad server, then good)"
seed wu2.db 1
"$ROOT/$SERVER" --db wu2.db --filedir filedir --uploaddir up2 --port 0 >s2.log 2>&1 & S2=$!
U2=$(url s2.log)
"$ROOT/$CLIENT" --server http://127.0.0.1:1 --server "$U2" --single \
   --dldir dl2 --workdir work2 --clientid fo >c2.log 2>&1; rc=$?
check "failover client exit 0" "$rc" 0
check "wu recorded via good server (3)" "$(st wu2.db wu0)" 3
kill $S2 2>/dev/null

echo "### 4: server TLS + client --certsha1 pinning"
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 1 -nodes \
   -subj '/CN=localhost' -addext 'subjectAltName=IP:127.0.0.1,DNS:localhost' 2>/dev/null
SHA1=$(openssl x509 -in cert.pem -outform DER | sha1sum | cut -d' ' -f1)
seed wu3.db 1
"$ROOT/$SERVER" --db wu3.db --filedir filedir --uploaddir up3 --cert cert.pem --key key.pem --port 0 >s3.log 2>&1 & S3=$!
U3=$(url s3.log); echo "  server: $U3  cert-sha1: $SHA1"
"$ROOT/$CLIENT" --server "$U3" --certsha1 "$SHA1" --single \
   --dldir dl3 --workdir work3 --clientid tls >c3.log 2>&1; rc=$?
check "TLS+certsha1 client exit 0" "$rc" 0
check "wu recorded over TLS (3)" "$(st wu3.db wu0)" 3
# negative: wrong fingerprint must be rejected
"$ROOT/$CLIENT" --server "$U3" --certsha1 deadbeef --single \
   --dldir dl3b --workdir work3b --clientid tlsbad >c3b.log 2>&1; rc=$?
check "wrong certsha1 rejected (nonzero exit)" "$([ $rc -ne 0 ] && echo bad || echo ok)" bad
kill $S3 2>/dev/null

echo "### 5: server IP whitelist enforcement"
seed wu5.db 1
# a non-loopback whitelist must block the loopback peer (403)...
"$ROOT/$SERVER" --db wu5.db --filedir filedir --uploaddir up5 --whitelist 10.255.255.1 --port 0 >s5.log 2>&1 & S5=$!
U5=$(url s5.log)
code=$(curl -s -o /dev/null -w '%{http_code}' "$U5/")
check "non-loopback whitelist blocks loopback (403)" "$code" 403
kill $S5 2>/dev/null
# ...while admitting 127.0.0.1 lets the same request through (200)
"$ROOT/$SERVER" --db wu5.db --filedir filedir --uploaddir up5b --whitelist 127.0.0.1,::1 --port 0 >s5b.log 2>&1 & S5B=$!
U5B=$(url s5b.log)
code=$(curl -s -o /dev/null -w '%{http_code}' "$U5B/")
check "loopback whitelist admits loopback (200)" "$code" 200
kill $S5B 2>/dev/null

echo "### 6: client STDIN<n> redirection"
python3 - wu6.db <<'PY'
import sqlite3,json,sys
db=sys.argv[1]; c=sqlite3.connect(db)
c.executescript("""CREATE TABLE IF NOT EXISTS workunits(wurowid INTEGER PRIMARY KEY ASC UNIQUE NOT NULL,
 wuid VARCHAR(512) UNIQUE NOT NULL, submitter VARCHAR(512), status INTEGER NOT NULL, wu TEXT NOT NULL,
 timecreated TEXT,timeassigned TEXT,assignedclient TEXT,timeresult TEXT,resultclient TEXT,
 errorcode INTEGER,failedcommand INTEGER,timeverified TEXT,retryof INTEGER,priority INTEGER);
CREATE TABLE IF NOT EXISTS files(filesrowid INTEGER PRIMARY KEY ASC UNIQUE NOT NULL,filename TEXT,
 path VARCHAR(512) UNIQUE NOT NULL,type TEXT,command INTEGER,wurowid INTEGER);""")
# /bin/cat with no args copies stdin -> stdout; STDIN0 feeds it, STDOUT0 is uploaded.
wu={"id":"wu0","commands":["/bin/cat"],
    "files":{"STDIN0":{"filename":"in.txt"},"STDOUT0":{"filename":"wu0.out","upload":True}}}
c.execute("INSERT INTO workunits(wuid,status,wu,timecreated) VALUES(?,?,?,?)",("wu0",0,json.dumps(wu),"0"))
c.commit();c.close()
PY
mkdir -p work6 dl6
printf 'stdin-redirection-payload' > work6/in.txt
"$ROOT/$SERVER" --db wu6.db --filedir filedir --uploaddir up6 --port 0 >s6.log 2>&1 & S6=$!
U6=$(url s6.log)
"$ROOT/$CLIENT" --server "$U6" --single --dldir dl6 --workdir work6 --clientid stdin >c6.log 2>&1; rc=$?
check "STDIN client exit 0" "$rc" 0
check "wu recorded (3)" "$(st wu6.db wu0)" 3
got=$(cat up6/wu0.out 2>/dev/null)
check "stdin flowed through cat to uploaded stdout" "$got" "stdin-redirection-payload"
kill $S6 2>/dev/null

echo "## robustness: $PASS passed, $FAIL failed   (artifacts: $T)"
exit $FAIL
