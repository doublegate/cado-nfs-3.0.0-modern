# Rust orchestration (Phase 4)

CADO-NFS distributes its sieving/polyselect work over HTTP: a Python **server**
(`scripts/cadofactor/api_server.py`, Flask) hands **work-units** to **clients**
(`cado-nfs-client.py`), which run the commands and upload results. The math is in
C/C++; this layer is the network/DB substrate. Phase 4 ports it to Rust for a
single static-binary client and (later) an async server — for robustness and
many-client scaling, **not** single-machine factoring speed.

The guiding constraint: **keep the exact existing HTTP/JSON protocol**, so a Rust
binary interoperates with an unmodified `cado-nfs.py` run during migration.

## The work-unit protocol (as implemented by `api_server.py`)

Five endpoints:

| method + path | purpose |
|---|---|
| `GET /` | health/hello |
| `GET /workunit` | hand out a fresh work-unit |
| `GET /file/<path>` | download an input file (binary, poly, factor base, …) |
| `GET /files` | list registered files |
| `POST /upload` | upload result files |

Two non-obvious details, both matched by the Rust client:

- **`GET /workunit` carries `clientid` in a form-urlencoded *body*** (the Python
  client does `requests.get(url, data={'clientid': ...})`; Werkzeug parses it
  into `request.form`). Responses: `200` work-unit JSON, `404` no work yet (retry),
  `410` computation finished (exit).
- **`POST /upload` is `multipart/form-data`** with text fields `clientid`,
  `WUid`, optional `errorcode`/`failedcommand`, a `fileinfo` JSON
  (`{basename: {WUid, key}}`), plus the result files.

A work-unit (`workunit.py`) is JSON:

```json
{ "id": "c60_polyselect1_0-5000",
  "commands": ["${EXECFILE} -P 420 -N ... -admax 5000"],
  "timeout": 10800,
  "files": {
    "EXECFILE": {"filename":"polyselect","download":true,"checksum":"...","algorithm":"sha1","suggest_path":"polyselect"},
    "STDOUT0":  {"filename":"...","upload":true}
  } }
```

Each file id's prefix maps it to a directory and role: `FILE*`/`EXECFILE*` →
download dir, `WDIR*`/`RESULT*`/`STDOUT*`/`STDERR*`/`STDIN*` → work dir. Commands
use `$FID`/`${FID}` placeholders substituted with the local file paths (Python
`string.Template.safe_substitute`); the client strips `'` (bug 21827), splits the
result on spaces, and execs directly (no shell).

## `cado-nfs-client-rs` (this deliverable)

`rust/cado-nfs-client` — a single static binary (reqwest + **rustls**, no
OpenSSL; serde_json; sha1/sha2/sha3) implementing the full client loop:

1. `GET /workunit` (form body `clientid`) → parse WU JSON (`404`→wait, `410`→exit).
2. download every `download:true` file from `/file/<name>` (with `$ARCH`
   substitution), **verify its sha1/sha256/sha3_256 checksum**, mark `EXECFILE*`
   executable.
3. build the file-id→path map by prefix, substitute `$FID`/`${FID}` into each
   command, run them (argv split on spaces, no shell — exactly as the Python
   client), routing each command's stdout/stderr to its `STDOUT%d`/`STDERR%d`
   file or capturing it for upload.
4. `POST /upload` (multipart) the `upload:true` files + captured stdio, with the
   `fileinfo` JSON and `WUid`/`clientid`/`errorcode`/`failedcommand`.

```
cd rust && cargo build --release      # -> rust/target/release/cado-nfs-client-rs
cado-nfs-client-rs --server http://host:port [--clientid ID] \
    [--dldir DIR] [--workdir DIR] [--arch S] [--downloadretry SECS] [--single]
# TLS: env CADO_NFS_INSECURE=1 (accept the self-signed dev cert) or CADO_NFS_CAFILE=<pem>
```

### Validated: real interop with the stock Python server

`rust/interop-test.sh` starts an unmodified `cado-nfs.py <N> server.ssl=no` and
points the Rust client at it. Result:

```
# got workunit c60_polyselect1_0-5000
# running: .../polyselect -P 420 -N 9037762929...693 -degree 4 -t 2 -admin 0 -admax 5000 -incr 60 -nq 64
# uploaded results for c60_polyselect1_0-5000
## rust client exit code: 0
```

The Rust client fetched a genuine work-unit, **downloaded + checksum-verified +
chmod'd the `polyselect` binary**, ran the real command, and **uploaded the
result, which the Python server accepted (HTTP 200)** — full protocol interop.

## `cado-wu-server-rs` (the server + DB port)

`rust/cado-wu-server` — an async server (**axum** + **tokio** + **rusqlite**,
bundled SQLite so it is self-contained) that implements the same five endpoints
over the same **`wudb` SQLite schema** (`wudb.py` tables `workunits` and
`files`, with the exact `status` codes: 0 AVAILABLE, 1 ASSIGNED, 3 RECEIVED_OK,
4 RECEIVED_ERROR, …). It replicates the `WuAccess` core logic:

- **`GET /workunit`** → assign: `SELECT … WHERE status=0 LIMIT 1`, then
  `UPDATE status=1, assignedclient, timeassigned` (single-writer atomic; lost
  races return 404). `404` when none available.
- **`POST /upload`** → record: verify the row is ASSIGNED, save the result files,
  `INSERT` them into `files` (linked by `wurowid`), `UPDATE status=3/4,
  resultclient, errorcode, timeresult`.
- **`GET /file/<path>`** serves input files from `--filedir` (with a `..` guard);
  **`GET /files`** lists them; **`GET /`** is the health check.

Because it reads/writes the exact `wudb` schema and status codes, a database the
Python driver (`cadotask.py`) populates with AVAILABLE rows is consumable by this
server, and the results it writes back are visible to the driver.

```
cd rust && cargo build --release      # -> rust/target/release/cado-wu-server-rs
cado-wu-server-rs --db <wu.db> [--filedir DIR] [--uploaddir DIR] [--addr 127.0.0.1] [--port N]
# --port 0 picks an ephemeral port; the bound URL is printed as `SERVER_URL http://...`
```

### Validated: full work-unit lifecycle, Rust server + Rust client

`rust/server-interop-test.sh` seeds a DB (wudb schema) with one AVAILABLE
work-unit, starts the server, and runs the Rust client against it:

```
got workunit testwu1 -> cat input.txt (sha1-verified) -> uploaded results
workunit status=3 (RECEIVED_OK) resultclient=srvtest
result files recorded=1  type=STDOUT0
uploaded result content='hello cado rust'   ## PASS
```

The row advanced **AVAILABLE(0) → ASSIGNED(1) → RECEIVED_OK(3)** and the result
file was recorded in `files` — the complete server-side lifecycle over the real
schema, driving the same Rust client that also runs against the Python server.

## Robustness features (all validated)

`rust/robustness-test.sh` exercises the production-hardening features — **13/13
pass**:

- **Server, stale-work reassignment** — a background task returns ASSIGNED rows
  to AVAILABLE after `--wutimeout` seconds, so a dead client's work is re-handed
  out (verified: assign → wait → row back to AVAILABLE).
- **Server, serving-finished `410`** — `POST /control action=finish|resume`
  toggles a flag; `/workunit` then answers `410` so clients terminate cleanly
  (the Python driver's "computation done" signal).
- **Server, connection pool** — r2d2 over SQLite in WAL mode (was a single
  mutex-guarded connection).
- **Server TLS** — `--cert/--key` PEM serve HTTPS (axum-server + rustls),
  matching the Python self-signed-cert server.
- **Client failover** — multiple `--server` URLs; each work-unit's downloads and
  upload stick to the server that handed it out; connection errors rotate to the
  next, and *all-unreachable* exits non-zero (distinct from "no work").
- **Client cert pinning** — `--certsha1 <hex>` accepts the TLS handshake only if
  the server's certificate hashes to the pinned value (verified: correct pin
  connects over TLS; a wrong pin is rejected). Plus `--insecure` / `--cafile`.
- **Client `--niceness`** renices children; downloads take an advisory `flock`.
- **Server IP whitelist** — `--whitelist <ip|cidr,…>` (matching `api_server.py`'s
  `api_limit_remote_addr`): empty allows all, otherwise the peer IP must fall in
  one of the IPv4/IPv6 exact-or-CIDR entries, else `403` (verified: a non-loopback
  whitelist blocks loopback, a loopback whitelist admits it).
- **Client `STDIN<n>` redirection** — if a work-unit declares a `STDIN<n>` file
  for command *n*, the client opens it and feeds it to the child's stdin (verified
  end-to-end: payload flows `STDIN0 → /bin/cat → uploaded STDOUT0`). The stock
  Python client never implemented this; the Rust client does.

## Deployment: a real factorization, driven entirely by Rust clients

`rust/deploy-test.sh` runs `cado-nfs.py <N> server.ssl=no slaves.nrclients=0
server.whitelist=localhost` — i.e. cado-nfs.py starts **only the server +
driver, no Python clients** — and launches two `cado-nfs-client-rs` instances as
the *only* workers. They process every polyselect and `las` sieve work-unit until
the server signals `410`:

```
cado-nfs.py exit code: 0
factors: 260938498861057 588120598053661 760926063870977 773951836515617
19 work-units processed by Rust clients   ## PASS
```

The 59-digit input was **factored end-to-end with the Rust client as the sole
distributed worker** — the deployment integration, validated. (`slaves.nrclients=0`
is the stock "I'll provide my own clients" mode; `server.whitelist=localhost`
admits the external client, which cado-nfs.py would otherwise whitelist only for
the hosts it spawns itself.)

## In-process server swap: the Rust server *inside* a cado-nfs.py run

The final integration: `cado-wu-server-rs` running **in place of**
`api_server.py` within a normal `cado-nfs.py` run. A thin Python shim
(`scripts/cadofactor/external_api_server.py`, `ExternalApiServer`) implements the
`ApiServer` interface the driver uses (`get_port`/`serve`/`shutdown`/
`stop_serving_wus`/`get_url`/`get_cert_sha1`) by launching the Rust binary as a
subprocess over the same `wudb` SQLite database; `cadotask.py` switches to it when
`CADO_RUST_WU_SERVER` is set. Input files are served from the
`server_registered_filenames` table the driver already maintains.

`rust/server-swap-test.sh` validates it with the **stock Python clients**:

```
CADO_RUST_WU_SERVER=…/cado-wu-server-rs  cado-nfs.py <N> server.ssl=no -t 2
→ 16 work-units assigned, 15 results recorded by the Rust server
→ factors: 260938498861057 588120598053661 760926063870977 773951836515617
→ cado-nfs.py exit 0   ## PASS
```

Getting this to work surfaced three protocol details the standalone tests could
not (now all fixed in `cado-wu-server-rs`):
1. **`timeassigned` format** — must be `str(datetime.utcnow())`
   (`YYYY-MM-DD HH:MM:SS.ffffff`), not epoch seconds, or the driver mis-judges
   timeouts and resubmits endlessly.
2. **`//upload`** — the stock client POSTs to a double-slash path (POSTRESULTPATH
   begins with `/` and the client joins with another `/`); Flask/Werkzeug merge
   slashes, axum does not, so the `//upload` route is registered explicitly.
3. **`files.command`** — the result-file row must store the originating command
   index (from `fileinfo`), which the driver reads as `int(files.command)` when
   collecting stdio.

### Whitelist + TLS through the swap (both validated)

The shim now passes the driver's `server.whitelist` through to the Rust server's
`--whitelist` (resolving hostnames to IPs, always admitting loopback), so the
IP allow-list is enforced by the Rust process itself rather than bypassed.
`server-swap-test.sh` runs with it active (`--whitelist 127.0.0.1,…,::1`).

TLS is wired end-to-end. With the default `server.ssl=yes`, the shim generates
the same self-signed cert the Python server would (`cadofactor_tools.certificate.
create_certificate`, SAN augmented with `127.0.0.1`/`::1` because the Rust server
binds loopback and the `requests`-based client verifies the hostname against the
pinned cert), serves it with `--cert/--key`, and reports its SHA1 via
`get_cert_sha1()` so the stock Python clients pin it exactly as before.
`rust/server-swap-tls-test.sh`:

```
cado-nfs.py <N> -t 2   (CADO_RUST_WU_SERVER set, ssl left on)
→ launch line carries --cert/--key; SERVER_URL https://127.0.0.1:PORT
→ 16 work-units assigned, 15 results recorded by the Rust server (over TLS)
→ factors: 260938498861057 588120598053661 760926063870977 773951836515617
→ cado-nfs.py exit 0   ## PASS
```

## Scope

**Implemented & validated end-to-end:**
- Client (full loop, failover, TLS + cert-pinning, niceness, flock, `STDIN<n>`
  redirection) — live with the Python server (`interop-test.sh`).
- Server (five endpoints, `wudb` assign/result lifecycle, timeout reassignment,
  `410`, pool, TLS, IP whitelist) — with the Rust client (`server-interop-test.sh`,
  `robustness-test.sh` 13/13).
- A real factorization run entirely by **external Rust clients** against the
  stock cado-nfs.py driver (`deploy-test.sh`).
- A real factorization run with the **Rust server swapped in** for `api_server.py`,
  serving the stock Python clients, over **plain HTTP** (`server-swap-test.sh`)
  and over **TLS** with cert-pinning + whitelist enforcement
  (`server-swap-tls-test.sh`).

**Deliberately left in Python:** the high-level task DAG / scheduler
(`cadotask.py`, ~7.5K LOC). Phase 4's scope is the *network/DB substrate* — the
work-unit transport, the `wudb` store, the client — which is where Rust buys
robustness (no GIL, connection pooling, a single static binary) without touching
the NFS mathematics. `cadotask.py` is the orchestration *logic* (which sub-task
runs when, with which parameters); porting it is a multi-week effort with no
robustness or throughput payoff and a large correctness surface, so the plan
keeps it in Python behind the stable `ApiServer` seam the shim implements. This
is a scope decision, not an unfinished item — the Rust substrate is complete and
validated both standalone and swapped into a live `cado-nfs.py` run.
