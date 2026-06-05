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

## Scope

**Implemented & validated:** the complete single-server client loop — WU fetch,
checksummed downloads, prefix-mapped command substitution, no-shell exec,
stdout/stderr routing, multipart upload — interoperating live with the Python
server.

**Deferred (documented follow-ons):**
- Client: multi-server failover, automatic certificate download/pinning
  (`--certsha1`), file locking + half-download backlog, `STDIN` redirection
  (dead in the Python client too), `--niceness`.
- **Server + DB** (plan item 1): port `api_server.py` + the `wudb` SQLite layer
  to async Rust (axum/tokio + rusqlite/sqlx) behind the same protocol, so the
  Rust server interoperates with the Python driver and clients. This is the
  larger scalability piece; the client above is the self-contained first step and
  the proof that the protocol is correctly understood.
