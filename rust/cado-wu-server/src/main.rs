// cado-wu-server-rs -- an async work-unit server for CADO-NFS.
//
// Implements the api_server.py endpoints over the wudb SQLite schema
// (tables `workunits` + `files`; status 0=AVAILABLE 1=ASSIGNED 3=RECEIVED_OK
// 4=RECEIVED_ERROR), so it serves the stock Python clients and cado-nfs-client-rs:
//
//   GET  /            -> hello
//   GET  /workunit    -> assign an AVAILABLE work-unit (form body `clientid`);
//                        410 once serving is finished
//   GET  /file/<path> -> serve an input file from --filedir
//   GET  /files       -> list servable files
//   POST /upload      -> record a result (multipart)
//   POST /control     -> admin: finish | resume serving (optional --admin-token)
//
// Robustness: a connection pool (r2d2 + WAL); a background task that returns
// stale ASSIGNED rows to AVAILABLE after --wutimeout seconds (so a dead client's
// work is re-handed-out); a serving flag that makes /workunit answer 410 when the
// computation is finished; and optional TLS (--cert/--key) matching the Python
// self-signed-cert server.

use anyhow::{Context, Result};
use axum::{
    body::Bytes,
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use r2d2_sqlite::SqliteConnectionManager;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

type Pool = r2d2::Pool<SqliteConnectionManager>;

const AVAILABLE: i64 = 0;
const ASSIGNED: i64 = 1;
const RECEIVED_OK: i64 = 3;
const RECEIVED_ERROR: i64 = 4;

struct AppState {
    pool: Pool,
    filedir: PathBuf,
    uploaddir: PathBuf,
    serving: AtomicBool,
    admin_token: Option<String>,
    whitelist: Vec<String>, // IP/CIDR entries; empty = allow all
}

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = parse_args()?;
    std::fs::create_dir_all(&cfg.uploaddir).ok();

    // pooled connections. WAL lets our reads proceed concurrently with the
    // Python driver's writes (the swap shares one SQLite file; cado uses the
    // default rollback-journal otherwise, where a writer blocks all readers).
    // A long busy_timeout rides out the driver's write transactions instead of
    // failing with SQLITE_BUSY.
    let manager = SqliteConnectionManager::file(&cfg.db)
        .with_init(|c| c.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=30000;"));
    let pool = r2d2::Pool::builder()
        .max_size(8)
        .build(manager)
        .context("building sqlite pool")?;
    {
        let conn = pool.get()?;
        ensure_schema(&conn)?;
    }

    let state = Arc::new(AppState {
        pool: pool.clone(),
        filedir: cfg.filedir,
        uploaddir: cfg.uploaddir,
        serving: AtomicBool::new(true),
        admin_token: cfg.admin_token,
        whitelist: cfg.whitelist,
    });

    // background: reclaim stale ASSIGNED work-units after --wutimeout
    if cfg.wutimeout > 0 {
        let pool2 = pool.clone();
        let wutimeout = cfg.wutimeout;
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(30.min(wutimeout.max(1))));
            loop {
                tick.tick().await;
                if let Ok(conn) = pool2.get() {
                    let cutoff = dt_secs_ago(wutimeout);
                    match conn.execute(
                        "UPDATE workunits SET status=?1, assignedclient=NULL, timeassigned=NULL \
                         WHERE status=?2 AND timeassigned IS NOT NULL AND timeassigned < ?3",
                        rusqlite::params![AVAILABLE, ASSIGNED, cutoff],
                    ) {
                        Ok(n) if n > 0 => {
                            eprintln!("# reclaimed {n} stale work-unit(s) -> AVAILABLE")
                        }
                        _ => {}
                    }
                }
            }
        });
    }

    let app = Router::new()
        .route("/", get(hello))
        .route("/workunit", get(get_workunit))
        .route("/files", get(list_files))
        .route("/file/*path", get(download_file))
        .route("/upload", post(upload))
        // The stock Python client POSTs to `//upload` (POSTRESULTPATH already
        // begins with `/`, and the client joins with another `/`). Flask's
        // Werkzeug merges duplicate slashes; axum does not, so we register the
        // `//upload` form explicitly too. (GET /workunit and /file use slash-less
        // path settings, so only upload is affected.)
        .route("//upload", post(upload))
        .route("/control", post(control))
        .route("/status", get(status))
        .fallback(fallback_404)
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            whitelist_mw,
        ))
        .with_state(state);

    let addr: std::net::SocketAddr = format!("{}:{}", cfg.addr, cfg.port).parse()?;
    let handle = axum_server::Handle::new();
    {
        // print the actually-bound URL (so --port 0 is usable)
        let handle = handle.clone();
        let scheme = if cfg.cert.is_some() { "https" } else { "http" };
        let db = cfg.db.clone();
        tokio::spawn(async move {
            if let Some(local) = handle.listening().await {
                println!("SERVER_URL {scheme}://{local}");
                eprintln!("# cado-wu-server-rs listening on {scheme}://{local}  db={db:?}");
            }
        });
    }

    match (cfg.cert, cfg.key) {
        (Some(cert), Some(key)) => {
            // rustls 0.23 needs a process-default crypto provider installed
            let _ = rustls::crypto::ring::default_provider().install_default();
            let tls = axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert, &key)
                .await
                .with_context(|| format!("loading TLS cert/key {cert:?} {key:?}"))?;
            axum_server::bind_rustls(addr, tls)
                .handle(handle)
                .serve(
                    app.clone()
                        .into_make_service_with_connect_info::<std::net::SocketAddr>(),
                )
                .await?;
        }
        _ => {
            axum_server::bind(addr)
                .handle(handle)
                .serve(
                    app.clone()
                        .into_make_service_with_connect_info::<std::net::SocketAddr>(),
                )
                .await?;
        }
    }
    Ok(())
}

fn ensure_schema(conn: &rusqlite::Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS workunits (
            wurowid INTEGER PRIMARY KEY ASC UNIQUE NOT NULL,
            wuid VARCHAR(512) UNIQUE NOT NULL,
            submitter VARCHAR(512), status INTEGER NOT NULL, wu TEXT NOT NULL,
            timecreated TEXT, timeassigned TEXT, assignedclient TEXT,
            timeresult TEXT, resultclient TEXT,
            errorcode INTEGER, failedcommand INTEGER, timeverified TEXT,
            retryof INTEGER REFERENCES workunits(wurowid), priority INTEGER);
         CREATE TABLE IF NOT EXISTS files (
            filesrowid INTEGER PRIMARY KEY ASC UNIQUE NOT NULL,
            filename TEXT, path VARCHAR(512) UNIQUE NOT NULL,
            type TEXT, command INTEGER,
            wurowid INTEGER REFERENCES workunits(wurowid));",
    )
    .context("creating wudb schema")?;
    Ok(())
}

async fn hello() -> &'static str {
    "CADO-NFS work-unit server (Rust)\n"
}

// logs the method+path of any request that matched no route
async fn fallback_404(method: axum::http::Method, uri: axum::http::Uri) -> Response {
    eprintln!("# 404 no route for {method} {uri}");
    (StatusCode::NOT_FOUND, "no such endpoint").into_response()
}

// IP allow-list, matching api_server.py's api_limit_remote_addr. Empty list =
// allow all; otherwise the peer IP must fall in one of the IP/CIDR entries.
async fn whitelist_mw(
    State(s): State<Arc<AppState>>,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<std::net::SocketAddr>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if !s.whitelist.is_empty() && !s.whitelist.iter().any(|e| ip_matches(peer.ip(), e)) {
        eprintln!("# blocked request from {}", peer.ip());
        return (StatusCode::FORBIDDEN, "forbidden").into_response();
    }
    next.run(req).await
}

fn ip_matches(ip: std::net::IpAddr, entry: &str) -> bool {
    use std::net::IpAddr;
    if let Some((addr, prefix)) = entry.split_once('/') {
        let (Ok(net), Ok(plen)) = (addr.parse::<IpAddr>(), prefix.parse::<u32>()) else {
            return false;
        };
        match (ip, net) {
            (IpAddr::V4(a), IpAddr::V4(b)) => {
                let m = if plen >= 32 {
                    u32::MAX
                } else {
                    !(u32::MAX >> plen)
                };
                (u32::from(a) & m) == (u32::from(b) & m)
            }
            (IpAddr::V6(a), IpAddr::V6(b)) => {
                let m = if plen >= 128 {
                    u128::MAX
                } else {
                    !(u128::MAX >> plen)
                };
                (u128::from(a) & m) == (u128::from(b) & m)
            }
            _ => false,
        }
    } else {
        entry.parse::<IpAddr>().map(|a| a == ip).unwrap_or(false)
    }
}

async fn get_workunit(State(s): State<Arc<AppState>>, body: Bytes) -> Response {
    if !s.serving.load(Ordering::Relaxed) {
        return (StatusCode::GONE, "Distributed computation finished").into_response();
    }
    let form = parse_form(&body);
    let Some(clientid) = form.get("clientid").cloned() else {
        return (StatusCode::FORBIDDEN, "clientid must be provided").into_response();
    };
    let Ok(conn) = s.pool.get() else {
        return (StatusCode::INTERNAL_SERVER_ERROR, "db").into_response();
    };
    let row: Option<(i64, String)> = conn
        .query_row(
            "SELECT wurowid, wu FROM workunits WHERE status=?1 ORDER BY wurowid LIMIT 1",
            [AVAILABLE],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .ok();
    match row {
        None => (StatusCode::NOT_FOUND, "no work available").into_response(),
        Some((rowid, wu)) => {
            match conn.execute(
                "UPDATE workunits SET status=?1, assignedclient=?2, timeassigned=?3 \
                 WHERE wurowid=?4 AND status=?5",
                rusqlite::params![ASSIGNED, clientid, now_dt(), rowid, AVAILABLE],
            ) {
                Ok(1) => {
                    eprintln!("# assigned wu {rowid} to {clientid}");
                    ([("content-type", "application/json")], wu).into_response()
                }
                _ => (StatusCode::NOT_FOUND, "no work available").into_response(),
            }
        }
    }
}

async fn download_file(State(s): State<Arc<AppState>>, Path(path): Path<String>) -> Response {
    if path.split('/').any(|c| c == "..") {
        return (StatusCode::BAD_REQUEST, "bad path").into_response();
    }
    // First the cado-nfs registry table (server_registered_filenames: kkey->path,
    // type 0 = str) used by an in-process swap; then fall back to --filedir for
    // standalone use.
    let mut real: Option<PathBuf> = None;
    if let Ok(conn) = s.pool.get() {
        if let Ok(p) = conn.query_row(
            "SELECT value FROM server_registered_filenames WHERE kkey=?1 AND type=0",
            [&path],
            |r| r.get::<_, String>(0),
        ) {
            real = Some(PathBuf::from(p));
        }
    }
    let full = real.unwrap_or_else(|| s.filedir.join(&path));
    match std::fs::read(&full) {
        Ok(bytes) => ([("content-type", "application/octet-stream")], bytes).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "file not registered").into_response(),
    }
}

async fn list_files(State(s): State<Arc<AppState>>) -> Response {
    let mut names = vec![];
    if let Ok(rd) = std::fs::read_dir(&s.filedir) {
        for e in rd.flatten() {
            if let Ok(n) = e.file_name().into_string() {
                names.push(n);
            }
        }
    }
    axum::Json(names).into_response()
}

// GET /status -- work-unit progress for dashboards/tooling (Track 3.2). The Rust
// server has no view of the orchestration's phase/ETA (that lives in the Python
// driver's status reporter and its --json-status file), but it owns the wudb, so
// it reports live work-unit counts by status plus the serving flag. Reuses the
// same SQLite the /workunit and /upload handlers read.
async fn status(State(s): State<Arc<AppState>>) -> Response {
    let Ok(conn) = s.pool.get() else {
        return (StatusCode::SERVICE_UNAVAILABLE, "db unavailable").into_response();
    };
    let count = |st: i64| -> i64 {
        conn.query_row(
            "SELECT COUNT(*) FROM workunits WHERE status=?1",
            [st],
            |r| r.get(0),
        )
        .unwrap_or(0)
    };
    let total: i64 = conn
        .query_row("SELECT COUNT(*) FROM workunits", [], |r| r.get(0))
        .unwrap_or(0);
    let available = count(AVAILABLE);
    let assigned = count(ASSIGNED);
    let ok = count(RECEIVED_OK);
    let error = count(RECEIVED_ERROR);
    let done = ok + error;
    let percent = if total > 0 {
        (done as f64) * 100.0 / (total as f64)
    } else {
        0.0
    };
    axum::Json(serde_json::json!({
        "schema": "cado-nfs-wu-status/1",
        "server": "cado-wu-server-rs",
        "serving": s.serving.load(Ordering::Relaxed),
        "workunits": {
            "total": total,
            "available": available,
            "assigned": assigned,
            "ok": ok,
            "error": error,
            "done": done,
        },
        "percent": (percent * 10.0).round() / 10.0,
    }))
    .into_response()
}

// POST /control -- finish or resume serving work-units. Guarded by --admin-token
// when set (form field `token`). Lets a driver signal "computation done" -> 410.
async fn control(State(s): State<Arc<AppState>>, body: Bytes) -> Response {
    let form = parse_form(&body);
    if let Some(tok) = &s.admin_token {
        if form.get("token").map(|t| t != tok).unwrap_or(true) {
            return (StatusCode::FORBIDDEN, "bad token").into_response();
        }
    }
    match form.get("action").map(|a| a.as_str()) {
        Some("finish") => {
            s.serving.store(false, Ordering::Relaxed);
            (StatusCode::OK, "serving finished\n").into_response()
        }
        Some("resume") => {
            s.serving.store(true, Ordering::Relaxed);
            (StatusCode::OK, "serving resumed\n").into_response()
        }
        _ => (StatusCode::BAD_REQUEST, "action must be finish|resume").into_response(),
    }
}

async fn upload(State(s): State<Arc<AppState>>, mut mp: Multipart) -> Response {
    let mut fields: HashMap<String, String> = HashMap::new();
    let mut saved: Vec<(String, PathBuf)> = vec![];
    while let Some(field) = match mp.next_field().await {
        Ok(f) => f,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("multipart: {e}")).into_response(),
    } {
        let name = field.name().unwrap_or("").to_string();
        match field.file_name().map(|s| s.to_string()) {
            Some(fname) => {
                let data = match field.bytes().await {
                    Ok(b) => b,
                    Err(e) => {
                        return (StatusCode::BAD_REQUEST, format!("read: {e}")).into_response()
                    }
                };
                let safe = sanitize(&fname);
                let dest = s.uploaddir.join(&safe);
                if dest.exists() {
                    return (StatusCode::FORBIDDEN, "File already exists").into_response();
                }
                if std::fs::write(&dest, &data).is_err() {
                    return (StatusCode::INTERNAL_SERVER_ERROR, "save").into_response();
                }
                saved.push((safe, dest));
            }
            None => {
                if let Ok(t) = field.text().await {
                    fields.insert(name, t);
                }
            }
        }
    }
    if std::env::var("CADO_WU_DEBUG").is_ok() {
        let keys: Vec<&String> = fields.keys().collect();
        eprintln!("# upload fields={keys:?} files={}", saved.len());
    }
    let (Some(clientid), Some(wuid)) = (fields.get("clientid"), fields.get("WUid")) else {
        return (StatusCode::BAD_REQUEST, "missing WUid and/or clientid").into_response();
    };
    let fileinfo: serde_json::Value =
        serde_json::from_str(fields.get("fileinfo").map(|s| s.as_str()).unwrap_or("{}"))
            .unwrap_or(serde_json::json!({}));
    let errorcode: Option<i64> = fields.get("errorcode").and_then(|v| v.parse().ok());
    record_result(&s, wuid, clientid, &saved, &fileinfo, errorcode)
}

fn record_result(
    s: &AppState,
    wuid: &str,
    clientid: &str,
    saved: &[(String, PathBuf)],
    fileinfo: &serde_json::Value,
    errorcode: Option<i64>,
) -> Response {
    let Ok(conn) = s.pool.get() else {
        return (StatusCode::INTERNAL_SERVER_ERROR, "db").into_response();
    };
    // CRITICAL: distinguish a genuinely-absent WUid (-> 404) from a transient DB
    // error such as a lock (-> 503, retryable). Collapsing both to 404 makes the
    // client give up and the driver resubmit, which deadlocks the swap.
    let (rowid, status) = match conn.query_row(
        "SELECT wurowid, status FROM workunits WHERE wuid=?1",
        [wuid],
        |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)),
    ) {
        Ok(row) => row,
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            eprintln!("# upload: WUid {wuid:?} not found in workunits");
            return (StatusCode::NOT_FOUND, "unknown WUid").into_response();
        }
        Err(e) => {
            eprintln!("# upload: db error looking up {wuid:?}: {e}");
            return (StatusCode::SERVICE_UNAVAILABLE, "db busy, retry").into_response();
        }
    };
    if status != ASSIGNED {
        eprintln!("# warning: wu {wuid} not currently ASSIGNED (status {status})");
    }
    for (basename, destpath) in saved {
        let info = fileinfo.get(basename);
        let key = info
            .and_then(|v| v.get("key"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        // `command` (the command index a STDOUT/STDERR file came from) must be
        // stored: the driver does int(files.command) when reading stdio. It may
        // arrive as a JSON number or string.
        let command: Option<i64> = info.and_then(|v| v.get("command")).and_then(|c| {
            c.as_i64()
                .or_else(|| c.as_str().and_then(|s| s.parse().ok()))
        });
        let _ = conn.execute(
            "INSERT INTO files (filename, path, type, command, wurowid) VALUES (?1,?2,?3,?4,?5)",
            rusqlite::params![basename, destpath.to_string_lossy(), key, command, rowid],
        );
    }
    let new_status = if errorcode.unwrap_or(0) == 0 {
        RECEIVED_OK
    } else {
        RECEIVED_ERROR
    };
    let _ = conn.execute(
        "UPDATE workunits SET status=?1, resultclient=?2, errorcode=?3, timeresult=?4 \
         WHERE wurowid=?5",
        rusqlite::params![new_status, clientid, errorcode, now_dt(), rowid],
    );
    eprintln!(
        "# recorded result for wu {wuid} from {clientid}: status {new_status}, {} files",
        saved.len()
    );
    (StatusCode::OK, "ok").into_response()
}

// Match the Python wudb's timestamp format, str(datetime.utcnow()):
// "YYYY-MM-DD HH:MM:SS.ffffff". The driver parses these to decide work-unit
// timeouts, so the format must match for the in-process swap to interoperate.
fn now_dt() -> String {
    chrono::Utc::now()
        .format("%Y-%m-%d %H:%M:%S%.6f")
        .to_string()
}
// A timestamp `secs` seconds in the past, same format -- used (with lexical
// comparison, which is valid for this zero-padded format) to find stale rows.
fn dt_secs_ago(secs: u64) -> String {
    (chrono::Utc::now() - chrono::Duration::seconds(secs as i64))
        .format("%Y-%m-%d %H:%M:%S%.6f")
        .to_string()
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "._-".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn parse_form(body: &[u8]) -> HashMap<String, String> {
    let mut m = HashMap::new();
    let s = String::from_utf8_lossy(body);
    for pair in s.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            m.insert(urldecode(k), urldecode(v));
        }
    }
    m
}

fn urldecode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => {
                out.push(' ');
                i += 1;
            }
            b'%' if i + 2 < b.len() => match u8::from_str_radix(&s[i + 1..i + 3], 16) {
                Ok(byte) => {
                    out.push(byte as char);
                    i += 3;
                }
                Err(_) => {
                    out.push('%');
                    i += 1;
                }
            },
            c => {
                out.push(c as char);
                i += 1;
            }
        }
    }
    out
}

struct Cfg {
    db: PathBuf,
    filedir: PathBuf,
    uploaddir: PathBuf,
    addr: String,
    port: u16,
    wutimeout: u64,
    cert: Option<PathBuf>,
    key: Option<PathBuf>,
    admin_token: Option<String>,
    whitelist: Vec<String>,
}

/// clap front-end (Track 3.4: real --help / validation). Same flag names and
/// semantics as before; mapped into Cfg so main() and the rest are unchanged.
/// `--whitelist` still accepts a comma-separated list (and may be repeated).
#[derive(clap::Parser)]
#[command(
    name = "cado-wu-server-rs",
    version,
    about = "Async work-unit server for CADO-NFS \
                   (same HTTP protocol + wudb SQLite schema as api_server.py)"
)]
struct Args {
    /// wudb SQLite database (shared with the Python driver)
    #[arg(long)]
    db: PathBuf,
    /// directory of downloadable input files
    #[arg(long, default_value = ".")]
    filedir: PathBuf,
    /// directory for uploaded results (default: <tmp>/cado-wu-uploads)
    #[arg(long)]
    uploaddir: Option<PathBuf>,
    /// bind address
    #[arg(long, default_value = "127.0.0.1")]
    addr: String,
    /// bind port (0 = pick a free port and print it)
    #[arg(long, default_value_t = 0)]
    port: u16,
    /// seconds before a stale ASSIGNED work-unit is reclaimed (0 = never)
    #[arg(long, default_value_t = 3600)]
    wutimeout: u64,
    /// TLS certificate (PEM); pair with --key to serve HTTPS
    #[arg(long)]
    cert: Option<PathBuf>,
    /// TLS private key (PEM)
    #[arg(long)]
    key: Option<PathBuf>,
    /// admin token required by POST /control
    #[arg(long = "admin-token")]
    admin_token: Option<String>,
    /// allowed client IP/CIDR(s), comma-separated and/or repeated (empty = all)
    #[arg(long, value_delimiter = ',')]
    whitelist: Vec<String>,
    /// print a shell completion script (bash|zsh|fish|...) to stdout and exit
    #[arg(long, value_enum, exclusive = true)]
    completions: Option<clap_complete::Shell>,
}

/// `--completions <shell>` (Roadmap E6): emit the script to stdout and exit,
/// before clap's required-argument check runs. Handles `--completions bash` and
/// `--completions=bash`.
fn maybe_emit_completions() {
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let val = if a == "--completions" {
            it.next()
        } else if let Some(v) = a.strip_prefix("--completions=") {
            Some(v.to_string())
        } else {
            continue;
        };
        if let Some(v) = val {
            use clap::{CommandFactory, ValueEnum};
            if let Ok(shell) = clap_complete::Shell::from_str(&v, true) {
                clap_complete::generate(
                    shell,
                    &mut Args::command(),
                    "cado-wu-server-rs",
                    &mut std::io::stdout(),
                );
                std::process::exit(0);
            }
        }
        return;
    }
}

fn parse_args() -> Result<Cfg> {
    use clap::Parser;
    maybe_emit_completions();
    let a = Args::parse();
    Ok(Cfg {
        db: a.db,
        filedir: a.filedir,
        uploaddir: a
            .uploaddir
            .unwrap_or_else(|| std::env::temp_dir().join("cado-wu-uploads")),
        addr: a.addr,
        port: a.port,
        wutimeout: a.wutimeout,
        cert: a.cert,
        key: a.key,
        admin_token: a.admin_token,
        whitelist: a
            .whitelist
            .into_iter()
            .map(|x| x.trim().to_string())
            .filter(|x| !x.is_empty())
            .collect(),
    })
}
