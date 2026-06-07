// cado-nfs-client-rs -- a static-binary work-unit client for CADO-NFS.
//
// Speaks the exact HTTP/JSON protocol of the stock Python server
// (scripts/cadofactor/api_server.py + workunit.py), so it interoperates with an
// unmodified cado-nfs.py run:
//
//   GET  /workunit  (form body `clientid=...`)  -> 200 WU JSON | 404 wait | 410 done
//   GET  /file/<name>                            -> input file (sha1/256/3_256 checked)
//   POST /upload    (multipart: clientid, WUid, fileinfo JSON, result files)
//
// Loop: fetch a WU, download its `download` files (checksum-verified, advisory
// flock so co-located clients don't race), substitute `$FID`/`${FID}` into each
// command (file ids dir-mapped by prefix), run them (argv split on spaces, no
// shell, as the Python client; optional renice), capturing stdout/stderr to
// STDOUT%d/STDERR%d files or for upload, then POST the `upload` files + stdio.
//
// Robustness: pass --server multiple times for failover (each WU's downloads and
// upload stick to the server that handed it out); --niceness renices children;
// TLS via --insecure / CADO_NFS_CAFILE / --certsha1 (cert fingerprint pinning).

use anyhow::{bail, Context, Result};
use sha1::Digest;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(serde::Deserialize)]
struct FileSpec {
    filename: String,
    checksum: Option<String>,
    algorithm: Option<String>,
    #[serde(default)]
    upload: bool,
    #[serde(default)]
    download: bool,
    #[allow(dead_code)]
    suggest_path: Option<String>,
}

#[derive(serde::Deserialize)]
struct Workunit {
    id: String,
    #[serde(default)]
    commands: Vec<String>,
    #[serde(default)]
    files: HashMap<String, FileSpec>,
}

struct Settings {
    servers: Vec<String>,
    clientid: String,
    dldir: PathBuf,
    workdir: PathBuf,
    arch: String,
    download_retry: u64,
    single: bool,
    niceness: i32,
}

enum Fetch {
    Work(Workunit),
    Empty,       // some server reachable, but no work (404)
    Unreachable, // no server could be reached at all
    Done,        // 410
}

fn main() -> Result<()> {
    let s = parse_args()?;
    std::fs::create_dir_all(&s.dldir).ok();
    std::fs::create_dir_all(&s.workdir).ok();
    let client = build_http_client()?;
    eprintln!(
        "# cado-nfs-client-rs: {} server(s), clientid={}",
        s.servers.len(),
        s.clientid
    );

    let mut idx = 0usize;
    loop {
        match fetch_wu_failover(&client, &s, &mut idx) {
            (Fetch::Done, _) => {
                eprintln!("# server says the computation is finished; exiting");
                break;
            }
            (Fetch::Empty, _) => {
                if s.single {
                    eprintln!("# no work available; --single, exiting");
                    break;
                }
                std::thread::sleep(Duration::from_secs(s.download_retry));
            }
            (Fetch::Unreachable, _) => {
                eprintln!("# no server could be reached");
                if s.single {
                    std::process::exit(2);
                }
                std::thread::sleep(Duration::from_secs(s.download_retry));
            }
            (Fetch::Work(wu), server) => {
                let id = wu.id.clone();
                match process_wu(&client, &s, &server, &wu) {
                    Ok(()) => eprintln!("# workunit {id} done"),
                    Err(e) => eprintln!("# workunit {id} failed: {e:#}"),
                }
                if s.single {
                    break;
                }
            }
        }
    }
    Ok(())
}

fn build_http_client() -> Result<reqwest::blocking::Client> {
    let mut b = reqwest::blocking::Client::builder().timeout(Duration::from_secs(600));
    if std::env::var("CADO_NFS_INSECURE").is_ok() {
        b = b
            .danger_accept_invalid_certs(true)
            .danger_accept_invalid_hostnames(true);
    }
    if let Ok(ca) = std::env::var("CADO_NFS_CAFILE") {
        let pem = std::fs::read(&ca).with_context(|| format!("reading cafile {ca}"))?;
        b = b.add_root_certificate(reqwest::Certificate::from_pem(&pem).context("parsing cafile")?);
    }
    // --certsha1: pin the server cert by sha1 fingerprint. A custom rustls
    // verifier accepts the connection iff the presented end-entity cert hashes to
    // the pinned value (the Python client's "trust this exact cert" model). This
    // works regardless of CA flags/hostname, which is the point of pinning.
    if let Ok(fp) = std::env::var("CADO_NFS_CERTSHA1") {
        let want = fp.replace([':', ' '], "").to_ascii_lowercase();
        let provider = std::sync::Arc::new(rustls::crypto::ring::default_provider());
        let config = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .context("tls versions")?
            .dangerous()
            .with_custom_certificate_verifier(std::sync::Arc::new(certpin::PinnedVerifier { want }))
            .with_no_client_auth();
        eprintln!("# pinning server cert sha1={fp}");
        b = b.use_preconfigured_tls(config);
    }
    b.build().context("building HTTP client")
}

// Try each server (rotating from idx) until one answers; failover on connection
// errors. Returns the work-unit/verdict and the server URL that produced it.
fn fetch_wu_failover(
    client: &reqwest::blocking::Client,
    s: &Settings,
    idx: &mut usize,
) -> (Fetch, String) {
    let n = s.servers.len();
    let mut reachable = false;
    for k in 0..n {
        let i = (*idx + k) % n;
        let srv = &s.servers[i];
        match fetch_one(client, srv, &s.clientid) {
            Ok(Fetch::Work(wu)) => {
                *idx = i;
                return (Fetch::Work(wu), srv.clone());
            }
            Ok(Fetch::Done) => return (Fetch::Done, srv.clone()),
            Ok(_) => reachable = true, // Empty: reachable, just no work
            Err(e) => eprintln!("# server {srv} unreachable: {e}; trying next"),
        }
    }
    (
        if reachable {
            Fetch::Empty
        } else {
            Fetch::Unreachable
        },
        String::new(),
    )
}

fn fetch_one(client: &reqwest::blocking::Client, server: &str, clientid: &str) -> Result<Fetch> {
    let url = format!("{}/workunit", server.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .form(&[("clientid", clientid)])
        .send()
        .context("requesting workunit")?;
    match resp.status().as_u16() {
        200 => {
            let body = resp.text().context("reading workunit body")?;
            let wu: Workunit = serde_json::from_str(&body)
                .with_context(|| format!("parsing workunit json: {body}"))?;
            eprintln!("# got workunit {} from {server}", wu.id);
            Ok(Fetch::Work(wu))
        }
        404 => Ok(Fetch::Empty),
        410 => Ok(Fetch::Done),
        other => bail!("unexpected status {other} from {url}"),
    }
}

// fileinfo entry for an uploaded file: {WUid, key, [command]}. `command` is the
// command index a STDOUT<n>/STDERR<n>/RESULT<n> file came from (the trailing
// digits of the file id); the server stores it and the driver reads stdio by it.
fn fileinfo_entry(wuid: &str, key: &str) -> serde_json::Value {
    let digits: String = key
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let mut v = serde_json::json!({"WUid": wuid, "key": key});
    if !digits.is_empty() {
        if let Ok(cmd) = digits.chars().rev().collect::<String>().parse::<i64>() {
            v["command"] = serde_json::json!(cmd);
        }
    }
    v
}

fn process_wu(
    client: &reqwest::blocking::Client,
    s: &Settings,
    server: &str,
    wu: &Workunit,
) -> Result<()> {
    download_files(client, s, server, wu)?;
    let (errorcode, failedcommand, stdio) = run_commands(s, wu)?;
    upload(client, s, server, wu, errorcode, failedcommand, stdio)
}

fn subst_arch(name: &str, arch: &str) -> String {
    substitute(
        name,
        &HashMap::from([("ARCH".to_string(), arch.to_string())]),
    )
}

fn download_files(
    client: &reqwest::blocking::Client,
    s: &Settings,
    server: &str,
    wu: &Workunit,
) -> Result<()> {
    for (fid, f) in &wu.files {
        if !f.download {
            continue;
        }
        let urlname = subst_arch(&f.filename, &s.arch);
        let dlname = subst_arch(&f.filename, "");
        let dlpath = s.dldir.join(&dlname);
        let url = format!("{}/file/{}", server.trim_end_matches('/'), urlname);

        let resp = client
            .get(&url)
            .send()
            .with_context(|| format!("GET {url}"))?;
        if !resp.status().is_success() {
            bail!("download {url} -> status {}", resp.status());
        }
        let bytes = resp.bytes().context("reading file body")?;
        if let (Some(want), Some(algo)) = (&f.checksum, &f.algorithm) {
            let got = checksum(&bytes, algo)?;
            if !got.eq_ignore_ascii_case(want) {
                bail!("checksum mismatch for {dlname}: want {want}, got {got}");
            }
        }
        if let Some(parent) = dlpath.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        write_locked(&dlpath, &bytes).with_context(|| format!("writing {dlpath:?}"))?;
        if fid.starts_with("EXECFILE") {
            make_executable(&dlpath);
        }
    }
    Ok(())
}

// Write under an advisory exclusive lock so two clients sharing a dldir don't
// clobber each other's partial downloads.
fn write_locked(path: &Path, data: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    flock(&f, true);
    f.write_all(data)?;
    f.flush()?;
    flock(&f, false);
    Ok(())
}

#[cfg(unix)]
fn flock(f: &std::fs::File, lock: bool) {
    use std::os::unix::io::AsRawFd;
    let op = if lock { libc::LOCK_EX } else { libc::LOCK_UN };
    unsafe {
        libc::flock(f.as_raw_fd(), op);
    }
}
#[cfg(not(unix))]
fn flock(_f: &std::fs::File, _lock: bool) {}

fn checksum(data: &[u8], algo: &str) -> Result<String> {
    Ok(match algo {
        "sha1" => hex(&sha1::Sha1::digest(data)),
        "sha256" => hex(&sha2::Sha256::digest(data)),
        "sha3_256" => hex(&sha3::Sha3_256::digest(data)),
        other => bail!("unknown checksum algorithm {other}"),
    })
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(unix)]
fn make_executable(p: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(meta) = std::fs::metadata(p) {
        let mut perm = meta.permissions();
        perm.set_mode(perm.mode() | 0o755);
        std::fs::set_permissions(p, perm).ok();
    }
}
#[cfg(not(unix))]
fn make_executable(_p: &Path) {}

fn file_map(s: &Settings, wu: &Workunit) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for (fid, f) in &wu.files {
        let name = subst_arch(&f.filename, "");
        let path: PathBuf = if fid.starts_with("FILE") || fid.starts_with("EXECFILE") {
            s.dldir.join(&name)
        } else if fid.starts_with("RESULT")
            || fid.starts_with("WDIR")
            || fid.starts_with("STDOUT")
            || fid.starts_with("STDERR")
            || fid.starts_with("STDIN")
        {
            s.workdir.join(&name)
        } else {
            PathBuf::from(&name)
        };
        m.insert(fid.clone(), path.to_string_lossy().into_owned());
    }
    m
}

// Template.safe_substitute: $ident and ${ident}; $$ -> $; unknown left intact.
fn substitute(input: &str, map: &HashMap<String, String>) -> String {
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'$' {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }
        if i + 1 < bytes.len() && bytes[i + 1] == b'$' {
            out.push('$');
            i += 2;
            continue;
        }
        let (name, next) = if i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            match input[i + 2..].find('}') {
                Some(rel) => (input[i + 2..i + 2 + rel].to_string(), i + 2 + rel + 1),
                None => {
                    out.push('$');
                    i += 1;
                    continue;
                }
            }
        } else {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len()
                && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_')
                && !(j == start && bytes[j].is_ascii_digit())
            {
                j += 1;
            }
            if j == start {
                out.push('$');
                i += 1;
                continue;
            }
            (input[start..j].to_string(), j)
        };
        match map.get(&name) {
            Some(v) => out.push_str(v),
            None => {
                out.push('$');
                out.push_str(&name);
            }
        }
        i = next;
    }
    out
}

type Stdio = HashMap<String, Vec<u8>>;

fn run_commands(s: &Settings, wu: &Workunit) -> Result<(Option<i32>, Option<String>, Stdio)> {
    let map = file_map(s, wu);
    let mut stdio: Stdio = HashMap::new();
    for (counter, raw) in wu.commands.iter().enumerate() {
        let command = raw.replace('\'', "");
        let command = substitute(&command, &map);
        let argv: Vec<String> = command
            .split(' ')
            .filter(|a| !a.is_empty())
            .map(|a| a.to_string())
            .collect();
        if argv.is_empty() {
            continue;
        }
        eprintln!("# running: {command}");
        let mut cmd = std::process::Command::new(&argv[0]);
        cmd.args(&argv[1..]);
        set_niceness(&mut cmd, s.niceness);
        // STDIN<n> redirection: if the WU declares a STDIN file for this command,
        // feed it on the child's stdin.
        if let Some(path) = map.get(&format!("STDIN{counter}")) {
            match std::fs::File::open(path) {
                Ok(f) => {
                    cmd.stdin(std::process::Stdio::from(f));
                }
                Err(e) => eprintln!("# warning: STDIN{counter} {path}: {e}"),
            }
        }
        let out = cmd
            .output()
            .with_context(|| format!("spawning {}", argv[0]))?;

        let so_key = format!("STDOUT{counter}");
        if let Some(path) = map.get(&so_key) {
            std::fs::write(path, &out.stdout).ok();
        } else if !out.stdout.is_empty() {
            stdio.insert(so_key, out.stdout);
        }
        let se_key = format!("STDERR{counter}");
        if let Some(path) = map.get(&se_key) {
            std::fs::write(path, &out.stderr).ok();
        } else if !out.stderr.is_empty() {
            stdio.insert(se_key, out.stderr);
        }

        let code = out.status.code().unwrap_or(-1);
        if code != 0 {
            eprintln!("# command {counter} exited with {code}");
            return Ok((Some(code), Some(command), stdio));
        }
    }
    Ok((None, None, stdio))
}

#[cfg(unix)]
fn set_niceness(cmd: &mut std::process::Command, niceness: i32) {
    if niceness <= 0 {
        return;
    }
    use std::os::unix::process::CommandExt;
    unsafe {
        cmd.pre_exec(move || {
            libc::setpriority(libc::PRIO_PROCESS, 0, niceness);
            Ok(())
        });
    }
}
#[cfg(not(unix))]
fn set_niceness(_cmd: &mut std::process::Command, _niceness: i32) {}

fn upload(
    client: &reqwest::blocking::Client,
    s: &Settings,
    server: &str,
    wu: &Workunit,
    errorcode: Option<i32>,
    failedcommand: Option<String>,
    stdio: Stdio,
) -> Result<()> {
    use reqwest::blocking::multipart::{Form, Part};
    let mut form = Form::new()
        .text("clientid", s.clientid.clone())
        .text("WUid", wu.id.clone());
    if let Some(c) = errorcode {
        form = form.text("errorcode", c.to_string());
    }
    if let Some(fc) = failedcommand {
        form = form.text("failedcommand", fc);
    }
    let mut fileinfo = serde_json::Map::new();

    for (fid, f) in &wu.files {
        if !f.upload {
            continue;
        }
        let name = subst_arch(&f.filename, "");
        let path = s.workdir.join(&name);
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("# warning: declared upload file missing: {path:?}");
                continue;
            }
        };
        fileinfo.insert(name.clone(), fileinfo_entry(&wu.id, fid));
        form = form.part(name.clone(), Part::bytes(data).file_name(name));
    }
    for (key, blob) in stdio {
        let basename = format!("{}.{}", wu.id, key);
        fileinfo.insert(basename.clone(), fileinfo_entry(&wu.id, &key));
        form = form.part(basename.clone(), Part::bytes(blob).file_name(basename));
    }
    form = form.text("fileinfo", serde_json::Value::Object(fileinfo).to_string());

    let url = format!("{}/upload", server.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .multipart(form)
        .send()
        .context("POST /upload")?;
    if !resp.status().is_success() {
        bail!("upload -> status {}", resp.status());
    }
    eprintln!("# uploaded results for {} to {server}", wu.id);
    Ok(())
}

/// clap front-end (Track 3.4: real --help / validation). Same flag names and
/// semantics as before; the TLS flags still set the CADO_NFS_* env vars the rest
/// of the client reads, and `--server` may be repeated for failover.
#[derive(clap::Parser)]
#[command(
    name = "cado-nfs-client-rs",
    version,
    about = "Static-binary work-unit client for CADO-NFS \
                   (speaks the stock Python api_server protocol)"
)]
struct Args {
    /// server URL; repeat for failover (downloads/upload stick to the issuer)
    #[arg(long = "server", required = true)]
    server: Vec<String>,
    /// client identifier (default: <user>@<host>.<pid>)
    #[arg(long)]
    clientid: Option<String>,
    /// download directory (default: <tmp>/cado-client-dl)
    #[arg(long)]
    dldir: Option<PathBuf>,
    /// work directory (default: <tmp>/cado-client-work)
    #[arg(long)]
    workdir: Option<PathBuf>,
    /// architecture string advertised to the server
    #[arg(long, default_value = "")]
    arch: String,
    /// seconds to wait/retry on a failed download
    #[arg(long = "downloadretry", default_value_t = 10)]
    download_retry: u64,
    /// renice spawned children by this amount
    #[arg(long, default_value_t = 0)]
    niceness: i32,
    /// process a single work-unit then exit
    #[arg(long)]
    single: bool,
    /// skip TLS certificate verification (sets CADO_NFS_INSECURE)
    #[arg(long)]
    insecure: bool,
    /// CA certificate PEM to trust (sets CADO_NFS_CAFILE)
    #[arg(long)]
    cafile: Option<String>,
    /// pin the server cert by sha1 fingerprint (sets CADO_NFS_CERTSHA1)
    #[arg(long)]
    certsha1: Option<String>,
    /// print a shell completion script (bash|zsh|fish|...) to stdout and exit
    #[arg(long, value_enum, exclusive = true)]
    completions: Option<clap_complete::Shell>,
}

/// `--completions <shell>` (Roadmap E6): emit the script to stdout and exit,
/// before clap's required-argument check runs (so it works standalone). Handles
/// both `--completions bash` and `--completions=bash`.
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
                    "cado-nfs-client-rs",
                    &mut std::io::stdout(),
                );
                std::process::exit(0);
            }
        }
        return; // bad/missing value: let clap report it
    }
}

fn parse_args() -> Result<Settings> {
    use clap::Parser;
    maybe_emit_completions();
    let a = Args::parse();
    // TLS options are passed to the rest of the client via env vars, exactly as
    // the previous hand-rolled parser did.
    if a.insecure {
        std::env::set_var("CADO_NFS_INSECURE", "1");
    }
    if let Some(v) = a.cafile {
        std::env::set_var("CADO_NFS_CAFILE", v);
    }
    if let Some(v) = a.certsha1 {
        std::env::set_var("CADO_NFS_CERTSHA1", v);
    }
    Ok(Settings {
        servers: a.server,
        clientid: a.clientid.unwrap_or_else(default_clientid),
        dldir: a
            .dldir
            .unwrap_or_else(|| std::env::temp_dir().join("cado-client-dl")),
        workdir: a
            .workdir
            .unwrap_or_else(|| std::env::temp_dir().join("cado-client-work")),
        arch: a.arch,
        download_retry: a.download_retry,
        single: a.single,
        niceness: a.niceness,
    })
}

fn default_clientid() -> String {
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "host".to_string());
    format!("{host}-rs-{}", std::process::id())
}

// --certsha1 cert pinning: a rustls verifier that accepts the TLS handshake iff
// the server's end-entity certificate hashes (sha1) to the pinned value.
mod certpin {
    use sha1::Digest;

    #[derive(Debug)]
    pub struct PinnedVerifier {
        pub want: String, // lowercase hex sha1
    }

    impl rustls::client::danger::ServerCertVerifier for PinnedVerifier {
        fn verify_server_cert(
            &self,
            end_entity: &rustls::pki_types::CertificateDer<'_>,
            _intermediates: &[rustls::pki_types::CertificateDer<'_>],
            _name: &rustls::pki_types::ServerName<'_>,
            _ocsp: &[u8],
            _now: rustls::pki_types::UnixTime,
        ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
            let got = super::hex(&sha1::Sha1::digest(end_entity.as_ref()));
            if got == self.want {
                Ok(rustls::client::danger::ServerCertVerified::assertion())
            } else {
                Err(rustls::Error::General(format!(
                    "pinned cert sha1 mismatch: got {got}, want {}",
                    self.want
                )))
            }
        }
        // the fingerprint pin is the trust decision; accept the cert's own signatures
        fn verify_tls12_signature(
            &self,
            _m: &[u8],
            _c: &rustls::pki_types::CertificateDer<'_>,
            _d: &rustls::DigitallySignedStruct,
        ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
            Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
        }
        fn verify_tls13_signature(
            &self,
            _m: &[u8],
            _c: &rustls::pki_types::CertificateDer<'_>,
            _d: &rustls::DigitallySignedStruct,
        ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
            Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
        }
        fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
            rustls::crypto::ring::default_provider()
                .signature_verification_algorithms
                .supported_schemes()
        }
    }
}
