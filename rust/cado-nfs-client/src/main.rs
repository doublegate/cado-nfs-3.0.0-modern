// cado-nfs-client-rs -- a static-binary work-unit client for CADO-NFS.
//
// It speaks the exact HTTP/JSON protocol of the stock Python server
// (scripts/cadofactor/api_server.py + workunit.py), so it interoperates with an
// unmodified cado-nfs.py run:
//
//   GET  /workunit  (form body `clientid=...`)  -> 200 WU JSON | 404 wait | 410 done
//   GET  /file/<name>                            -> input file (sha1/256/3_256 checked)
//   POST /upload    (multipart: clientid, WUid, fileinfo JSON, result files)
//
// The work loop: fetch a WU, download its `download` files (checksum-verified),
// substitute `$FID`/`${FID}` placeholders into each command (file ids are
// dir-mapped by prefix: FILE->dldir, WDIR/RESULT/STDOUT/...->workdir,
// EXECFILE->downloaded binary), run them (argv split on spaces, no shell, as the
// Python client does), capturing stdout/stderr to STDOUT%d/STDERR%d files or for
// upload, then POST the `upload` files + captured stdio back.
//
// Scope vs the Python client: single server, core loop. Deferred (documented in
// docs/rust-orchestration.md): multi-server failover, automatic certificate
// download, file locking/backlog. TLS: pass --insecure for the self-signed dev
// cert, or --cafile <pem>.

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
    server: String,
    clientid: String,
    dldir: PathBuf,
    workdir: PathBuf,
    arch: String,
    download_retry: u64,
    single: bool,
}

enum Fetch {
    Work(Workunit),
    Empty, // 404: no work yet, retry later
    Done,  // 410: computation finished
}

fn main() -> Result<()> {
    let s = parse_args()?;
    std::fs::create_dir_all(&s.dldir).ok();
    std::fs::create_dir_all(&s.workdir).ok();

    let client = build_http_client()?;
    eprintln!(
        "# cado-nfs-client-rs: server={} clientid={}",
        s.server, s.clientid
    );

    loop {
        match fetch_wu(&client, &s)? {
            Fetch::Done => {
                eprintln!("# server says the computation is finished; exiting");
                break;
            }
            Fetch::Empty => {
                if s.single {
                    eprintln!("# no work available (404); --single, exiting");
                    break;
                }
                std::thread::sleep(Duration::from_secs(s.download_retry));
            }
            Fetch::Work(wu) => {
                let id = wu.id.clone();
                match process_wu(&client, &s, &wu) {
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
    let insecure = std::env::var("CADO_NFS_INSECURE").is_ok();
    let mut b = reqwest::blocking::Client::builder().timeout(Duration::from_secs(600));
    if insecure {
        b = b.danger_accept_invalid_certs(true).danger_accept_invalid_hostnames(true);
    }
    if let Ok(ca) = std::env::var("CADO_NFS_CAFILE") {
        let pem = std::fs::read(&ca).with_context(|| format!("reading cafile {ca}"))?;
        let cert = reqwest::Certificate::from_pem(&pem).context("parsing cafile")?;
        b = b.add_root_certificate(cert);
    }
    b.build().context("building HTTP client")
}

// GET /workunit with a form-urlencoded body `clientid=...` (matches the Python
// client's `requests.get(url, data={'clientid': ...})`, which Werkzeug parses
// into request.form on the server).
fn fetch_wu(client: &reqwest::blocking::Client, s: &Settings) -> Result<Fetch> {
    let url = format!("{}/workunit", s.server.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .form(&[("clientid", s.clientid.as_str())])
        .send()
        .context("requesting workunit")?;
    match resp.status().as_u16() {
        200 => {
            let body = resp.text().context("reading workunit body")?;
            let wu: Workunit = serde_json::from_str(&body)
                .with_context(|| format!("parsing workunit json: {body}"))?;
            eprintln!("# got workunit {}", wu.id);
            Ok(Fetch::Work(wu))
        }
        404 => Ok(Fetch::Empty),
        410 => Ok(Fetch::Done),
        other => bail!("unexpected status {other} from {url}"),
    }
}

fn process_wu(client: &reqwest::blocking::Client, s: &Settings, wu: &Workunit) -> Result<()> {
    download_files(client, s, wu)?;
    let (errorcode, failedcommand, stdio) = run_commands(s, wu)?;
    upload(client, s, wu, errorcode, failedcommand, stdio)
}

// Substitute $ARCH (Template-style) in a download filename.
fn subst_arch(name: &str, arch: &str) -> String {
    substitute(name, &HashMap::from([("ARCH".to_string(), arch.to_string())]))
}

fn download_files(client: &reqwest::blocking::Client, s: &Settings, wu: &Workunit) -> Result<()> {
    for (fid, f) in &wu.files {
        if !f.download {
            continue;
        }
        let urlname = subst_arch(&f.filename, &s.arch);
        let dlname = subst_arch(&f.filename, ""); // saved name uses blank ARCH
        let dlpath = s.dldir.join(&dlname);
        let url = format!("{}/file/{}", s.server.trim_end_matches('/'), urlname);

        let resp = client.get(&url).send().with_context(|| format!("GET {url}"))?;
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
        std::fs::write(&dlpath, &bytes).with_context(|| format!("writing {dlpath:?}"))?;
        if fid.starts_with("EXECFILE") {
            make_executable(&dlpath);
        }
    }
    Ok(())
}

fn checksum(data: &[u8], algo: &str) -> Result<String> {
    let hex = match algo {
        "sha1" => {
            let mut h = sha1::Sha1::new();
            h.update(data);
            hex(&h.finalize())
        }
        "sha256" => {
            let mut h = sha2::Sha256::new();
            h.update(data);
            hex(&h.finalize())
        }
        "sha3_256" => {
            let mut h = sha3::Sha3_256::new();
            h.update(data);
            hex(&h.finalize())
        }
        other => bail!("unknown checksum algorithm {other}"),
    };
    Ok(hex)
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

// Map each file id to the local path used for command substitution, following
// the Python client's prefix->directory rules.
fn file_map(s: &Settings, wu: &Workunit) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for (fid, f) in &wu.files {
        let name = subst_arch(&f.filename, ""); // local names use blank ARCH
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

// Template.safe_substitute: replace $ident and ${ident} from the map; leave
// unknown placeholders and `$$` (-> literal `$`) intact.
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
        // at a '$'
        if i + 1 < bytes.len() && bytes[i + 1] == b'$' {
            out.push('$'); // $$ -> $
            i += 2;
            continue;
        }
        let (name, next) = if i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            // ${ident}
            match input[i + 2..].find('}') {
                Some(rel) => (input[i + 2..i + 2 + rel].to_string(), i + 2 + rel + 1),
                None => {
                    out.push('$');
                    i += 1;
                    continue;
                }
            }
        } else {
            // $ident (identifier = [A-Za-z_][A-Za-z0-9_]*)
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
                // safe_substitute leaves it as-is
                out.push('$');
                out.push_str(&name); // approximate; fine for our placeholders
            }
        }
        i = next;
    }
    out
}

type Stdio = HashMap<String, Vec<u8>>;

// Returns (errorcode, failedcommand, captured stdio blobs keyed by "STDOUT<n>"/"STDERR<n>").
fn run_commands(s: &Settings, wu: &Workunit) -> Result<(Option<i32>, Option<String>, Stdio)> {
    let map = file_map(s, wu);
    let mut stdio: Stdio = HashMap::new();
    for (counter, raw) in wu.commands.iter().enumerate() {
        let command = raw.replace('\'', ""); // bug 21827
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
        let out = std::process::Command::new(&argv[0])
            .args(&argv[1..])
            .output()
            .with_context(|| format!("spawning {}", argv[0]))?;

        // route stdout: to STDOUT<n> file if declared, else keep for upload
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

// POST /upload multipart: clientid, WUid, [errorcode], [failedcommand],
// fileinfo (JSON {basename: {WUid, key}}), plus the `upload` files and captured
// stdio blobs.
fn upload(
    client: &reqwest::blocking::Client,
    s: &Settings,
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

    // declared upload files (from workdir)
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
        let basename = name.clone();
        fileinfo.insert(
            basename.clone(),
            serde_json::json!({"WUid": wu.id, "key": fid}),
        );
        form = form.part(
            basename.clone(),
            Part::bytes(data).file_name(basename),
        );
    }

    // captured stdout/stderr blobs -> "<wuid>.STDOUT<n>" etc.
    for (key, blob) in stdio {
        let basename = format!("{}.{}", wu.id, key);
        fileinfo.insert(
            basename.clone(),
            serde_json::json!({"WUid": wu.id, "key": key}),
        );
        form = form.part(
            basename.clone(),
            Part::bytes(blob).file_name(basename),
        );
    }

    form = form.text("fileinfo", serde_json::Value::Object(fileinfo).to_string());

    let url = format!("{}/upload", s.server.trim_end_matches('/'));
    let resp = client.post(&url).multipart(form).send().context("POST /upload")?;
    if !resp.status().is_success() {
        bail!("upload -> status {}", resp.status());
    }
    eprintln!("# uploaded results for {}", wu.id);
    Ok(())
}

fn parse_args() -> Result<Settings> {
    let mut server = None;
    let mut clientid = None;
    let mut dldir = None;
    let mut workdir = None;
    let mut arch = String::new();
    let mut download_retry = 10u64;
    let mut single = false;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--server" => server = it.next(),
            "--clientid" => clientid = it.next(),
            "--dldir" => dldir = it.next().map(PathBuf::from),
            "--workdir" => workdir = it.next().map(PathBuf::from),
            "--arch" => arch = it.next().unwrap_or_default(),
            "--downloadretry" => {
                download_retry = it.next().and_then(|v| v.parse().ok()).unwrap_or(10)
            }
            "--single" => single = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: cado-nfs-client-rs --server URL [--clientid ID] \
                     [--dldir DIR] [--workdir DIR] [--arch S] [--downloadretry SECS] [--single]\n\
                     env: CADO_NFS_INSECURE=1 (accept self-signed TLS), CADO_NFS_CAFILE=<pem>"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument {other} (try --help)"),
        }
    }

    let server = server.context("--server URL is required")?;
    let clientid = clientid.unwrap_or_else(default_clientid);
    let dldir = dldir.unwrap_or_else(|| std::env::temp_dir().join("cado-client-dl"));
    let workdir = workdir.unwrap_or_else(|| std::env::temp_dir().join("cado-client-work"));
    Ok(Settings {
        server,
        clientid,
        dldir,
        workdir,
        arch,
        download_retry,
        single,
    })
}

fn default_clientid() -> String {
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "host".to_string());
    format!("{host}-rs-{}", std::process::id())
}
