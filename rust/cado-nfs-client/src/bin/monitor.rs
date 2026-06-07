// cado-nfs-monitor-rs -- a small terminal dashboard for a CADO-NFS run (Track
// 3.4, optional). It polls a server's /status endpoint and renders a live view:
// progress gauge, phase/state, work-unit counts, ETA, and discovered factors.
//
// It understands both status schemas this fork serves:
//   - cado-nfs-status/1     (Flask api_server: phase/percent/ETA/factors)
//   - cado-nfs-wu-status/1  (cado-wu-server-rs: work-unit counts + percent)
//
//   cado-nfs-monitor-rs --server http://127.0.0.1:4242 [--interval 2] [--insecure]
//
// Keys: q / Esc / Ctrl-C to quit. Reuses the client crate's blocking reqwest.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::{
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Row, Table},
    Frame,
};
use serde_json::Value;

#[derive(Parser)]
#[command(
    name = "cado-nfs-monitor-rs",
    version,
    about = "Live terminal monitor for a CADO-NFS run (polls a server's /status)"
)]
struct Args {
    /// server base URL (e.g. http://127.0.0.1:4242)
    #[arg(long)]
    server: String,
    /// seconds between polls
    #[arg(long, default_value_t = 2.0)]
    interval: f64,
    /// skip TLS certificate verification (for self-signed https servers)
    #[arg(long)]
    insecure: bool,
    /// fetch /status once, print a plain-text summary, and exit (no TUI;
    /// scriptable and usable without a terminal)
    #[arg(long)]
    once: bool,
    /// print a shell completion script (bash|zsh|fish|...) to stdout and exit
    #[arg(long, value_enum, exclusive = true)]
    completions: Option<clap_complete::Shell>,
}

/// What we managed to read from /status, normalised across the two schemas.
#[derive(Default)]
struct Snapshot {
    title: String,
    state: String,
    phase: String,
    percent: Option<f64>,
    eta: Option<String>,
    rows: Vec<(String, String)>,
    factors: Vec<String>,
    error: Option<String>,
    /// numeric work-unit progress (for the throughput estimate), if exposed
    wu_done: Option<i64>,
    wu_total: Option<i64>,
}

fn jstr(v: &Value, k: &str) -> Option<String> {
    match v.get(k) {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Number(n)) => Some(n.to_string()),
        Some(Value::Bool(b)) => Some(b.to_string()),
        _ => None,
    }
}
fn jf64(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(|x| x.as_f64())
}
fn ji64(v: &Value, k: &str) -> Option<i64> {
    v.get(k).and_then(|x| x.as_i64())
}

fn parse_snapshot(v: &Value) -> Snapshot {
    let mut s = Snapshot::default();
    let schema = jstr(v, "schema").unwrap_or_default();
    if schema.starts_with("cado-nfs-wu-status") {
        // Rust work-unit server: counts + percent + serving flag.
        s.title = jstr(v, "server").unwrap_or_else(|| "cado-wu-server".into());
        s.state = if v.get("serving").and_then(|x| x.as_bool()).unwrap_or(false) {
            "serving".into()
        } else {
            "finished".into()
        };
        s.phase = "work-unit distribution".into();
        s.percent = jf64(v, "percent");
        if let Some(wu) = v.get("workunits") {
            for k in ["total", "available", "assigned", "ok", "error", "done"] {
                if let Some(n) = ji64(wu, k) {
                    s.rows.push((k.to_string(), n.to_string()));
                }
            }
            s.wu_done = ji64(wu, "done");
            s.wu_total = ji64(wu, "total");
        }
    } else {
        // Flask driver status: phase / percent / ETA / factors.
        s.title = jstr(v, "name").unwrap_or_else(|| "cado-nfs".into());
        s.state = jstr(v, "state").unwrap_or_default();
        let mut phase = jstr(v, "phase").unwrap_or_default();
        if let (Some(i), Some(t)) = (ji64(v, "phase_index"), ji64(v, "phase_total")) {
            phase = format!("[{i}/{t}] {phase}");
        }
        s.phase = phase;
        s.percent = jf64(v, "phase_percent");
        s.eta = jstr(v, "eta").filter(|e| e != "Unknown");
        s.wu_done = ji64(v, "wu_done");
        s.wu_total = ji64(v, "wu_total");
        for k in [
            "computation",
            "input_digits",
            "wu_done",
            "wu_total",
            "updated",
        ] {
            if let Some(val) = jstr(v, k) {
                s.rows.push((k.replace('_', " "), val));
            }
        }
        if let Some(Value::Array(fs)) = v.get("factors") {
            s.factors = fs
                .iter()
                .filter_map(|f| f.as_str().map(String::from))
                .collect();
        }
    }
    s
}

fn fetch(client: &reqwest::blocking::Client, url: &str) -> Snapshot {
    match client.get(url).send().and_then(|r| r.error_for_status()) {
        Ok(resp) => match resp.json::<Value>() {
            Ok(v) => parse_snapshot(&v),
            Err(e) => Snapshot {
                error: Some(format!("bad JSON: {e}")),
                ..Default::default()
            },
        },
        Err(e) => Snapshot {
            error: Some(format!("{e}")),
            ..Default::default()
        },
    }
}

/// Derived live metrics the monitor computes itself from the poll time-series
/// and the local host (Roadmap E4): a trailing-window ETA + throughput that do
/// not depend on the server reporting them, plus local CPU/GPU utilisation.
#[derive(Default)]
struct Derived {
    pct_per_min: Option<f64>,
    eta_trend: Option<String>,
    wu_per_min: Option<f64>,
    cpu_pct: Option<f64>,
    gpu: Option<String>,
}

/// Human ETA from a minutes-remaining estimate.
fn fmt_eta_mins(mins: f64) -> String {
    if mins < 1.5 {
        format!("{:.0} s", mins * 60.0)
    } else if mins < 90.0 {
        format!("{mins:.1} min")
    } else if mins < 2880.0 {
        format!("{:.1} h", mins / 60.0)
    } else {
        format!("{:.1} days", mins / 1440.0)
    }
}

/// /proc/stat first line -> (idle_jiffies, total_jiffies); None off Linux.
fn read_cpu_sample() -> Option<(u64, u64)> {
    let stat = std::fs::read_to_string("/proc/stat").ok()?;
    let line = stat.lines().next()?;
    let nums: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|x| x.parse().ok())
        .collect();
    if nums.len() < 5 {
        return None;
    }
    let idle = nums[3] + nums[4]; // idle + iowait
    let total: u64 = nums.iter().sum();
    Some((idle, total))
}

/// Local GPU "util% mem/totalMB" via nvidia-smi (read-only); None if absent.
fn read_gpu() -> Option<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout);
    let first = line.lines().next()?;
    let f: Vec<&str> = first.split(',').map(|x| x.trim()).collect();
    if f.len() >= 3 {
        Some(format!("{}% util, {}/{} MiB", f[0], f[1], f[2]))
    } else {
        None
    }
}

/// Rolling poll history -> trailing-window rate/ETA/throughput + local telemetry.
struct Telemetry {
    pts: VecDeque<(Instant, f64, Option<i64>)>, // (t, percent, wu_done)
    prev_cpu: Option<(u64, u64)>,
    window: Duration,
}

impl Telemetry {
    fn new() -> Self {
        Telemetry {
            pts: VecDeque::new(),
            prev_cpu: None,
            window: Duration::from_secs(120),
        }
    }

    /// Record one poll and recompute the derived metrics (call once per poll so
    /// the CPU-utilisation delta is taken over the poll interval).
    fn update(&mut self, snap: &Snapshot) -> Derived {
        let now = Instant::now();
        if let Some(p) = snap.percent {
            self.pts.push_back((now, p, snap.wu_done));
            while self.pts.len() > 2 {
                let old = self.pts.front().map(|&(t, _, _)| t).unwrap();
                if now.duration_since(old) > self.window {
                    self.pts.pop_front();
                } else {
                    break;
                }
            }
        }
        let mut d = Derived::default();
        if self.pts.len() >= 2 {
            let (t0, p0, w0) = *self.pts.front().unwrap();
            let (t1, p1, w1) = *self.pts.back().unwrap();
            let dt_min = t1.duration_since(t0).as_secs_f64() / 60.0;
            if dt_min > 1e-6 {
                let rate = (p1 - p0) / dt_min;
                if rate > 1e-6 {
                    d.pct_per_min = Some(rate);
                    d.eta_trend = Some(fmt_eta_mins((100.0 - p1).max(0.0) / rate));
                }
                if let (Some(a), Some(b)) = (w0, w1) {
                    let dwu = (b - a) as f64;
                    if dwu >= 0.0 {
                        d.wu_per_min = Some(dwu / dt_min);
                    }
                }
            }
        }
        if let Some((idle, total)) = read_cpu_sample() {
            if let Some((pi, pt)) = self.prev_cpu {
                let di = idle.saturating_sub(pi) as f64;
                let dtot = total.saturating_sub(pt) as f64;
                if dtot > 0.0 {
                    d.cpu_pct = Some(((1.0 - di / dtot) * 100.0).clamp(0.0, 100.0));
                }
            }
            self.prev_cpu = Some((idle, total));
        }
        d.gpu = read_gpu();
        d
    }
}

fn draw(f: &mut Frame, url: &str, s: &Snapshot, d: &Derived) {
    let chunks = Layout::vertical([
        Constraint::Length(3), // header
        Constraint::Length(3), // gauge
        Constraint::Min(3),    // table
        Constraint::Length(5), // factors
        Constraint::Length(1), // footer
    ])
    .split(f.area());

    let state_color = match s.state.as_str() {
        "done" | "serving" => Color::Green,
        "error" => Color::Red,
        _ => Color::Cyan,
    };
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            format!(" {} ", s.title),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(s.state.clone(), Style::default().fg(state_color)),
        Span::raw("   "),
        Span::raw(s.phase.clone()),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" cado-nfs monitor "),
    );
    f.render_widget(header, chunks[0]);

    let pct = s.percent.unwrap_or(0.0).clamp(0.0, 100.0);
    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" progress "))
        .gauge_style(Style::default().fg(Color::Blue))
        .percent(pct as u16)
        .label(match s.percent {
            Some(p) => format!("{p:.1}%"),
            None => "n/a".into(),
        });
    f.render_widget(gauge, chunks[1]);

    let mut rows: Vec<Row> = s
        .rows
        .iter()
        .map(|(k, v)| Row::new(vec![k.clone(), v.clone()]))
        .collect();
    if let Some(eta) = &s.eta {
        rows.push(Row::new(vec!["ETA (server)".to_string(), eta.clone()]));
    }
    // Derived live metrics computed by the monitor itself (Roadmap E4).
    if let Some(eta) = &d.eta_trend {
        let rate = d
            .pct_per_min
            .map(|r| format!("  ({r:.2} %/min)"))
            .unwrap_or_default();
        rows.push(Row::new(vec![
            "ETA (trend)".to_string(),
            format!("{eta}{rate}"),
        ]));
    }
    if let Some(w) = d.wu_per_min {
        rows.push(Row::new(vec![
            "throughput".to_string(),
            format!("{w:.1} work-units/min"),
        ]));
    }
    if let Some(c) = d.cpu_pct {
        rows.push(Row::new(vec![
            "host CPU".to_string(),
            format!("{c:.0}% busy (local)"),
        ]));
    }
    if let Some(g) = &d.gpu {
        rows.push(Row::new(vec![
            "host GPU".to_string(),
            format!("{g} (local)"),
        ]));
    }
    let table = Table::new(rows, [Constraint::Length(16), Constraint::Min(10)])
        .block(Block::default().borders(Borders::ALL).title(" status "));
    f.render_widget(table, chunks[2]);

    let ftext = if let Some(err) = &s.error {
        vec![Line::from(Span::styled(
            format!("server unreachable: {err}"),
            Style::default().fg(Color::Red),
        ))]
    } else if s.factors.is_empty() {
        vec![Line::from(Span::raw("(none yet)"))]
    } else {
        s.factors
            .iter()
            .map(|x| Line::from(Span::raw(x.clone())))
            .collect()
    };
    let factors =
        Paragraph::new(ftext).block(Block::default().borders(Borders::ALL).title(" factors "));
    f.render_widget(factors, chunks[3]);

    let footer = Paragraph::new(Line::from(Span::styled(
        format!(" {url}    q/Esc to quit "),
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(footer, chunks[4]);
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
                    "cado-nfs-monitor-rs",
                    &mut std::io::stdout(),
                );
                std::process::exit(0);
            }
        }
        return;
    }
}

fn main() -> Result<()> {
    maybe_emit_completions();
    let args = Args::parse();
    let url = format!("{}/status", args.server.trim_end_matches('/'));
    let client = reqwest::blocking::Client::builder()
        .danger_accept_invalid_certs(args.insecure)
        .timeout(Duration::from_secs(10))
        .build()
        .context("building http client")?;
    let period = Duration::from_secs_f64(args.interval.max(0.2));

    if args.once {
        let s = fetch(&client, &url);
        if let Some(err) = &s.error {
            eprintln!("server unreachable: {err}");
            std::process::exit(1);
        }
        println!("title:   {}", s.title);
        println!("state:   {}", s.state);
        println!("phase:   {}", s.phase);
        println!(
            "percent: {}",
            s.percent
                .map(|p| format!("{p:.1}%"))
                .unwrap_or_else(|| "n/a".into())
        );
        for (k, v) in &s.rows {
            println!("{k}: {v}");
        }
        if let Some(eta) = &s.eta {
            println!("ETA: {eta}");
        }
        // local host telemetry (a short CPU delta + a GPU sample), Roadmap E4.
        if let Some((i0, t0)) = read_cpu_sample() {
            std::thread::sleep(Duration::from_millis(250));
            if let Some((i1, t1)) = read_cpu_sample() {
                let dt = t1.saturating_sub(t0) as f64;
                if dt > 0.0 {
                    let busy = (1.0 - (i1.saturating_sub(i0) as f64) / dt) * 100.0;
                    println!("host CPU: {:.0}% busy (local)", busy.clamp(0.0, 100.0));
                }
            }
        }
        if let Some(g) = read_gpu() {
            println!("host GPU: {g} (local)");
        }
        if !s.factors.is_empty() {
            println!("factors: {}", s.factors.join(" "));
        }
        return Ok(());
    }

    let mut terminal = ratatui::init();
    let mut tele = Telemetry::new();
    let mut snap = fetch(&client, &url);
    let mut der = tele.update(&snap);
    let mut last = Instant::now();
    let res = (|| -> Result<()> {
        loop {
            terminal.draw(|f| draw(f, &url, &snap, &der))?;
            // wait for a key up to the poll period, then refresh.
            let wait = period.saturating_sub(last.elapsed());
            if event::poll(wait.max(Duration::from_millis(50)))? {
                if let Event::Key(k) = event::read()? {
                    let quit = matches!(k.code, KeyCode::Char('q') | KeyCode::Esc)
                        || (k.code == KeyCode::Char('c')
                            && k.modifiers.contains(KeyModifiers::CONTROL));
                    if quit {
                        break;
                    }
                }
            }
            if last.elapsed() >= period {
                snap = fetch(&client, &url);
                der = tele.update(&snap);
                last = Instant::now();
            }
        }
        Ok(())
    })();
    ratatui::restore();
    res
}
