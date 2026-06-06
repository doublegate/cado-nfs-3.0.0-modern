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
        for k in ["computation", "input_digits", "wu_done", "wu_total", "updated"] {
            if let Some(val) = jstr(v, k) {
                s.rows.push((k.replace('_', " "), val));
            }
        }
        if let Some(Value::Array(fs)) = v.get("factors") {
            s.factors = fs.iter().filter_map(|f| f.as_str().map(String::from)).collect();
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

fn draw(f: &mut Frame, url: &str, s: &Snapshot) {
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
    .block(Block::default().borders(Borders::ALL).title(" cado-nfs monitor "));
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
        rows.push(Row::new(vec!["ETA".to_string(), eta.clone()]));
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
        s.factors.iter().map(|x| Line::from(Span::raw(x.clone()))).collect()
    };
    let factors = Paragraph::new(ftext)
        .block(Block::default().borders(Borders::ALL).title(" factors "));
    f.render_widget(factors, chunks[3]);

    let footer = Paragraph::new(Line::from(Span::styled(
        format!(" {url}    q/Esc to quit "),
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(footer, chunks[4]);
}

fn main() -> Result<()> {
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
            s.percent.map(|p| format!("{p:.1}%")).unwrap_or_else(|| "n/a".into())
        );
        for (k, v) in &s.rows {
            println!("{k}: {v}");
        }
        if let Some(eta) = &s.eta {
            println!("ETA: {eta}");
        }
        if !s.factors.is_empty() {
            println!("factors: {}", s.factors.join(" "));
        }
        return Ok(());
    }

    let mut terminal = ratatui::init();
    let mut snap = fetch(&client, &url);
    let mut last = Instant::now();
    let res = (|| -> Result<()> {
        loop {
            terminal.draw(|f| draw(f, &url, &snap))?;
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
                last = Instant::now();
            }
        }
        Ok(())
    })();
    ratatui::restore();
    res
}
