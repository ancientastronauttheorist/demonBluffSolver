use solver_core::solver::solve;
use solver_core::types::GameState;
use std::io::{self, BufRead, Read, Write};

fn main() {
    let daemon_mode = std::env::args().any(|a| a == "--daemon");
    if daemon_mode {
        run_daemon();
    } else {
        run_oneshot();
    }
}

/// Original one-shot mode: read all stdin, solve once, exit.
fn run_oneshot() {
    let mut input = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut input) {
        let err = serde_json::json!({"error": format!("stdin read error: {e}")});
        println!("{err}");
        std::process::exit(1);
    }

    let value: serde_json::Value = match serde_json::from_str(&input) {
        Ok(v) => v,
        Err(e) => {
            let err = serde_json::json!({"error": format!("JSON parse error: {e}")});
            println!("{err}");
            std::process::exit(1);
        }
    };

    let state = match GameState::from_json(&value) {
        Ok(s) => s,
        Err(e) => {
            let err = serde_json::json!({"error": format!("GameState parse error: {e}")});
            println!("{err}");
            std::process::exit(1);
        }
    };

    // Catch panics from solve()
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| solve(&state)));

    match result {
        Ok(solver_result) => {
            // Check --summary flag
            let summary_mode = std::env::args().any(|a| a == "--summary");
            if summary_mode {
                let out = serde_json::json!({
                    "definite_evil": solver_result.definite_evil,
                    "definite_good": solver_result.definite_good,
                    "bombardier_positions": solver_result.bombardier_positions,
                    "n_scenarios": solver_result.n_scenarios,
                    "n_surviving": solver_result.n_surviving,
                });
                println!("{out}");
            } else {
                match serde_json::to_string(&solver_result) {
                    Ok(json) => println!("{json}"),
                    Err(e) => {
                        let err = serde_json::json!({"error": format!("serialization error: {e}")});
                        println!("{err}");
                        std::process::exit(2);
                    }
                }
            }
        }
        Err(panic_info) => {
            let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "unknown panic".to_string()
            };
            let err = serde_json::json!({"error": format!("solver panic: {msg}")});
            println!("{err}");
            std::process::exit(2);
        }
    }
}

/// Daemon mode: read JSON lines from stdin, solve each, write compact JSON response line.
/// Runs until stdin is closed (EOF). Panics are caught per-request.
fn run_daemon() {
    let stdin = io::stdin();
    let reader = io::BufReader::new(stdin.lock());
    let mut stdout = io::stdout().lock();

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => break, // EOF or read error — exit cleanly
        };

        // Skip blank lines
        if line.trim().is_empty() {
            continue;
        }

        // Parse JSON
        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let err = serde_json::json!({"error": format!("JSON parse error: {e}")});
                let _ = writeln!(stdout, "{err}");
                let _ = stdout.flush();
                continue;
            }
        };

        // Check per-request __summary field
        let summary_mode = value.get("__summary").and_then(|v| v.as_bool()).unwrap_or(false);

        // Parse GameState
        let state = match GameState::from_json(&value) {
            Ok(s) => s,
            Err(e) => {
                let err = serde_json::json!({"error": format!("GameState parse error: {e}")});
                let _ = writeln!(stdout, "{err}");
                let _ = stdout.flush();
                continue;
            }
        };

        // Solve with panic guard
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| solve(&state)));

        match result {
            Ok(solver_result) => {
                if summary_mode {
                    let out = serde_json::json!({
                        "definite_evil": solver_result.definite_evil,
                        "definite_good": solver_result.definite_good,
                        "bombardier_positions": solver_result.bombardier_positions,
                        "n_scenarios": solver_result.n_scenarios,
                        "n_surviving": solver_result.n_surviving,
                    });
                    let _ = writeln!(stdout, "{out}");
                } else {
                    match serde_json::to_string(&solver_result) {
                        Ok(json) => {
                            let _ = writeln!(stdout, "{json}");
                        }
                        Err(e) => {
                            let err = serde_json::json!({"error": format!("serialization error: {e}")});
                            let _ = writeln!(stdout, "{err}");
                        }
                    }
                }
            }
            Err(panic_info) => {
                let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "unknown panic".to_string()
                };
                let err = serde_json::json!({"error": format!("solver panic: {msg}")});
                let _ = writeln!(stdout, "{err}");
            }
        }

        // Always flush after each response (critical for Windows pipe buffering)
        let _ = stdout.flush();
    }

    // EOF reached — exit cleanly
}
