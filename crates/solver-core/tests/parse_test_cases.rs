//! Integration test: verify we can parse every v2 test case JSON file.

use solver_core::types::TestCase;
use std::path::PathBuf;

fn test_cases_dir() -> PathBuf {
    // Navigate from crates/solver-core/ up to repo root, then into tests/cases_v2/
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.join("../../tests/cases_v2")
}

fn legacy_cases_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.join("../../tests/cases")
}

#[test]
fn parse_all_v2_test_cases() {
    let dir = test_cases_dir();
    assert!(dir.exists(), "v2 test cases dir not found: {:?}", dir);

    let mut count = 0;
    let mut errors = Vec::new();

    for entry in glob::glob(dir.join("*.json").to_str().unwrap()).unwrap() {
        let path = entry.unwrap();
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        let content = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();

        match TestCase::from_json(&value) {
            Ok(tc) => {
                // Basic sanity checks
                assert!(tc.state.n_cards > 0, "{name}: n_cards is 0");
                assert!(!tc.true_evil_positions.is_empty(), "{name}: no true_evil_positions");
                count += 1;
            }
            Err(e) => {
                errors.push(format!("{name}: {e}"));
            }
        }
    }

    if !errors.is_empty() {
        panic!(
            "Failed to parse {} of {} v2 test cases:\n{}",
            errors.len(),
            count + errors.len(),
            errors.join("\n")
        );
    }

    println!("Successfully parsed {count} v2 test cases");
    assert!(count >= 100, "Expected at least 100 v2 test cases, found {count}");
}

#[test]
fn parse_all_legacy_test_cases() {
    let dir = legacy_cases_dir();
    if !dir.exists() {
        println!("Legacy test cases dir not found, skipping");
        return;
    }

    let mut count = 0;
    let mut errors = Vec::new();

    for entry in glob::glob(dir.join("*.json").to_str().unwrap()).unwrap() {
        let path = entry.unwrap();
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        let content = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();

        match TestCase::from_json(&value) {
            Ok(_) => count += 1,
            Err(e) => errors.push(format!("{name}: {e}")),
        }
    }

    if !errors.is_empty() {
        panic!(
            "Failed to parse {} of {} legacy test cases:\n{}",
            errors.len(),
            count + errors.len(),
            errors.join("\n")
        );
    }

    println!("Successfully parsed {count} legacy test cases");
}
