#!/usr/bin/env python3
"""Compare baseline and typed Ghidra exports without emitting decompiled text.

The JSON report contains only export completion counts, filenames, and aggregate
or per-file integer metrics. It intentionally excludes source excerpts, matched
tokens, content hashes, and private filesystem paths so the report is safe to
check in independently from the private decompiler artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final


SUMMARY_NAME: Final = "_export_summary.json"
REQUIRED_SUMMARY_COUNTS: Final = ("requested", "processed", "exported", "failed")

METRIC_DIRECTIONS: Final[dict[str, str]] = {
    "bare_undefined_type_tokens": "lower_is_better",
    "byte_count": "informational",
    "decompiler_error_markers": "lower_is_better",
    "decompiler_warning_markers": "lower_is_better",
    "indirect_call_patterns": "lower_is_better",
    "line_count": "informational",
    "named_struct_field_accesses": "higher_is_better",
    "nonblank_line_count": "informational",
    "placeholder_parameter_tokens": "lower_is_better",
    "raw_field_offset_accesses": "lower_is_better",
    "raw_integer_type_tokens": "lower_is_better",
    "raw_pointer_casts": "lower_is_better",
    "signature_parameter_name_tokens": "higher_is_better",
    "typed_il2cpp_type_tokens": "higher_is_better",
    "unknown_data_symbol_tokens": "lower_is_better",
    "unknown_function_symbol_tokens": "lower_is_better",
    "unknown_label_tokens": "lower_is_better",
    "unresolved_type_tokens": "lower_is_better",
}

# These signals are sufficiently direct that an aggregate increase is treated
# as a typed-output regression by --check. Other directional metrics remain in
# the report but are not gates because a more complete decompilation can
# legitimately contain more casts, symbols, or warnings than a truncated one.
CHECK_NON_REGRESSION_METRICS: Final = (
    "decompiler_error_markers",
    "raw_field_offset_accesses",
    "unresolved_type_tokens",
)

PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "bare_undefined_type_tokens": re.compile(r"\bundefined\b"),
    "decompiler_error_markers": re.compile(
        r"DECOMPILER ERROR|/\*\s*ERROR:|Could not recover|Bad instruction|"
        r"(?:failed|unable) to decompile",
        re.IGNORECASE,
    ),
    "decompiler_warning_markers": re.compile(r"/\*\s*WARNING:"),
    "indirect_call_patterns": re.compile(r"\(\s*\*\s*\*\s*\(\s*code\s*\*+"),
    "named_struct_field_accesses": re.compile(
        r"(?:->|\.)fields(?:->|\.)[A-Za-z_][A-Za-z0-9_]*"
    ),
    "placeholder_parameter_tokens": re.compile(r"\bparam_[0-9]+\b"),
    "raw_field_offset_accesses": re.compile(
        r"\*\s*\(\s*[^()\n;]*\*+\s*\)\s*\([^;\n]*?\+\s*"
        r"(?:0x[0-9A-Fa-f]+|[1-9][0-9]*)\s*\)"
    ),
    "raw_integer_type_tokens": re.compile(r"\b(?:longlong|ulonglong)\b"),
    "raw_pointer_casts": re.compile(
        r"\(\s*(?:const\s+)?(?:struct\s+)?[A-Za-z_][A-Za-z0-9_]*"
        r"(?:\s+[A-Za-z_][A-Za-z0-9_]*)*\s*\*+\s*\)"
    ),
    "signature_parameter_name_tokens": re.compile(r"\b(?:__this|method)\b"),
    "typed_il2cpp_type_tokens": re.compile(
        r"\b[A-Za-z_][A-Za-z0-9_]*(?:_o|_Fields|_StaticFields|_VTable)\b"
    ),
    "unknown_data_symbol_tokens": re.compile(r"\b(?:DAT|PTR|UNK)_[A-Za-z0-9_]+\b"),
    "unknown_function_symbol_tokens": re.compile(
        r"\b(?:thunk_)?FUN_[0-9A-Fa-f]+\b"
    ),
    "unknown_label_tokens": re.compile(r"\bLAB_[0-9A-Fa-f]+\b"),
    "unresolved_type_tokens": re.compile(r"\bundefined[0-9]*\b"),
}


class AuditError(ValueError):
    """Raised when an export directory cannot be compared safely."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--typed", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write canonical JSON atomically instead of printing it to stdout",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return 1 if robust aggregate quality signals regress",
    )
    return parser.parse_args(argv)


def require_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise AuditError(f"{label} export directory does not exist: {path}")
    return resolved


def read_export_summary(directory: Path, label: str) -> dict[str, int | bool]:
    path = directory / SUMMARY_NAME
    if not path.is_file():
        raise AuditError(f"{label} export summary is missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{label} export summary is not valid UTF-8 JSON") from error
    if not isinstance(raw, Mapping):
        raise AuditError(f"{label} export summary must be a JSON object")

    summary: dict[str, int | bool] = {}
    for field in REQUIRED_SUMMARY_COUNTS:
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AuditError(f"{label} export summary has invalid {field!r}")
        summary[field] = value
    cancelled = raw.get("cancelled")
    if not isinstance(cancelled, bool):
        raise AuditError(f"{label} export summary has invalid 'cancelled'")
    summary["cancelled"] = cancelled

    requested = int(summary["requested"])
    if (
        requested <= 0
        or summary["processed"] != requested
        or summary["exported"] != requested
        or summary["failed"] != 0
        or summary["cancelled"]
    ):
        raise AuditError(f"{label} export summary is incomplete")
    return summary


def export_files(directory: Path, label: str) -> dict[str, Path]:
    files = {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix == ".c"
    }
    if not files:
        raise AuditError(f"{label} export directory contains no C files")
    for name, path in files.items():
        if path.stat().st_size <= 0:
            raise AuditError(f"{label} export is empty: {name}")
    return files


def count_metrics(path: Path) -> dict[str, int]:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuditError(f"export is not valid UTF-8: {path.name}") from error
    lines = text.splitlines()
    metrics = {
        "byte_count": len(data),
        "line_count": len(lines),
        "nonblank_line_count": sum(bool(line.strip()) for line in lines),
    }
    for name, pattern in PATTERNS.items():
        metrics[name] = sum(1 for _ in pattern.finditer(text))
    if set(metrics) != set(METRIC_DIRECTIONS):
        raise AssertionError("metric implementation and direction manifest disagree")
    return {name: metrics[name] for name in sorted(metrics)}


def sum_metrics(metrics: Sequence[Mapping[str, int]]) -> dict[str, int]:
    return {
        name: sum(item[name] for item in metrics)
        for name in sorted(METRIC_DIRECTIONS)
    }


def deltas(baseline: Mapping[str, int], typed: Mapping[str, int]) -> dict[str, int]:
    return {name: typed[name] - baseline[name] for name in sorted(METRIC_DIRECTIONS)}


def quality_deltas(
    baseline: Mapping[str, int], typed: Mapping[str, int]
) -> dict[str, int]:
    result: dict[str, int] = {}
    for name in sorted(METRIC_DIRECTIONS):
        direction = METRIC_DIRECTIONS[name]
        if direction == "lower_is_better":
            result[name] = baseline[name] - typed[name]
        elif direction == "higher_is_better":
            result[name] = typed[name] - baseline[name]
    return result


def build_report(
    baseline_directory: Path, typed_directory: Path, *, check: bool = False
) -> dict[str, object]:
    baseline_root = require_directory(baseline_directory, "baseline")
    typed_root = require_directory(typed_directory, "typed")
    if baseline_root == typed_root:
        raise AuditError("baseline and typed export directories must be different")

    baseline_summary = read_export_summary(baseline_root, "baseline")
    typed_summary = read_export_summary(typed_root, "typed")
    baseline_files = export_files(baseline_root, "baseline")
    typed_files = export_files(typed_root, "typed")
    baseline_names = sorted(baseline_files)
    typed_names = sorted(typed_files)
    if baseline_names != typed_names:
        missing_from_typed = sorted(set(baseline_names) - set(typed_names))
        extra_in_typed = sorted(set(typed_names) - set(baseline_names))
        raise AuditError(
            "export filenames do not match "
            f"(missing_from_typed={missing_from_typed}, extra_in_typed={extra_in_typed})"
        )
    if baseline_summary != typed_summary:
        raise AuditError("baseline and typed export completion summaries do not match")
    if baseline_summary["exported"] != len(baseline_names):
        raise AuditError("export summary count does not match the C filename count")

    targets: list[dict[str, object]] = []
    baseline_metric_sets: list[dict[str, int]] = []
    typed_metric_sets: list[dict[str, int]] = []
    for name in baseline_names:
        baseline_metrics = count_metrics(baseline_files[name])
        typed_metrics = count_metrics(typed_files[name])
        baseline_metric_sets.append(baseline_metrics)
        typed_metric_sets.append(typed_metrics)
        targets.append(
            {
                "baseline": baseline_metrics,
                "delta": deltas(baseline_metrics, typed_metrics),
                "filename": name,
                "quality_delta": quality_deltas(baseline_metrics, typed_metrics),
                "typed": typed_metrics,
            }
        )

    baseline_aggregate = sum_metrics(baseline_metric_sets)
    typed_aggregate = sum_metrics(typed_metric_sets)
    aggregate_quality_delta = quality_deltas(baseline_aggregate, typed_aggregate)
    regressions = [
        {
            "baseline": baseline_aggregate[name],
            "metric": name,
            "typed": typed_aggregate[name],
        }
        for name in CHECK_NON_REGRESSION_METRICS
        if typed_aggregate[name] > baseline_aggregate[name]
    ]
    improvements = [
        name
        for name in sorted(aggregate_quality_delta)
        if aggregate_quality_delta[name] > 0
    ]

    return {
        "aggregate": {
            "baseline": baseline_aggregate,
            "delta": deltas(baseline_aggregate, typed_aggregate),
            "quality_delta": aggregate_quality_delta,
            "typed": typed_aggregate,
        },
        "check": {
            "enabled": check,
            "improved_directional_metrics": improvements,
            "passed": not regressions,
            "policy_metrics": list(CHECK_NON_REGRESSION_METRICS),
            "regressions": regressions,
        },
        "metric_directions": dict(sorted(METRIC_DIRECTIONS.items())),
        "schema_version": 1,
        "summaries": {
            "baseline": baseline_summary,
            "typed": typed_summary,
        },
        "target_count": len(targets),
        "targets": targets,
        "validation": {
            "filenames_match": True,
            "nonempty_utf8_exports": True,
            "summaries_complete_and_match": True,
        },
    }


def canonical_json(report: Mapping[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def write_report(path: Path, content: str, protected_inputs: Sequence[Path]) -> None:
    resolved = path.resolve()
    protected = {item.resolve() for item in protected_inputs}
    if resolved in protected or resolved.suffix.lower() != ".json":
        raise AuditError("--output must be a distinct .json report path")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args.baseline, args.typed, check=args.check)
        content = canonical_json(report)
        if args.output is None:
            sys.stdout.write(content)
        else:
            protected = [
                args.baseline / SUMMARY_NAME,
                args.typed / SUMMARY_NAME,
                *args.baseline.glob("*.c"),
                *args.typed.glob("*.c"),
            ]
            write_report(args.output, content, protected)
        if args.check and not report["check"]["passed"]:  # type: ignore[index]
            return 1
        return 0
    except (AuditError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
