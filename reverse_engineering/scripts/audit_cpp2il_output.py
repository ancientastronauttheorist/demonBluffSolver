#!/usr/bin/env python3
"""Measure quality markers in local Cpp2IL -> ILSpy C# output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PATTERNS = {
    "decompiler_issue_notes": "Cpp2ILHelpers.NoteDecompilerIssue",
    "method_not_found_notes": "Method not found @",
    "indirect_call_notes": "Indirect call:",
    "ilspy_expected_type_warnings": "Expected I",
    "unknown_result_type_warnings": "Unknown result type",
    "throw_null_statements": "throw null;",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def audit(source_dir: Path, assembly_dll: Path | None) -> dict:
    files = sorted(source_dir.rglob("*.cs"))
    counts = {name: 0 for name in PATTERNS}
    files_with_issue_notes = 0
    total_bytes = 0
    total_lines = 0
    for path in files:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        total_bytes += path.stat().st_size
        total_lines += text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        if PATTERNS["decompiler_issue_notes"] in text:
            files_with_issue_notes += 1
        for name, pattern in PATTERNS.items():
            counts[name] += text.count(pattern)

    result = {
        "schema_version": 1,
        "source_file_count": len(files),
        "source_bytes": total_bytes,
        "source_line_count": total_lines,
        "files_with_decompiler_issue_notes": files_with_issue_notes,
        "quality_markers": counts,
    }
    if assembly_dll is not None:
        result["assembly_csharp"] = {
            "sha256": sha256(assembly_dll),
            "size": assembly_dll.stat().st_size,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--assembly-dll", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.source_dir, args.assembly_dll)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
