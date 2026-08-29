#!/usr/bin/env python3
"""Create deterministic Ghidra C-parser inputs from Il2CppDumper output.

Raw dumper output stays outside the repository.  This command verifies that
the supplied header belongs to the checked-in extraction manifest, translates
the small amount of C++ syntax used by Il2CppDumper, and writes build-keyed
metadata consumed by ``BuildIl2CppTypeArchive.java``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from collections.abc import Sequence
from typing import Any


NORMALIZED_HEADER_NAME = "il2cpp_ghidra.h"
PROTOTYPE_HEADER_NAME = "il2cpp_target_prototypes.h"
ALIGNMENT_MANIFEST_NAME = "il2cpp_alignments.json"
NORMALIZATION_SUMMARY_NAME = "normalization-summary.json"

HEADER_BASELINES = {
    # Current Demon Bluff Playtest Il2CppDumper 6.7.46 header.  This is
    # intentionally independent of the mutable extraction manifest.
    "E1556B4DC5953FEA2D9DE2D071E76AD0B795A3B502AC31DA10D47DCA51D688C6": {
        "alignment_count": 6159,
        "inheritance_rewrite_count": 5830,
    },
}

PRELUDE = """\
/* Deterministic Windows x86-64 primitive types for Ghidra's C parser. */
#ifndef DEMON_BLUFF_IL2CPP_FIXED_WIDTH_TYPES
#define DEMON_BLUFF_IL2CPP_FIXED_WIDTH_TYPES
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long long int64_t;
typedef signed long long intptr_t;
typedef unsigned long long uintptr_t;
typedef unsigned long long size_t;
typedef _Bool bool;
#endif

"""

INHERITANCE_RE = re.compile(
    r"^struct (?P<derived>[A-Za-z_][A-Za-z0-9_]*_Fields) : "
    r"(?P<base>[A-Za-z_][A-Za-z0-9_]*_Fields) \{$"
)
BROAD_INHERITANCE_RE = re.compile(r"^struct\s+.+\s:\s.+\{$")
ALIGN_DECL_RE = re.compile(
    r"^struct __declspec\(align\(8\)\) "
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*) \{$"
)
ALIGN_USE_RE = re.compile(r"__declspec\(align\((?P<value>[0-9]+)\)\)")
FUNCTION_NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
BUILD_ID_RE = re.compile(r"^[0-9a-f]{12}_[0-9a-f]{12}$")


class NormalizationError(ValueError):
    """Raised when an input cannot be normalized without guessing."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise NormalizationError(f"Expected a JSON object: {path}")
    return value


def validate_header(
    header_bytes: bytes, extraction: dict[str, Any]
) -> tuple[str, str]:
    build_id = extraction.get("build_id")
    if not isinstance(build_id, str) or not BUILD_ID_RE.fullmatch(build_id):
        raise NormalizationError(f"Invalid extraction build_id: {build_id!r}")

    try:
        expected = extraction["outputs"]["il2cpp_h"]
        expected_hash = str(expected["sha256"]).upper()
        expected_size = int(expected["size"])
    except (KeyError, TypeError, ValueError) as error:
        raise NormalizationError("Extraction manifest has no valid il2cpp_h entry") from error

    actual_hash = sha256_bytes(header_bytes)
    if actual_hash != expected_hash or len(header_bytes) != expected_size:
        raise NormalizationError(
            "il2cpp.h does not match the extraction manifest: "
            f"expected {expected_hash}/{expected_size}, "
            f"found {actual_hash}/{len(header_bytes)}"
        )
    return build_id, actual_hash


def normalize_header(source: str, build_id: str, source_hash: str) -> tuple[str, list[str], int]:
    output: list[str] = [
        "/*",
        " * Generated from private Il2CppDumper output.",
        f" * Build: {build_id}",
        f" * Source SHA-256: {source_hash}",
        " */",
        PRELUDE.rstrip("\n"),
    ]
    aligned_names: list[str] = []
    aligned_seen: set[str] = set()
    rewritten = 0

    for line_number, line in enumerate(source.splitlines(), 1):
        inheritance = INHERITANCE_RE.fullmatch(line)
        if inheritance:
            output.append(f"struct {inheritance.group('derived')} {{")
            output.append(f"\tstruct {inheritance.group('base')} super;")
            rewritten += 1
            continue
        if BROAD_INHERITANCE_RE.fullmatch(line):
            raise NormalizationError(
                f"Unsupported inheritance declaration at il2cpp.h:{line_number}: {line}"
            )

        alignment = ALIGN_DECL_RE.fullmatch(line)
        if alignment:
            name = alignment.group("name")
            if name in aligned_seen:
                raise NormalizationError(
                    f"Duplicate align(8) structure name at il2cpp.h:{line_number}: {name}"
                )
            aligned_seen.add(name)
            aligned_names.append(name)

        for use in ALIGN_USE_RE.finditer(line):
            if use.group("value") != "8":
                raise NormalizationError(
                    f"Unsupported alignment at il2cpp.h:{line_number}: {use.group(0)}"
                )
            if alignment is None:
                raise NormalizationError(
                    f"align(8) is not on a named structure at il2cpp.h:{line_number}"
                )
        output.append(line)

    normalized = "\n".join(output) + "\n"
    if BROAD_INHERITANCE_RE.search(normalized):
        raise AssertionError("Normalizer left C++ inheritance syntax in its output")
    return normalized, aligned_names, rewritten


def generate_prototypes(
    target_sets: Sequence[dict[str, Any]], build_id: str
) -> tuple[str, list[str]]:
    signatures_by_name: dict[str, str] = {}
    for set_index, targets in enumerate(target_sets):
        if targets.get("build_id") != build_id:
            raise NormalizationError(
                f"Target set {set_index} belongs to {targets.get('build_id')!r}, "
                f"not {build_id}"
            )
        functions = targets.get("functions")
        if not isinstance(functions, list) or not functions:
            raise NormalizationError(f"Target set {set_index} contains no functions")
        for function_index, target in enumerate(functions):
            if not isinstance(target, dict):
                raise NormalizationError(
                    f"Target function {set_index}:{function_index} is not an object"
                )
            signature = target.get("signature")
            if not isinstance(signature, str) or "\n" in signature or "\r" in signature:
                raise NormalizationError(
                    f"Target function {set_index}:{function_index} has an invalid signature"
                )
            signature = signature.strip()
            if not signature.endswith(";"):
                raise NormalizationError(f"Target signature lacks a semicolon: {signature!r}")
            match = FUNCTION_NAME_RE.search(signature)
            if match is None:
                raise NormalizationError(f"Cannot identify function name in {signature!r}")
            name = match.group(1)
            previous = signatures_by_name.get(name)
            if previous is not None and previous != signature:
                raise NormalizationError(
                    f"Conflicting signatures for prototype {name}: "
                    f"{previous!r} != {signature!r}"
                )
            signatures_by_name[name] = signature

    names = sorted(signatures_by_name)
    signatures = [signatures_by_name[name] for name in names]

    lines = [
        "/* Generated target prototypes for Ghidra's C parser. */",
        f"/* Build: {build_id} */",
        "#ifndef DEMON_BLUFF_IL2CPP_TARGET_PROTOTYPES",
        "#define DEMON_BLUFF_IL2CPP_TARGET_PROTOTYPES",
        *signatures,
        "#endif",
        "",
    ]
    return "\n".join(lines), names


def build_outputs(
    header_path: Path,
    targets_paths: Sequence[Path],
    extraction_manifest_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    if not targets_paths:
        raise NormalizationError("At least one target JSON is required")
    header_bytes = header_path.read_bytes()
    extraction = load_json_object(extraction_manifest_path)
    build_id, source_hash = validate_header(header_bytes, extraction)
    target_inputs: list[dict[str, str]] = []
    loaded_target_sets: list[dict[str, Any]] = []
    sortable_targets: list[tuple[str, str, Path, bytes]] = []
    for path in targets_paths:
        target_bytes = path.read_bytes()
        sortable_targets.append(
            (path.name, sha256_bytes(target_bytes), path, target_bytes)
        )
    sortable_targets.sort(key=lambda item: (item[0], item[1]))
    for name, target_hash, path, _ in sortable_targets:
        loaded_target_sets.append(load_json_object(path))
        target_inputs.append({"name": name, "sha256": target_hash})

    try:
        source = header_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise NormalizationError(f"il2cpp.h is not UTF-8: {header_path}") from error

    normalized, aligned_names, inheritance_count = normalize_header(
        source, build_id, source_hash
    )
    baseline = HEADER_BASELINES.get(source_hash)
    if baseline is not None:
        actual_counts = {
            "alignment_count": len(aligned_names),
            "inheritance_rewrite_count": inheritance_count,
        }
        if actual_counts != baseline:
            raise NormalizationError(
                f"Known-header normalization baseline changed for {source_hash}: "
                f"expected {baseline}, found {actual_counts}"
            )
    prototypes, prototype_names = generate_prototypes(loaded_target_sets, build_id)
    normalized_bytes = normalized.encode("utf-8")
    prototype_bytes = prototypes.encode("utf-8")

    alignment_manifest = {
        "alignment": 8,
        "alignment_count": len(aligned_names),
        "build_id": build_id,
        "inheritance_rewrite_count": inheritance_count,
        "inputs": {
            "il2cpp_h_sha256": source_hash,
            "target_sets": target_inputs,
        },
        "names": aligned_names,
        "outputs": {
            "normalized_header_sha256": sha256_bytes(normalized_bytes),
            "prototype_header_sha256": sha256_bytes(prototype_bytes),
        },
        "prototype_count": len(prototype_names),
        "prototype_names": prototype_names,
        "schema_version": 1,
    }

    normalized_path = output_directory / NORMALIZED_HEADER_NAME
    prototype_path = output_directory / PROTOTYPE_HEADER_NAME
    alignment_path = output_directory / ALIGNMENT_MANIFEST_NAME
    summary_path = output_directory / NORMALIZATION_SUMMARY_NAME
    alignment_bytes = canonical_json(alignment_manifest)
    normalization_summary = {
        "alignment_count": len(aligned_names),
        "alignment_manifest": {
            "name": ALIGNMENT_MANIFEST_NAME,
            "sha256": sha256_bytes(alignment_bytes),
            "size": len(alignment_bytes),
        },
        "build_id": build_id,
        "inheritance_rewrite_count": inheritance_count,
        "normalized_header": {
            "name": NORMALIZED_HEADER_NAME,
            "sha256": sha256_bytes(normalized_bytes),
            "size": len(normalized_bytes),
        },
        "prototype_count": len(prototype_names),
        "prototype_header": {
            "name": PROTOTYPE_HEADER_NAME,
            "sha256": sha256_bytes(prototype_bytes),
            "size": len(prototype_bytes),
        },
        "schema_version": 1,
        "success": True,
        "target_sets": target_inputs,
    }
    atomic_write(normalized_path, normalized_bytes)
    atomic_write(prototype_path, prototype_bytes)
    atomic_write(alignment_path, alignment_bytes)
    atomic_write(summary_path, canonical_json(normalization_summary))

    return {
        "alignment_manifest": alignment_path,
        "build_id": build_id,
        "normalized_header": normalized_path,
        "prototype_header": prototype_path,
        "normalization_summary": summary_path,
        "summary": alignment_manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--il2cpp-h", required=True, type=Path)
    parser.add_argument(
        "--targets",
        required=True,
        action="append",
        type=Path,
        help="Checked-in target JSON; repeat for the union of target sets",
    )
    parser.add_argument("--extraction-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_outputs(
        args.il2cpp_h,
        args.targets,
        args.extraction_manifest,
        args.output_dir,
    )
    summary = result["summary"]
    print(f"build_id={result['build_id']}")
    print(f"normalized_header={result['normalized_header']}")
    print(f"prototype_header={result['prototype_header']}")
    print(f"alignment_manifest={result['alignment_manifest']}")
    print(f"normalization_summary={result['normalization_summary']}")
    print(f"inheritance_rewrites={summary['inheritance_rewrite_count']}")
    print(f"aligned_structures={summary['alignment_count']}")
    print(f"prototypes={summary['prototype_count']}")
    print(f"target_sets={len(summary['inputs']['target_sets'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
