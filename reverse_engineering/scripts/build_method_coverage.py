#!/usr/bin/env python3
"""Build the public Assembly-CSharp method-coverage ledger.

The raw Il2CppDumper artifacts and GameAssembly binary remain private.  This
script emits only normalized metadata identities, native RVA relationships,
and aggregate counts suitable for the public repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
GENERATOR_VERSION = 1
ASSEMBLY = "Assembly-CSharp.dll"

TERMINAL_STATES = (
    "reconstructed",
    "understood",
    "generated",
    "unreachable",
    "unresolved",
)
EVIDENCE_LEVELS = (
    "metadata",
    "native-static",
    "live-validated",
    "behavioral",
    "hypothesis",
)

EXPECTED_BUILD_COUNTS = {
    "f530404b0f3f_807de4a83df4": {
        "method_definitions": 4207,
        "direct_rva_bindings": 4133,
        "primary_rva_missing": 74,
        "generic_definitions": 19,
        "generic_rva_bindings": 25,
        "generic_instantiation_symbols": 28,
        "abstract_methods": 55,
        "native_bindings": 4158,
        "methods_with_native_binding": 4152,
        "methods_without_native_binding": 55,
        "unique_native_bodies": 3066,
        "shared_native_body_groups": 107,
        "bindings_to_shared_native_bodies": 1199,
        "extra_bindings_over_unique_bodies": 1092,
        "max_native_body_bindings": 210,
    }
}

TYPE_RE = re.compile(r"^(?P<declaration>.+) // TypeDefIndex: (?P<index>\d+)$")
RVA_RE = re.compile(r"^\s*// RVA: (?P<rva>-1|0x[0-9A-Fa-f]+)")
GENERIC_RVA_RE = re.compile(r"^\s*\|-RVA: (?P<rva>0x[0-9A-Fa-f]+)")
GENERIC_NAME_RE = re.compile(r"^\s*\|-(?!RVA:)(?P<name>.+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def normalized_declaration(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    if value.endswith("{ }"):
        value = value[:-3].rstrip()
    elif value.endswith(";"):
        value = value[:-1].rstrip()
    return value


def normalized_generic_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def body_id(rva: int) -> str:
    return f"ga:rva:{rva:08X}"


def body_rva(identifier: str) -> int:
    prefix = "ga:rva:"
    if not identifier.startswith(prefix):
        raise ValueError(f"invalid native body id: {identifier}")
    return int(identifier[len(prefix) :], 16)


def executable_sections(path: Path) -> list[tuple[int, int, str]]:
    """Return executable PE section RVA ranges as (start, end, name)."""

    with path.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ValueError(f"not a PE file: {path}")
        stream.seek(0x3C)
        pe_offset = struct.unpack("<I", stream.read(4))[0]
        stream.seek(pe_offset)
        if stream.read(4) != b"PE\0\0":
            raise ValueError(f"invalid PE signature: {path}")
        coff = stream.read(20)
        if len(coff) != 20:
            raise ValueError(f"truncated PE header: {path}")
        _, section_count, _, _, _, optional_size, _ = struct.unpack("<HHIIIHH", coff)
        stream.seek(optional_size, 1)
        result: list[tuple[int, int, str]] = []
        for _ in range(section_count):
            section = stream.read(40)
            if len(section) != 40:
                raise ValueError(f"truncated PE section table: {path}")
            name = section[:8].rstrip(b"\0").decode("ascii", errors="replace")
            virtual_size, virtual_address, raw_size = struct.unpack_from("<III", section, 8)
            characteristics = struct.unpack_from("<I", section, 36)[0]
            if characteristics & 0x20000000:
                result.append(
                    (
                        virtual_address,
                        virtual_address + max(virtual_size, raw_size),
                        name,
                    )
                )
    if not result:
        raise ValueError(f"PE has no executable sections: {path}")
    return result


def assert_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual.upper() != expected.upper():
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def full_type_name(item: dict[str, Any]) -> str:
    namespace = item.get("namespace", "")
    return f"{namespace}.{item['name']}" if namespace else item["name"]


def parse_methods(
    dump_cs: Path, type_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    type_by_index = {int(item["type_def_index"]): item for item in type_items}
    expected_indices = set(type_by_index)
    ordinals: defaultdict[int, int] = defaultdict(int)
    records: list[dict[str, Any]] = []

    current_type: int | None = None
    in_methods = False
    pending: dict[str, Any] | None = None
    current_generic_binding: dict[str, Any] | None = None

    def finish_pending() -> None:
        nonlocal pending, current_generic_binding
        if pending is None:
            return
        if pending["declaration"] is None:
            raise ValueError(f"method declaration missing for {pending['id']}")

        native = pending.pop("native")
        primary_rva = pending.pop("primary_rva")
        declaration = normalized_declaration(pending.pop("declaration"))
        if primary_rva is not None:
            native.insert(
                0,
                {
                    "body": body_id(primary_rva),
                    "binding": "direct",
                },
            )

        for binding in native:
            if binding["binding"] == "generic-instance":
                if not binding["instantiations"]:
                    raise ValueError(
                        f"generic RVA without an instantiation name for {pending['id']}"
                    )
                binding["instantiations"] = list(
                    dict.fromkeys(binding["instantiations"])
                )

        if primary_rva is not None:
            implementation = "concrete"
        elif native:
            implementation = "generic-definition"
        elif re.search(r"\babstract\b", declaration):
            implementation = "abstract"
        else:
            implementation = "missing-native"

        item = type_by_index[pending["type_def_index"]]
        type_name = full_type_name(item)
        records.append(
            {
                "id": pending["id"],
                "type_def_index": pending["type_def_index"],
                "method_ordinal": pending["method_ordinal"],
                "declaring_type": type_name,
                "symbol_key": f"{type_name}::{declaration}",
                "implementation": implementation,
                "native": native,
            }
        )
        pending = None
        current_generic_binding = None

    with dump_cs.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\r\n")
            type_match = TYPE_RE.match(line)
            if type_match:
                finish_pending()
                parsed_index = int(type_match.group("index"))
                current_type = parsed_index if parsed_index in expected_indices else None
                in_methods = False
                continue

            if current_type is None:
                continue
            if line == "\t// Fields":
                finish_pending()
                in_methods = False
                continue
            if line == "\t// Methods":
                finish_pending()
                in_methods = True
                continue
            if not in_methods:
                continue

            rva_match = RVA_RE.match(line)
            if rva_match:
                finish_pending()
                ordinal = ordinals[current_type]
                ordinals[current_type] += 1
                value = rva_match.group("rva")
                pending = {
                    "id": f"tdi{current_type}.m{ordinal:04d}",
                    "type_def_index": current_type,
                    "method_ordinal": ordinal,
                    "primary_rva": None if value == "-1" else int(value, 16),
                    "declaration": None,
                    "native": [],
                }
                current_generic_binding = None
                continue

            if pending is None:
                continue

            generic_rva_match = GENERIC_RVA_RE.match(line)
            if generic_rva_match:
                current_generic_binding = {
                    "body": body_id(int(generic_rva_match.group("rva"), 16)),
                    "binding": "generic-instance",
                    "instantiations": [],
                }
                pending["native"].append(current_generic_binding)
                continue

            generic_name_match = GENERIC_NAME_RE.match(line)
            if generic_name_match and current_generic_binding is not None:
                current_generic_binding["instantiations"].append(
                    normalized_generic_name(generic_name_match.group("name"))
                )
                continue

            candidate = line.strip()
            if (
                pending["declaration"] is None
                and candidate
                and not candidate.startswith("//")
                and "(" in candidate
                and (candidate.endswith("{ }") or candidate.endswith(";"))
            ):
                pending["declaration"] = candidate

    finish_pending()
    return records


def validate_type_counts(
    methods: list[dict[str, Any]], type_items: list[dict[str, Any]]
) -> None:
    methods_by_type: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for method in methods:
        methods_by_type[method["type_def_index"]].append(method)

    errors: list[str] = []
    for item in type_items:
        type_index = int(item["type_def_index"])
        actual = methods_by_type.get(type_index, [])
        expected_methods = int(item["method_count"])
        expected_native = int(item["native_method_count"])
        actual_native = sum(
            any(binding["binding"] == "direct" for binding in method["native"])
            for method in actual
        )
        if len(actual) != expected_methods:
            errors.append(
                f"TypeDefIndex {type_index} method count: "
                f"expected {expected_methods}, got {len(actual)}"
            )
        if actual_native != expected_native:
            errors.append(
                f"TypeDefIndex {type_index} direct RVA count: "
                f"expected {expected_native}, got {actual_native}"
            )
    if errors:
        raise ValueError("type inventory mismatch:\n" + "\n".join(errors))


def coverage_counts(methods: list[dict[str, Any]]) -> dict[str, int]:
    bindings = [binding for method in methods for binding in method["native"]]
    body_counts = Counter(binding["body"] for binding in bindings)
    shared_counts = [count for count in body_counts.values() if count > 1]
    direct = sum(binding["binding"] == "direct" for binding in bindings)
    generic = sum(binding["binding"] == "generic-instance" for binding in bindings)
    generic_symbols = sum(
        len(binding.get("instantiations", []))
        for binding in bindings
        if binding["binding"] == "generic-instance"
    )
    methods_with_native = sum(bool(method["native"]) for method in methods)
    return {
        "method_definitions": len(methods),
        "direct_rva_bindings": direct,
        "primary_rva_missing": len(methods) - direct,
        "generic_definitions": sum(
            method["implementation"] == "generic-definition" for method in methods
        ),
        "generic_rva_bindings": generic,
        "generic_instantiation_symbols": generic_symbols,
        "abstract_methods": sum(
            method["implementation"] == "abstract" for method in methods
        ),
        "native_bindings": len(bindings),
        "methods_with_native_binding": methods_with_native,
        "methods_without_native_binding": len(methods) - methods_with_native,
        "unique_native_bodies": len(body_counts),
        "shared_native_body_groups": len(shared_counts),
        "bindings_to_shared_native_bodies": sum(shared_counts),
        "extra_bindings_over_unique_bodies": len(bindings) - len(body_counts),
        "max_native_body_bindings": max(body_counts.values(), default=0),
    }


def validate_executable_rvas(
    methods: list[dict[str, Any]], sections: list[tuple[int, int, str]]
) -> None:
    bad: list[str] = []
    for method in methods:
        for binding in method["native"]:
            rva = body_rva(binding["body"])
            if not any(start <= rva < end for start, end, _ in sections):
                bad.append(f"{method['id']} -> {binding['body']}")
    if bad:
        raise ValueError(
            "native RVAs outside executable PE sections:\n" + "\n".join(bad[:50])
        )


def validate_expected_counts(build_id: str, counts: dict[str, int]) -> None:
    expected = EXPECTED_BUILD_COUNTS.get(build_id)
    if expected is None:
        raise ValueError(f"no reviewed count baseline registered for build {build_id}")
    if counts != expected:
        differences = [
            f"{key}: expected {expected.get(key)}, got {counts.get(key)}"
            for key in sorted(set(expected) | set(counts))
            if expected.get(key) != counts.get(key)
        ]
        raise ValueError("coverage count mismatch:\n" + "\n".join(differences))


def jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    return text.encode("utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: JSONL record is not an object")
        records.append(value)
    return records


def validate_overlays(
    output_dir: Path, methods: list[dict[str, Any]]
) -> tuple[int, int]:
    method_ids = {method["id"] for method in methods}
    body_ids = {
        binding["body"] for method in methods for binding in method["native"]
    }
    evidence_path = output_dir / "evidence.v1.jsonl"
    classifications_path = output_dir / "classifications.v1.jsonl"
    evidence = load_jsonl(evidence_path)
    classifications = load_jsonl(classifications_path)

    evidence_ids: set[str] = set()
    for record in evidence:
        evidence_id = record.get("id")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("evidence record requires a non-empty string id")
        if evidence_id in evidence_ids:
            raise ValueError(f"duplicate evidence id: {evidence_id}")
        evidence_ids.add(evidence_id)
        if record.get("level") not in EVIDENCE_LEVELS:
            raise ValueError(f"{evidence_id}: invalid evidence level")
        targets = record.get("targets")
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"{evidence_id}: evidence requires targets")
        for target in targets:
            if target not in method_ids and target not in body_ids:
                raise ValueError(f"{evidence_id}: unknown target {target}")
        if not isinstance(record.get("claim"), str) or not record["claim"].strip():
            raise ValueError(f"{evidence_id}: evidence requires a claim")
        if not isinstance(record.get("sources"), list) or not record["sources"]:
            raise ValueError(f"{evidence_id}: evidence requires sources")

    classified_methods: set[str] = set()
    for record in classifications:
        method_id = record.get("method")
        if method_id not in method_ids:
            raise ValueError(f"classification references unknown method: {method_id}")
        if method_id in classified_methods:
            raise ValueError(f"duplicate method classification: {method_id}")
        classified_methods.add(method_id)
        if record.get("state") not in TERMINAL_STATES:
            raise ValueError(f"{method_id}: invalid classification state")
        if not isinstance(record.get("reason"), str) or not record["reason"].strip():
            raise ValueError(f"{method_id}: classification requires a reason")
        refs = record.get("evidence", [])
        if not isinstance(refs, list):
            raise ValueError(f"{method_id}: evidence references must be a list")
        unknown = [item for item in refs if item not in evidence_ids]
        if unknown:
            raise ValueError(f"{method_id}: unknown evidence references {unknown}")

    return len(classifications), len(evidence)


def create_outputs(
    *,
    build_manifest_path: Path,
    type_index_path: Path,
    dump_cs: Path,
    script_json: Path,
    game_assembly: Path,
) -> tuple[bytes, bytes, list[dict[str, Any]], dict[str, int]]:
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    type_index = json.loads(type_index_path.read_text(encoding="utf-8"))
    build_id = build_manifest["build_id"]

    if type_index.get("assembly") != ASSEMBLY:
        raise ValueError(f"expected {ASSEMBLY}, got {type_index.get('assembly')}")
    if type_index.get("type_start") != 5422 or type_index.get("type_end_exclusive") != 6192:
        raise ValueError("unexpected Assembly-CSharp TypeDefIndex range")

    dump_hash = assert_hash(
        dump_cs, type_index["source"]["dump_cs_sha256"], "dump.cs"
    )
    script_hash = assert_hash(
        script_json, type_index["source"]["script_json_sha256"], "script.json"
    )
    game_assembly_hash = assert_hash(
        game_assembly,
        build_manifest["inputs"]["game_assembly"]["sha256"],
        "GameAssembly.dll",
    )

    type_items = type_index["types"]
    methods = parse_methods(dump_cs, type_items)
    methods.sort(key=lambda item: (item["type_def_index"], item["method_ordinal"]))
    validate_type_counts(methods, type_items)
    validate_executable_rvas(methods, executable_sections(game_assembly))
    counts = coverage_counts(methods)
    validate_expected_counts(build_id, counts)
    missing_native = [
        method for method in methods if method["implementation"] == "missing-native"
    ]
    if missing_native:
        raise ValueError(
            "unexplained methods without native bodies: "
            + ", ".join(method["id"] for method in missing_native)
        )

    methods_data = jsonl_bytes(methods)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_id": build_id,
        "assembly": ASSEMBLY,
        "generator": {
            "path": "reverse_engineering/scripts/build_method_coverage.py",
            "version": GENERATOR_VERSION,
        },
        "source": {
            "build_manifest_sha256": sha256(build_manifest_path),
            "type_index_sha256": sha256(type_index_path),
            "dump_cs_sha256": dump_hash,
            "script_json_sha256": script_hash,
            "game_assembly_sha256": game_assembly_hash,
        },
        "type_range": {
            "start": type_index["type_start"],
            "end_exclusive": type_index["type_end_exclusive"],
        },
        "counts": counts,
        "classification": {
            "default": {"state": "unresolved", "reason": "not-reviewed"},
            "terminal_states": list(TERMINAL_STATES),
            "evidence_levels": list(EVIDENCE_LEVELS),
        },
        "files": {
            "methods": {
                "name": "methods.v1.jsonl",
                "records": len(methods),
                "sha256": sha256_bytes(methods_data),
            },
            "classifications": {"name": "classifications.v1.jsonl"},
            "evidence": {"name": "evidence.v1.jsonl"},
        },
    }
    manifest_data = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    return manifest_data, methods_data, methods, counts


def compare_file(path: Path, expected: bytes) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = path.read_bytes()
    if actual != expected:
        raise ValueError(f"generated file is stale: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-manifest", required=True, type=Path)
    parser.add_argument("--type-index", required=True, type=Path)
    parser.add_argument("--dump-cs", required=True, type=Path)
    parser.add_argument("--script-json", required=True, type=Path)
    parser.add_argument("--game-assembly", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate that checked-in generated files are byte-for-byte current",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_data, methods_data, methods, counts = create_outputs(
        build_manifest_path=args.build_manifest.resolve(),
        type_index_path=args.type_index.resolve(),
        dump_cs=args.dump_cs.resolve(),
        script_json=args.script_json.resolve(),
        game_assembly=args.game_assembly.resolve(),
    )
    build_id = json.loads(manifest_data)["build_id"]
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "coverage" / build_id
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.v1.json"
    methods_path = output_dir / "methods.v1.jsonl"
    classifications_path = output_dir / "classifications.v1.jsonl"
    evidence_path = output_dir / "evidence.v1.jsonl"

    if args.check:
        compare_file(manifest_path, manifest_data)
        compare_file(methods_path, methods_data)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(manifest_data)
        methods_path.write_bytes(methods_data)
        if not classifications_path.exists():
            classifications_path.write_text("", encoding="utf-8")
        if not evidence_path.exists():
            evidence_path.write_text("", encoding="utf-8")

    classification_count, evidence_count = validate_overlays(output_dir, methods)
    print(f"build_id={build_id}")
    print(f"method_definitions={counts['method_definitions']}")
    print(f"unique_native_bodies={counts['unique_native_bodies']}")
    print(f"shared_native_body_groups={counts['shared_native_body_groups']}")
    print(f"classifications={classification_count}")
    print(f"evidence_records={evidence_count}")
    print(f"mode={'check' if args.check else 'write'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
