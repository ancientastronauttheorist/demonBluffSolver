#!/usr/bin/env python3
"""Index Il2CppDumper output without copying bulk decompiler output into Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


IMAGE_RE = re.compile(r"^// Image (?P<index>\d+): (?P<name>.+) - (?P<start>\d+)$")
NAMESPACE_RE = re.compile(r"^// Namespace: ?(?P<namespace>.*)$")
TYPE_RE = re.compile(r"^(?P<declaration>.+) // TypeDefIndex: (?P<index>\d+)$")
TYPE_NAME_RE = re.compile(
    r"\b(?P<kind>class|struct|interface|enum)\s+(?P<name>[^\s:{]+)"
)
FIELD_RE = re.compile(r"^\s+.+; // 0x[0-9A-Fa-f]+$")
RVA_RE = re.compile(r"^\s*// RVA: (?P<rva>-1|0x[0-9A-Fa-f]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def assembly_ranges(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index, image in enumerate(images):
        end = images[index + 1]["type_start"] if index + 1 < len(images) else None
        result.append({**image, "type_end_exclusive": end})
    return result


def assembly_for(type_index: int, images: list[dict[str, Any]]) -> str:
    selected = images[0]
    for image in images:
        if image["type_start"] > type_index:
            break
        selected = image
    return selected["name"]


def parse_dump(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    images: list[dict[str, Any]] = []
    types: list[dict[str, Any]] = []
    current_namespace = ""
    current_type: dict[str, Any] | None = None
    section: str | None = None
    line_count = 0

    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line_count, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            image_match = IMAGE_RE.match(line)
            if image_match:
                images.append(
                    {
                        "image_index": int(image_match.group("index")),
                        "name": image_match.group("name"),
                        "type_start": int(image_match.group("start")),
                    }
                )
                continue
            namespace_match = NAMESPACE_RE.match(line)
            if namespace_match:
                current_namespace = namespace_match.group("namespace")
                section = None
                continue
            type_match = TYPE_RE.match(line)
            if type_match:
                name_match = TYPE_NAME_RE.search(type_match.group("declaration"))
                if not name_match:
                    continue
                current_type = {
                    "type_def_index": int(type_match.group("index")),
                    "namespace": current_namespace,
                    "name": name_match.group("name"),
                    "kind": name_match.group("kind"),
                    "field_count": 0,
                    "method_count": 0,
                    "native_method_count": 0,
                }
                types.append(current_type)
                section = None
                continue
            if current_type is None:
                continue
            if line == "\t// Fields":
                section = "fields"
                continue
            if line == "\t// Methods":
                section = "methods"
                continue
            if section == "fields" and FIELD_RE.match(line):
                current_type["field_count"] += 1
            elif section == "methods":
                rva_match = RVA_RE.match(line)
                if rva_match:
                    current_type["method_count"] += 1
                    if rva_match.group("rva") != "-1":
                        current_type["native_method_count"] += 1

    ranges = assembly_ranges(images)
    for item in types:
        item["assembly"] = assembly_for(item["type_def_index"], ranges)
    return ranges, types, line_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-cs", required=True, type=Path)
    parser.add_argument("--script-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    images, types, line_count = parse_dump(args.dump_cs)
    script_data = json.loads(args.script_json.read_text(encoding="utf-8"))
    game_types = [item for item in types if item["assembly"] == "Assembly-CSharp.dll"]
    kind_counts: dict[str, int] = {}
    for item in game_types:
        kind_counts[item["kind"]] = kind_counts.get(item["kind"], 0) + 1

    summary = {
        "schema_version": 1,
        "inputs": {
            "dump_cs_sha256": sha256(args.dump_cs),
            "script_json_sha256": sha256(args.script_json),
        },
        "dump": {
            "line_count": line_count,
            "image_count": len(images),
            "type_count": len(types),
        },
        "script_json": {
            key: len(value) if isinstance(value, list) else None
            for key, value in script_data.items()
        },
        "assembly_csharp": {
            "type_count": len(game_types),
            "kind_counts": dict(sorted(kind_counts.items())),
            "field_count": sum(item["field_count"] for item in game_types),
            "method_count": sum(item["method_count"] for item in game_types),
            "native_method_count": sum(item["native_method_count"] for item in game_types),
            "type_start": next(
                image["type_start"] for image in images if image["name"] == "Assembly-CSharp.dll"
            ),
            "type_end_exclusive": next(
                image["type_end_exclusive"]
                for image in images
                if image["name"] == "Assembly-CSharp.dll"
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    public_game_types = [
        {key: value for key, value in item.items() if key != "assembly"}
        for item in game_types
    ]
    (args.output_dir / "assembly_csharp_types.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assembly": "Assembly-CSharp.dll",
                "source": summary["inputs"],
                "type_start": summary["assembly_csharp"]["type_start"],
                "type_end_exclusive": summary["assembly_csharp"]["type_end_exclusive"],
                "types": public_game_types,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
