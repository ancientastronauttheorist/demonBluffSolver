#!/usr/bin/env python3
"""Validate a checked-in Ghidra target set against Il2CppDumper script.json."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--script-json", required=True, type=Path)
    return parser.parse_args()


def integer(value: int | str) -> int:
    if isinstance(value, int):
        return value
    return int(value, 0)


def main() -> int:
    args = parse_args()
    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    script = json.loads(args.script_json.read_text(encoding="utf-8-sig"))

    methods_by_name: dict[str, list[dict[str, object]]] = defaultdict(list)
    names_by_address: dict[int, set[str]] = defaultdict(set)
    for method in script["ScriptMethod"]:
        address = integer(method["Address"])
        name = method["Name"]
        methods_by_name[name].append(method)
        names_by_address[address].add(name)

    seen_names: set[str] = set()
    seen_file_names: set[str] = set()
    errors: list[str] = []
    shared_targets: list[tuple[str, int]] = []
    for target in targets["functions"]:
        name = target["name"]
        metadata_name = target["metadata_name"]
        signature = target["signature"]
        rva = integer(target["rva"])
        if name in seen_names:
            errors.append(f"duplicate target name: {name}")
        seen_names.add(name)
        file_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name) + ".c"
        if file_name in seen_file_names:
            errors.append(f"duplicate sanitized filename: {file_name}")
        seen_file_names.add(file_name)
        matching_methods = [
            method
            for method in methods_by_name.get(metadata_name, [])
            if integer(method["Address"]) == rva
        ]
        if not matching_methods:
            errors.append(
                f"{name}: {metadata_name!r} is not mapped to {target['rva']}"
            )
        elif not any(method.get("Signature") == signature for method in matching_methods):
            errors.append(f"{name}: metadata signature does not match script.json")
        alias_count = len(names_by_address.get(rva, set()))
        if alias_count > 1:
            shared_targets.append((name, alias_count))

    if errors:
        for error in errors:
            print(f"error={error}")
        return 1

    print(f"validated_targets={len(targets['functions'])}")
    print(f"targets_with_shared_native_bodies={len(shared_targets)}")
    for name, alias_count in shared_targets:
        print(f"shared_native_body={name}:{alias_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
