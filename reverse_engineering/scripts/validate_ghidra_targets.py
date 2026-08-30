#!/usr/bin/env python3
"""Validate a checked-in Ghidra target set against Il2CppDumper script.json."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


C_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
FUNCTION_NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


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
    prototype_signatures: dict[str, str] = {}
    prototype_names_by_signature: dict[str, str] = {}
    applied_names_by_signature: dict[str, str] = {}
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
        signature_match = FUNCTION_NAME_RE.search(signature)
        if signature_match is None:
            errors.append(f"{name}: cannot identify C function name in signature")
        else:
            declared_name = signature_match.group(1)
            prototype_name = target.get("prototype_name", declared_name)
            if not isinstance(prototype_name, str) or not C_IDENTIFIER_RE.fullmatch(
                prototype_name
            ):
                errors.append(f"{name}: invalid prototype_name {prototype_name!r}")
            else:
                applied_prototype_name = target.get(
                    "applied_prototype_name", prototype_name
                )
                if not isinstance(
                    applied_prototype_name, str
                ) or not C_IDENTIFIER_RE.fullmatch(applied_prototype_name):
                    errors.append(
                        f"{name}: invalid applied_prototype_name "
                        f"{applied_prototype_name!r}"
                    )
                else:
                    previous_applied_name = applied_names_by_signature.get(signature)
                    if (
                        previous_applied_name is not None
                        and previous_applied_name != applied_prototype_name
                    ):
                        errors.append(
                            f"{name}: signature is mapped to both applied prototypes "
                            f"{previous_applied_name!r} and {applied_prototype_name!r}"
                        )
                    applied_names_by_signature[signature] = applied_prototype_name
                previous_name = prototype_names_by_signature.get(signature)
                if previous_name is not None and previous_name != prototype_name:
                    errors.append(
                        f"{name}: signature is mapped to both {previous_name!r} "
                        f"and {prototype_name!r}"
                    )
                prototype_names_by_signature[signature] = prototype_name
                prototype_signature = (
                    signature[: signature_match.start(1)]
                    + prototype_name
                    + signature[signature_match.end(1) :]
                )
                previous_signature = prototype_signatures.get(prototype_name)
                if (
                    previous_signature is not None
                    and previous_signature != prototype_signature
                ):
                    errors.append(
                        f"{name}: conflicting signatures for prototype {prototype_name}"
                    )
                prototype_signatures[prototype_name] = prototype_signature
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
