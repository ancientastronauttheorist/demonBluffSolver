#!/usr/bin/env python3
"""Create a stable, redistributable fingerprint manifest for a game build."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PE_MACHINES = {
    0x014C: "x86",
    0x8664: "x86-64",
    0xAA64: "arm64",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def pe_info(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ValueError(f"not a PE file: {path}")
        stream.seek(0x3C)
        pe_offset = struct.unpack("<I", stream.read(4))[0]
        stream.seek(pe_offset)
        if stream.read(4) != b"PE\0\0":
            raise ValueError(f"invalid PE signature: {path}")
        machine, section_count, timestamp = struct.unpack("<HHI", stream.read(8))
        stream.seek(pe_offset + 24)
        optional_magic = struct.unpack("<H", stream.read(2))[0]
    return {
        "architecture": PE_MACHINES.get(machine, f"unknown-0x{machine:04X}"),
        "machine": f"0x{machine:04X}",
        "optional_header": {0x10B: "PE32", 0x20B: "PE32+"}.get(
            optional_magic, f"unknown-0x{optional_magic:04X}"
        ),
        "pe_timestamp": timestamp,
        "pe_timestamp_utc": datetime.fromtimestamp(
            timestamp, tz=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "section_count": section_count,
    }


def file_record(path: Path, relative_path: str, *, pe: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    record: dict[str, Any] = {
        "path": relative_path.replace("\\", "/"),
        "sha256": sha256(path),
        "size": path.stat().st_size,
    }
    if pe:
        record["pe"] = pe_info(path)
    return record


def metadata_info(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        magic, version = struct.unpack("<II", stream.read(8))
    return {
        "magic": f"0x{magic:08X}",
        "version": version,
    }


def unity_version(global_game_managers: Path) -> str:
    data = global_game_managers.read_bytes()
    match = re.search(rb"20\d{2}\.\d+\.\d+[abfp]\d+", data)
    if not match:
        raise ValueError("Unity version not found in globalgamemanagers")
    return match.group(0).decode("ascii")


def parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in re.findall(r'"([^"]+)"\s+"([^"]*)"', path.read_text(encoding="utf-8")):
        values.setdefault(key, value)
    return values


def steam_info(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    values = parse_key_values(path)
    allowed = ("appid", "name", "buildid", "installdir")
    result = {key: values[key] for key in allowed if key in values}
    depot_match = re.search(
        r'"InstalledDepots"\s*\{\s*"(?P<depot>\d+)"\s*\{.*?'
        r'"manifest"\s+"(?P<manifest>\d+)"',
        path.read_text(encoding="utf-8"),
        flags=re.DOTALL,
    )
    if depot_match:
        result["depot_id"] = depot_match.group("depot")
        result["depot_manifest"] = depot_match.group("manifest")
    return result


def create_manifest(game_root: Path, steam_manifest: Path | None) -> dict[str, Any]:
    data_root = game_root / "Demon Bluff_Data"
    metadata_path = data_root / "il2cpp_data" / "Metadata" / "global-metadata.dat"
    scripting_path = data_root / "ScriptingAssemblies.json"
    global_managers_path = data_root / "globalgamemanagers"

    records = {
        "executable": file_record(game_root / "Demon Bluff.exe", "Demon Bluff.exe", pe=True),
        "game_assembly": file_record(game_root / "GameAssembly.dll", "GameAssembly.dll", pe=True),
        "global_metadata": file_record(
            metadata_path,
            "Demon Bluff_Data/il2cpp_data/Metadata/global-metadata.dat",
        ),
        "globalgamemanagers": file_record(
            global_managers_path, "Demon Bluff_Data/globalgamemanagers"
        ),
        "level0": file_record(data_root / "level0", "Demon Bluff_Data/level0"),
        "sharedassets0": file_record(
            data_root / "sharedassets0.assets",
            "Demon Bluff_Data/sharedassets0.assets",
        ),
        "scripting_assemblies": file_record(
            scripting_path, "Demon Bluff_Data/ScriptingAssemblies.json"
        ),
        "unity_player": file_record(game_root / "UnityPlayer.dll", "UnityPlayer.dll", pe=True),
    }

    metadata = metadata_info(metadata_path)
    scripting = json.loads(scripting_path.read_text(encoding="utf-8"))
    app_lines = (data_root / "app.info").read_text(encoding="utf-8").splitlines()
    boot_values = dict(
        line.split("=", 1)
        for line in (data_root / "boot.config").read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    ga_prefix = records["game_assembly"]["sha256"][:12].lower()
    metadata_prefix = records["global_metadata"]["sha256"][:12].lower()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "build_id": f"{ga_prefix}_{metadata_prefix}",
        "product": {
            "company": app_lines[0] if app_lines else None,
            "name": app_lines[1] if len(app_lines) > 1 else game_root.name,
            "unity_version": unity_version(global_managers_path),
            "build_guid": boot_values.get("build-guid"),
            "backend": "IL2CPP",
            "platform": "Windows",
            "architecture": records["game_assembly"]["pe"]["architecture"],
        },
        "steam": steam_info(steam_manifest),
        "il2cpp": metadata,
        "managed_assemblies": {
            "declared_count": len(scripting.get("names", [])),
            "names": scripting.get("names", []),
        },
        "inputs": records,
    }
    if manifest["steam"] is None:
        del manifest["steam"]
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-root", required=True, type=Path)
    parser.add_argument("--steam-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = create_manifest(args.game_root.resolve(), args.steam_manifest)
    output = args.output
    if output is None:
        output = (
            Path(__file__).resolve().parents[1]
            / "manifests"
            / "builds"
            / f"{manifest['build_id']}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(f"build_id={manifest['build_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
