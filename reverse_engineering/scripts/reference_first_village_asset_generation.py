"""Independent asset-occurrence reference for the conditional original N5 domain.

Only Python's standard library is used. No native or solver transition is imported.
Recorded choices and float32 keys are inputs, never inferred probabilities.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import struct
from pathlib import Path

INPUT_SCHEMA = "asset_generation_input_v0"
OUTPUT_SCHEMA = "asset_generation_reference_v0"
FACTIONS = ("villagers", "outsiders", "minions", "demons")
COUNT_FIELDS = ("allCharCount", "town", "demon", "outs", "minion",
                "dTown", "dDemon", "dOuts", "dMinion")
N5_COUNTS = (5, 4, 0, 0, 1, 4, 0, 0, 1)
GENERATION_WIDTHS = (1, 5, 4, 3, 2, 1, 4, 3, 2, 1)
INPUT_REPORT_PINS = {
    "ascension": "6bed679ae21d08cb0c738085d34c717d49aa4fb6afb27e94bf9289b0905e8872",
    "character": "1a790f521ba8ec6983bb1634333accb7280f0fe93e49efbaec1af25914479d42",
}
FIXTURE_CONTRACT = {
    "collections": "successful stable supplied List/Contains/Add/Remove/ToArray",
    "list_capacity": 256, "initial_current_and_saved_versions": [0, 0, 0, 0],
    "initial_pool_versions": [0, 0, 0], "array_versions": None,
    "profile_copy": "field-faithful supplied copy, distinct identity",
    "source_metadata": "valid warmed supplied runtime headers and generic metadata",
    "sorting": "supplied stable ascending finite float32 key sort, source-index ties",
    "deferred_remove_all": "predicate observations before a supplied successful commit",
    "callbacks": "no intervening profile/roster/collection writers",
    "positions": "supplied inert positions service before pool setup",
    "boundary": "stop before first Character.Init; no board lifecycle executed",
}

# Saved API/schema: a stage contains `assets`, `occurrences`, and `lists`.
# Lists have {identity, kind, items:[occurrence_id], version, version_provenance}.
# Occurrences have {occurrence_id, asset:{namespace,file_id,path_id}, origin,
#                   parent_occurrence}. Asset equality excludes occurrence_id.
# `generation_stage` and `before_first_init` are separate immutable snapshots.
# run_generation_case(inputs, plan) consumes plan.generation_indices,
# plan.float_key_bits, and plan.pool_indices, without missing-choice defaults.
# run_minion_selector(inputs, state, die, index, invocation_contract) is separate.
# load_inputs(path,path) consumes only parsed ascension/character INPUT reports.
# Enumeration requires a positive explicit case capacity; exhaustion is explicit.


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode("utf-8")).hexdigest()


def asset_key(namespace, file_id, path_id):
    return {"namespace": namespace, "file_id": file_id, "path_id": path_id}


def same_asset(a, b):
    return a["asset"] == b["asset"]


def first_equal_remove(items, selected, occurrences):
    """Remove the first asset-equal occurrence, not necessarily the drawn one."""
    for i, occurrence in enumerate(items):
        if same_asset(occurrences[occurrence], occurrences[selected]):
            return items.pop(i)
    return None


def float32_key(bits):
    if type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF:
        raise ValueError("key bits must be an unsigned 32-bit integer")
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("only supplied finite float32 keys in [0,1] are admitted")
    return value


def stable_float_order(occurrences, bits):
    if len(occurrences) != len(bits):
        raise ValueError("one key is required for every source occurrence")
    keys = [float32_key(x) for x in bits]
    order = sorted(range(len(keys)), key=lambda i: (keys[i], i))
    return [occurrences[i] for i in order], order


class Boundary(Exception):
    def __init__(self, status, reason, detail=None):
        self.status, self.reason, self.detail = status, reason, detail


class Choices:
    def __init__(self, values, trace, family):
        if not isinstance(values, list):
            raise Boundary("unsupported", "choice_plan_not_list", family)
        self.values, self.trace, self.family, self.used = values, trace, family, 0

    def draw(self, width, purpose, minimum=0):
        ordinal = self.used
        if ordinal == len(self.values):
            raise Boundary("incomplete", "missing_recorded_choice",
                           {"family": self.family, "ordinal": ordinal,
                            "minimum": minimum, "width": width, "purpose": purpose})
        choice = self.values[ordinal]
        self.used += 1
        event = {"family": self.family, "ordinal": ordinal, "minimum": minimum,
                 "maximum_exclusive": minimum + width, "width": width,
                 "choice": choice, "purpose": purpose}
        self.trace.append(event)
        if width == 0:
            raise Boundary("failure", "zero_width_indexed_draw", event)
        if type(choice) is not int or not minimum <= choice < minimum + width:
            raise Boundary("unsupported", "choice_outside_declared_range", event)
        return choice - minimum

    def finish(self):
        if self.used != len(self.values):
            raise Boundary("unsupported", "unused_recorded_choices",
                           {"family": self.family, "used": self.used,
                            "supplied": len(self.values)})


def _read_report(path, required):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or not required <= value.keys():
        raise ValueError("input report schema lacks required fields")
    if value["schema_version"] != 1:
        raise ValueError("unsupported input report schema")
    return value, {"filename": path.name, "bytes": len(raw),
                   "sha256": hashlib.sha256(raw).hexdigest()}


def load_inputs(ascension_report, character_report, profile_id=21674):
    """Validate parsed input artifacts; no generation-output report is accepted."""
    asc, asc_pin = _read_report(ascension_report, {"schema_version", "build_id",
                                               "source_hashes", "ascensions"})
    chars, char_pin = _read_report(character_report, {"schema_version", "build_id",
        "asset_sha256", "game_assembly_sha256", "records", "record_count"})
    if asc_pin["sha256"] != INPUT_REPORT_PINS["ascension"] or char_pin["sha256"] != INPUT_REPORT_PINS["character"]:
        raise ValueError("input reports differ from the reviewed original input artifacts")
    if asc["build_id"] != chars["build_id"]:
        raise ValueError("input build identities differ")
    if not isinstance(asc["source_hashes"], dict):
        raise ValueError("asset source hashes must be a mapping")
    namespace = chars["asset_sha256"].lower()
    if asc["source_hashes"].get("sharedassets0.assets", "").lower() != namespace:
        raise ValueError("input reports name different serialized asset files")
    rows = asc["ascensions"]
    if not isinstance(rows, list):
        raise ValueError("ascensions must be rows")
    matches = [r for r in rows if isinstance(r, dict) and r.get("path_id") == profile_id]
    if len(matches) != 1 or not {"data", "object_sha256", "object_size"} <= matches[0].keys():
        raise ValueError("profile is absent, ambiguous, or incomplete")
    records = chars["records"]
    if not isinstance(records, list) or len(records) != chars["record_count"]:
        raise ValueError("character row count mismatch")
    assets = {}
    needed = {"path_id", "object_sha256", "object_size", "role_type",
              "role_type_def_index", "characterName", "name", "type",
              "startingAlignment", "bluffable", "picking"}
    for row in records:
        if not isinstance(row, dict) or not needed <= row.keys():
            raise ValueError("character input fields absent")
        key = "0:" + str(row["path_id"])
        if key in assets:
            raise ValueError("duplicate character asset identity")
        assets[key] = {"asset": asset_key(namespace, 0, row["path_id"]),
            "managed_class": {"build_id": chars["build_id"],
                              "type_def_index": row["role_type_def_index"],
                              "qualified_name": row["role_type"],
                              "assembly": "Assembly-CSharp",
                              "qualification_source": "input parser's exact role RID registry"},
            "public_name": row["characterName"] or None, "object_name": row["name"],
            "object_sha256": row["object_sha256"].lower(), "object_size": row["object_size"],
            "real_type": row["type"], "starting_alignment": row["startingAlignment"],
            "bluffable": row["bluffable"], "picking": row["picking"]}
    profile = copy.deepcopy(matches[0])
    result = {"schema": INPUT_SCHEMA, "build_id": chars["build_id"],
        "asset_namespace": namespace, "assets": assets, "profile": profile,
        "provenance": {"ascension_report": asc_pin, "character_report": char_pin,
                       "asset_source_hashes": copy.deepcopy(asc["source_hashes"]),
                       "game_assembly_sha256": chars["game_assembly_sha256"].lower()},
        "fixture_contract": copy.deepcopy(FIXTURE_CONTRACT),
        "domain": {"profile_id": profile_id, "mode_enum": 0,
                   "runtime_mode_class": "RoguelikeStandard", "ascension_index": 0,
                   "village_index": 0, "accumulation": False,
                   "starting_materialized": True, "initial_current_rosters": "empty",
                   "initial_saved_rosters": "empty", "no_intervening_graph_writers": True}}
    result["input_origin"] = "pinned_parsed_asset_reports"
    result["input_seal"] = digest(result)
    validate_inputs(result)
    return result


def validate_inputs(inputs):
    if inputs.get("input_seal") != digest({k: v for k, v in inputs.items() if k != "input_seal"}):
        raise ValueError("input seal is absent or changed")
    if inputs.get("schema") != INPUT_SCHEMA or inputs["profile"]["path_id"] != 21674:
        raise ValueError("only the original serialized profile21674 domain is admitted")
    data = inputs["profile"]["data"]
    for key in ("possibleScripts", "possibleScriptsData", "mustInlcude", "alwaysInDeck",
                "townsfolks", "outsiders", "minions", "demons"):
        if not isinstance(data.get(key), list):
            raise ValueError("profile array schema missing: " + key)
    if len(data["possibleScripts"]) != 1 or data["possibleScriptsData"]:
        raise ValueError("only singleton inline script selection is admitted")
    if data["mustInlcude"] or data["alwaysInDeck"]:
        raise ValueError("must-include/always-in-deck nonempty profile is unsupported")
    script = data["possibleScripts"][0]
    if script.get("mustInclude") or len(script.get("characterCounts", [])) != 1:
        raise ValueError("script must-include/count domain unsupported")
    counts = script["characterCounts"][0]
    if tuple(counts.get(k) for k in COUNT_FIELDS) != N5_COUNTS:
        raise ValueError("script is outside N5 counts")
    arrays = [script.get(k) for k in ("startingTownsfolks", "startingOutsiders",
                                     "startingMinions", "startingDemons")]
    if any(not isinstance(x, list) for x in arrays) or [len(x) for x in arrays] != [5, 0, 1, 0]:
        raise ValueError("starting array widths are outside N5")
    if arrays != [data.get(k) for k in ("startingTownsfolks", "startingOutsiders",
                                       "startingMinions", "startingDemons")]:
        raise ValueError("source stored starting fields differ from inline materialization")
    if inputs.get("fixture_contract") != FIXTURE_CONTRACT:
        raise ValueError("unsupported collection/runtime service contract")
    for refs in arrays + [data[k] for k in ("townsfolks", "outsiders", "minions", "demons")]:
        for ref in refs:
            if not isinstance(ref, list) or len(ref) != 2 or any(type(x) is not int for x in ref):
                raise ValueError("PPtr must be [file_id,path_id]")
            if ref[0] != 0 or f"{ref[0]}:{ref[1]}" not in inputs["assets"]:
                raise ValueError("unresolved asset namespace/file mapping")
    villagers = [inputs["assets"][f"{r[0]}:{r[1]}"] for r in arrays[0]]
    minion = inputs["assets"][f"{arrays[2][0][0]}:{arrays[2][0][1]}"]
    if len({tuple(r) for r in arrays[0]}) != 5 or any(
            r["real_type"] != 10 or not r["bluffable"] for r in villagers):
        raise ValueError("N5 pool derivation requires five distinct bluffable Villager assets")
    if minion["real_type"] != 30:
        raise ValueError("N5 singleton Minion must have real type30")
    if inputs["domain"] != {"profile_id": 21674, "mode_enum": 0,
        "runtime_mode_class": "RoguelikeStandard", "ascension_index": 0,
        "village_index": 0, "accumulation": False, "starting_materialized": True,
        "initial_current_rosters": "empty", "initial_saved_rosters": "empty",
        "no_intervening_graph_writers": True}:
        raise ValueError("unsupported generation-entry conditioning")


class Model:
    """Logical identities model the supplied successful stable collection service."""
    def __init__(self, inputs, capacity):
        self.inputs = inputs
        self.capacity = capacity
        self.occurrences, self.lists, self.events = {}, {}, []
        self.draws = []
        self.next_copy = 0
        script = inputs["profile"]["data"]["possibleScripts"][0]
        self.starting = []
        self.topology = []
        for faction, field in zip(FACTIONS, ("startingTownsfolks", "startingOutsiders",
                                             "startingMinions", "startingDemons")):
            inline = self.source(script[field], "inline:0:" + faction)
            stored = self.source(inputs["profile"]["data"][field], "stored_profile:" + field)
            self.new_list("source_inline:" + faction, inline)
            self.new_list("source_starting:" + faction, stored, "array")
            selected = [self.copy_occurrence(x, "selected_inline:" + faction) for x in inline]
            self.new_list("selected_inline:" + faction, selected)
            materialized = [self.copy_occurrence(x, "temporary_starting:" + faction) for x in selected]
            self.new_list("temporary_starting:" + faction, materialized, "array")
            self.starting.append(materialized)
            self.topology.append({"faction": faction,
                "pre_materialization": {"temporary_starting_alias": "source_starting:" + faction},
                "after_materialization": {"temporary_starting": "temporary_starting:" + faction,
                    "source": "selected_inline:" + faction, "original_stored_unchanged": True},
                "proof": "constructor/copy/materialization source contract",
                "corpus_alias_availability": "not serialized; no raw alias equality certificate"})
            self.new_list("current:" + faction, [])
            self.new_list("saved:" + faction, [])
        for name in ("unique_pool", "duplicate_pool", "must_include"):
            self.new_list(name, [])
        self.source_catalogue = {}
        for field in ("townsfolks", "outsiders", "minions", "demons"):
            self.source_catalogue[field] = self.source(inputs["profile"]["data"][field],
                                                       "catalogue:" + field)
        self.fallback_source = [*self.source_catalogue["townsfolks"],
            *self.source_catalogue["outsiders"], *self.source_catalogue["minions"]]
        # A repeated serialized array contributes a new source placement identity.
        self.fallback_source += self.source(inputs["profile"]["data"]["townsfolks"],
                                            "catalogue:townsfolks:second_append")
        self.fallback_eligible = [x for x in self.fallback_source
                                  if self.record(x)["bluffable"]
                                  and self.record(x)["starting_alignment"] == 10
                                  and self.record(x)["real_type"] == 10]
        self.new_list("fallback_source", self.fallback_source, "array")
        self.new_list("fallback_eligible", self.fallback_eligible)
        self.return_source, self.return_order, self.sort_keys = [], [], []
        self.boundary = None

    def source(self, refs, origin):
        result = []
        for i, (file_id, path_id) in enumerate(refs):
            occurrence = origin + ":" + str(i)
            self.occurrences[occurrence] = {
                "occurrence_id": occurrence,
                "asset": asset_key(self.inputs["asset_namespace"], file_id, path_id),
                "origin": {"profile_id": 21674, "source_array": origin, "index": i},
                "parent_occurrence": None}
            result.append(occurrence)
        return result

    def copy_occurrence(self, parent, destination):
        self.next_copy += 1
        occurrence = destination + ":placement:" + str(self.next_copy)
        self.occurrences[occurrence] = {"occurrence_id": occurrence,
            "asset": copy.deepcopy(self.occurrences[parent]["asset"]),
            "origin": {"destination": destination}, "parent_occurrence": parent}
        return occurrence

    def record(self, occurrence):
        a = self.occurrences[occurrence]["asset"]
        return self.inputs["assets"][f'{a["file_id"]}:{a["path_id"]}']

    def new_list(self, identity, items, kind="list"):
        if identity in self.lists:
            raise ValueError("logical collection identity reused")
        if len(items) > self.capacity:
            raise Boundary("capacity", "collection_capacity", identity)
        self.lists[identity] = {"identity": identity, "kind": kind, "items": list(items),
            "version": 0 if kind == "list" else None,
            "version_provenance": "authored_supplied_collection_contract" if kind == "list"
                                  else "array_has_no_list_version"}
        return self.lists[identity]["items"]

    def clear(self, identity):
        row = self.lists[identity]
        row["version"] += 1
        row["items"] = []
        self.events.append({"operation": "clear", "list": identity,
                            "version": row["version"]})

    def add(self, identity, selected):
        row = self.lists[identity]
        if len(row["items"]) == self.capacity:
            raise Boundary("capacity", "collection_capacity_before_add", identity)
        placed = self.copy_occurrence(selected, identity)
        row["items"].append(placed)
        row["version"] += 1
        self.events.append({"operation": "add", "list": identity, "selected": selected,
                            "placed": placed, "version": row["version"]})
        return placed

    def remove(self, items, selected, purpose):
        removed = first_equal_remove(items, selected, self.occurrences)
        if removed is None:
            raise Boundary("unsupported", "selected_asset_missing_for_remove", purpose)
        self.events.append({"operation": "first_equal_remove", "purpose": purpose,
                            "selected": selected, "removed": removed})
        for row in self.lists.values():
            if row["items"] is items and row["kind"] == "list":
                row["version"] += 1
                break

    def current(self, faction):
        return self.lists["current:" + faction]["items"]

    def snapshot(self, stage):
        return {"stage": stage, "assets": copy.deepcopy(self.inputs["assets"]),
            "occurrences": copy.deepcopy(self.occurrences), "lists": copy.deepcopy(self.lists),
            "current_rosters": {f: "current:" + f for f in FACTIONS},
            "saved_rosters": {f: "saved:" + f for f in FACTIONS},
            "source_inline": {f: "source_inline:" + f for f in FACTIONS},
            "selected_inline": {f: "selected_inline:" + f for f in FACTIONS},
            "source_starting": {f: "source_starting:" + f for f in FACTIONS},
            "temporary_starting": {f: "temporary_starting:" + f for f in FACTIONS},
            "source_copy_topology": copy.deepcopy(self.topology),
            "temporary_cache": "profile:21674:inline:0:field_copy",
            "current_script_identity": "profile:21674:inline:0:count:0:field_copy",
            "current_script_fields": dict(zip(COUNT_FIELDS, N5_COUNTS)),
            "selected_counts": [dict(zip(COUNT_FIELDS, N5_COUNTS))],
            "temporary_counts": [f"profile:21674:count:{i}:field_copy" for i in range(
                len(self.inputs["profile"]["data"].get("characterCounts", [])))],
            "temporary_count_source_fields": copy.deepcopy(
                self.inputs["profile"]["data"].get("characterCounts", [])),
            "input_origin": self.inputs["input_origin"],
            "input_digest": digest(self.inputs),
            "return_source": list(self.return_source), "returned_list": "roster_return",
            "returned_order": list(self.return_order), "sort_keys": copy.deepcopy(self.sort_keys),
            "fallback_source": list(self.fallback_source),
            "fallback_eligible": list(self.fallback_eligible),
            "boundary": copy.deepcopy(self.boundary),
            "board": [{"actor_occurrence": i, "data": x, "display_id": 5-i}
                      for i, x in enumerate(self.return_order)],
            "draws": copy.deepcopy(self.draws), "transition_events": copy.deepcopy(self.events),
            "pending_obligations": [row for i, x in enumerate(self.return_order)
                                    for row in callback_obligations(self.inputs, self.occurrences,
                                                                   i, x, "original")],
            "version_comparison_limit": "current/saved versions and saved identities not exposed by recorded factor corpus"}

    def generation(self, plan):
        choices = Choices(plan.get("generation_indices"), self.draws, "generation")
        # Standard samples original starting arrays; GetRandom samples new local
        # candidates, so roster order and sort-source order remain distinct.
        for faction, source, count in (("minions", self.starting[2], 1),
                                       ("villagers", self.starting[0], 4)):
            candidates = self.new_list("standard_candidates:" + faction, source)
            for _ in range(count):
                index = choices.draw(len(candidates), "standard:" + faction)
                selected = candidates[index]
                self.add("current:" + faction, selected)
                self.remove(candidates, selected, "standard:" + faction)
        for faction in ("minions", "villagers"):
            candidates = self.new_list("return_candidates:" + faction, self.current(faction))
            while candidates:
                index = choices.draw(len(candidates), "return_source:" + faction)
                selected = candidates[index]
                self.return_source.append(self.copy_occurrence(selected, "return_source"))
                self.remove(candidates, selected, "return_source:" + faction)
        choices.finish()
        bits = plan.get("float_key_bits")
        if not isinstance(bits, list) or len(bits) < len(self.return_source):
            raise Boundary("incomplete", "one_float_key_per_return_source_required")
        if len(bits) > len(self.return_source):
            raise Boundary("unsupported", "unused_float_keys")
        try:
            self.return_order, permutation = stable_float_order(self.return_source, bits)
        except ValueError as error:
            raise Boundary("unsupported", "float_key_domain", str(error)) from error
        self.sort_keys = [{"occurrence": x, "source_ordinal": i, "bits": bits[i],
                           "value": float32_key(bits[i])} for i, x in enumerate(self.return_source)]
        self.events.append({"operation": "supplied_stable_float32_sort",
                            "source_permutation": permutation})
        self.new_list("roster_return", self.return_order)

    def pool_prefix(self, plan):
        choices = Choices(plan.get("pool_indices"), self.draws, "pool")
        starting = [x for index in (3, 1, 2, 0) for x in self.starting[index]]
        script = [x for faction in FACTIONS for x in self.current(faction)]
        self.new_list("unique_starting_capture", starting, "array")
        self.new_list("unique_script_capture", script)
        self.events.append({"operation": "capture_before_unique_clear"})
        self.clear("unique_pool")
        remaining = [x for x in starting if not any(
            same_asset(self.occurrences[x], self.occurrences[y]) for y in script)]
        self.new_list("unique_after_deferred_remove_all", remaining, "array")
        self.events.append({"operation": "supplied_deferred_remove_all",
                            "removed": [x for x in starting if x not in remaining]})
        for real_type, cap in ((10, 4), (20, 1)):
            candidates = self.new_list("unique_candidates:" + str(real_type),
                [x for x in remaining if self.record(x)["bluffable"]
                 and self.record(x)["real_type"] == real_type])
            for _ in range(min(len(candidates), cap)):
                selected = candidates[choices.draw(len(candidates), "unique:direct")]
                self.add("unique_pool", selected)
                self.remove(candidates, selected, "unique:local")
        if len(self.lists["unique_pool"]["items"]) <= 1:
            selected = self.fallback_eligible[choices.draw(len(self.fallback_eligible), "unique:fallback")]
            self.add("unique_pool", selected)
        self.new_list("duplicate_script_capture", script)
        self.events.append({"operation": "fresh_duplicate_script_capture"})
        self.clear("duplicate_pool")
        bluffable = [x for x in script if self.record(x)["bluffable"]]
        self.new_list("duplicate_bluffable", bluffable)
        villagers = self.new_list("duplicate_candidates:10",
            [x for x in bluffable if self.record(x)["real_type"] == 10])
        outsiders = self.new_list("duplicate_candidates:20",
            [x for x in bluffable if self.record(x)["real_type"] == 20])
        # Native filter order is bluffable, Villager, Outcast, then discarded Good.
        self.new_list("discarded_duplicate_good_filter", [x for x in bluffable
                       if self.record(x)["starting_alignment"] == 10])
        if not villagers:
            choices.draw(0, "duplicate:unguarded_empty_villager")
        for _ in range(min(4, len(villagers))):
            selected = villagers[choices.draw(len(villagers), "duplicate:villager")]
            self.add("duplicate_pool", selected)
            self.remove(villagers, selected, "duplicate:local")
        if outsiders:
            selected = outsiders[choices.draw(len(outsiders), "duplicate:outsider")]
            self.add("duplicate_pool", selected)
            self.remove(outsiders, selected, "duplicate:local")
        choices.finish()
        self.boundary = {"kind": "before_first_init", "actor_occurrence": 0,
                         "data": self.return_order[0], "display_id": 5}


def callback_obligations(inputs, occurrences, actor, occurrence, role_kind):
    asset = occurrences[occurrence]["asset"]
    record = inputs["assets"][f'{asset["file_id"]}:{asset["path_id"]}']
    return [{"status": "pending_not_executed", "actor_occurrence": actor,
             "role_kind": role_kind, "data_occurrence": occurrence,
             "asset": copy.deepcopy(asset), "managed_class": copy.deepcopy(record["managed_class"]),
             "trigger": {"name": name, "value": value},
             "method_binding": {"status": "unresolved", "rva": None}}
            for name, value in (("Init", 3), ("AfterRoundStart", 7))]


def run_generation_case(inputs, plan):
    """One conditional choice history; return prefix state on every boundary."""
    immutable = digest(inputs)
    result = {"schema": OUTPUT_SCHEMA, "status": None, "reason": None, "detail": None,
              "input_digest": immutable, "conditioning": copy.deepcopy(inputs.get("domain")),
              "plan": copy.deepcopy(plan), "generation_stage": None, "before_first_init": None,
              "prefix_state": None, "fixture_contract": copy.deepcopy(FIXTURE_CONTRACT),
              "input_authentication_limit": "self-seal checks integrity only; original report binding requires load_inputs",
              "scope": "authored conditional reference; no native execution or weights"}
    model = None
    try:
        validate_inputs(inputs)
        capacity = plan.get("collection_capacity", 256)
        if type(capacity) is not int or capacity < 1:
            raise Boundary("unsupported", "capacity_not_positive_integer")
        model = Model(inputs, capacity)
        model.generation(plan)
        result["generation_stage"] = model.snapshot("generation")
        model.pool_prefix(plan)
        result["before_first_init"] = model.snapshot("before_first_init")
        result["status"] = "complete"
    except Boundary as error:
        result.update(status=error.status, reason=error.reason, detail=error.detail)
    except (KeyError, TypeError, ValueError) as error:
        result.update(status="unsupported", reason="input_schema_or_domain", detail=str(error))
    finally:
        if digest(inputs) != immutable:
            raise AssertionError("immutable input changed")
    if model is not None:
        result["prefix_state"] = model.snapshot("complete" if result["status"] == "complete" else "prefix")
    return result


def run_minion_selector(inputs, state, die, index, invocation_contract):
    """Separately conditioned ordinary Minion call; neither pool is consumed."""
    result = {"schema": "asset_minion_selector_reference_v0", "status": "unsupported",
              "input_state_digest": digest(state), "selected": None, "registered": False,
              "registration_attempted": False, "branch": None, "source_index": index,
              "draws": [], "state": copy.deepcopy(state), "selected_copy_obligations": []}
    try:
        validate_inputs(inputs)
        if state.get("stage") != "before_first_init":
            raise Boundary("unsupported", "selector_requires_before_first_init_state")
        if state.get("input_digest") != digest(inputs) or state.get("assets") != inputs["assets"]:
            raise Boundary("unsupported", "selector_input_state_binding")
        for occurrence in state["occurrences"].values():
            a = occurrence["asset"]
            if a["namespace"] != inputs["asset_namespace"] or f'{a["file_id"]}:{a["path_id"]}' not in inputs["assets"]:
                raise Boundary("unsupported", "selector_unresolved_asset")
        board = state["board"]
        actor = invocation_contract.get("actor_occurrence")
        if invocation_contract != {"kind": "supplied_ordinary_minion", "actor_occurrence": actor,
                                   "no_intervening_pool_writers": True} or type(actor) is not int or not 0 <= actor < len(board):
            raise Boundary("unsupported", "selector_invocation_contract")
        occurrence = board[actor]["data"]
        record = state["occurrences"][occurrence]["asset"]
        definition = inputs["assets"][f'{record["file_id"]}:{record["path_id"]}']
        if definition["managed_class"]["qualified_name"] != "Minion":
            raise Boundary("unsupported", "invocation_actor_is_not_ordinary_minion")
        draws = Choices([die, index], result["draws"], "minion_selector")
        draws.draw(10, "roll_die", minimum=1)
        branch = "duplicate_pool" if die <= 4 else "unique_pool"
        pool = state["lists"][branch]
        result["branch"] = branch
        position = draws.draw(len(pool["items"]), "selector:" + branch)
        selected = pool["items"][position]
        result["selected"] = selected
        draws.finish()
        result["selected_copy_obligations"] = callback_obligations(
            inputs, state["occurrences"], actor, selected, "selected_copied")
        if branch == "unique_pool":
            result["registration_attempted"] = True
            selected_record = inputs["assets"][f'{state["occurrences"][selected]["asset"]["file_id"]}:'
                                                 f'{state["occurrences"][selected]["asset"]["path_id"]}']
            faction = {10: "villagers", 20: "outsiders", 30: "minions", 100: "demons"}.get(selected_record["real_type"])
            if faction is None:
                raise Boundary("unsupported", "registration_real_type")
            current = result["state"]["lists"]["current:" + faction]
            if not any(same_asset(state["occurrences"][x], state["occurrences"][selected])
                       for x in current["items"]):
                capacity = invocation_contract.get("collection_capacity", 256)
                if len(current["items"]) >= capacity:
                    raise Boundary("capacity", "registration_capacity_before_add")
                placed = "selector:registration"
                if placed in result["state"]["occurrences"]:
                    raise Boundary("unsupported", "repeated_selector_requires_fresh_invocation_identity")
                result["state"]["occurrences"][placed] = {
                    "occurrence_id": placed, "asset": copy.deepcopy(state["occurrences"][selected]["asset"]),
                    "origin": {"destination": "current:" + faction}, "parent_occurrence": selected}
                current["items"].append(placed)
                current["version"] += 1
                result["registered"] = True
        result["status"] = "complete"
    except Boundary as error:
        result.update(status=error.status, reason=error.reason, detail=error.detail)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        result.update(status="unsupported", reason="selector_schema", detail=str(error))
    return result


def derived_choice_widths(inputs):
    """Derive the finite admitted N5 pool factor from input rows, never outputs."""
    validate_inputs(inputs)
    data = inputs["profile"]["data"]
    refs = data["townsfolks"] + data["outsiders"] + data["minions"] + data["townsfolks"]
    eligible = sum(bool(r["bluffable"]) and r["starting_alignment"] == 10
                   and r["real_type"] == 10 for r in
                   (inputs["assets"][f"{x[0]}:{x[1]}"] for x in refs))
    if eligible == 0:
        raise ValueError("empty fallback is a draw/index failure, outside legal enumeration domain")
    return {"generation": list(GENERATION_WIDTHS), "pool": [1, eligible, 4, 3, 2, 1]}


def enumerate_generation_cases(inputs, float_key_plans, case_capacity):
    """Finite supplied plans × integer histories; an explicit cap never prunes silently."""
    if type(case_capacity) is not int or case_capacity < 1:
        raise ValueError("positive explicit enumeration case capacity required")
    try:
        widths = derived_choice_widths(inputs)
    except (KeyError, TypeError, ValueError) as error:
        yield {"schema": "asset_reference_enumeration_boundary_v0", "status": "unsupported",
               "emitted": 0, "reason": "unsupported_n5_enumeration_domain", "detail": str(error)}
        return
    emitted = 0
    for bits in float_key_plans:
        try:
            stable_float_order(list(range(5)), bits)
        except (TypeError, ValueError) as error:
            yield {"schema": "asset_reference_enumeration_boundary_v0", "status": "unsupported",
                   "emitted": emitted, "reason": "unsupported_float_plan", "detail": str(error)}
            return
        for generation in itertools.product(*(range(x) for x in widths["generation"])):
            for pool in itertools.product(*(range(x) for x in widths["pool"])):
                if emitted == case_capacity:
                    yield {"schema": "asset_reference_enumeration_boundary_v0", "status": "capacity",
                           "emitted": emitted, "reason": "case_capacity_reached",
                           "next_plan": {"generation_indices": list(generation),
                                         "float_key_bits": list(bits), "pool_indices": list(pool)}}
                    return
                yield run_generation_case(inputs, {"generation_indices": list(generation),
                    "float_key_bits": list(bits), "pool_indices": list(pool)})
                emitted += 1
    yield {"schema": "asset_reference_enumeration_boundary_v0", "status": "complete",
           "emitted": emitted, "reason": "supplied_plan_domain_exhausted"}


def enumerate_minion_selectors(inputs, state, actor_occurrence, case_capacity):
    if type(case_capacity) is not int or case_capacity < 1:
        raise ValueError("positive explicit enumeration case capacity required")
    emitted = 0
    contract = {"kind": "supplied_ordinary_minion", "actor_occurrence": actor_occurrence,
                "no_intervening_pool_writers": True}
    for die in range(1, 11):
        branch = "duplicate_pool" if die <= 4 else "unique_pool"
        width = len(state["lists"][branch]["items"])
        # Empty pools remain a failure case, not an omitted branch.
        for index in range(max(1, width)):
            if emitted == case_capacity:
                yield {"status": "capacity", "emitted": emitted, "next_choice": [die, index]}
                return
            yield run_minion_selector(inputs, state, die, index, contract)
            emitted += 1
    yield {"status": "complete", "emitted": emitted, "reason": "conditioned_selector_domain_exhausted"}


def _synthetic_inputs():
    """Tiny authored test fixture, explicitly unrelated to the original asset proof."""
    inputs = {"schema": INPUT_SCHEMA, "build_id": "synthetic", "asset_namespace": "test",
              "assets": {}, "fixture_contract": copy.deepcopy(FIXTURE_CONTRACT),
              "input_origin": "authored_tiny_test_only", "provenance": {}}
    refs = [[0, i] for i in range(1, 6)]
    script = {"startingTownsfolks": refs, "startingOutsiders": [],
              "startingMinions": [[0, 6]], "startingDemons": [], "mustInclude": [],
              "characterCounts": [dict(zip(COUNT_FIELDS, N5_COUNTS))]}
    inputs["profile"] = {"path_id": 21674, "data": {
        "possibleScripts": [script], "possibleScriptsData": [], "mustInlcude": [],
        "alwaysInDeck": [], "townsfolks": refs, "outsiders": [], "minions": [[0, 6]],
        "demons": [], **{k: copy.deepcopy(script[k]) for k in
            ("startingTownsfolks", "startingOutsiders", "startingMinions", "startingDemons")}}}
    for i in range(1, 7):
        inputs["assets"][f"0:{i}"] = {"asset": asset_key("test", 0, i),
            "managed_class": {"qualified_name": "Minion" if i == 6 else "TestVillager",
                              "build_id": "synthetic", "type_def_index": i},
            "real_type": 30 if i == 6 else 10, "starting_alignment": 10,
            "bluffable": True, "public_name": None}
    inputs["domain"] = {"profile_id": 21674, "mode_enum": 0,
        "runtime_mode_class": "RoguelikeStandard", "ascension_index": 0,
        "village_index": 0, "accumulation": False, "starting_materialized": True,
        "initial_current_rosters": "empty", "initial_saved_rosters": "empty",
        "no_intervening_graph_writers": True}
    inputs["input_seal"] = digest(inputs)
    return inputs


def self_test():
    """Finite hand-authored cases; no audited output or native import is read."""
    checked = []
    occurrences = {key: {"asset": asset_key("test", 0, asset)}
                   for key, asset in (("a0", 1), ("b", 2), ("a2", 1))}
    items = ["a0", "b", "a2"]
    assert first_equal_remove(items, "a2", occurrences) == "a0" and items == ["b", "a2"]
    checked.append("duplicate_asset_first_equal_removal")
    assert stable_float_order(["m", "c", "a", "d", "b"],
        [0x3E800000, 0x3F000000, 0x3F000000, 0x3E000000, 0x3F000000])[0] == ["d", "m", "c", "a", "b"]
    checked.append("stable_float32_ties_follow_source_occurrence")
    assert stable_float_order(["negative_zero", "positive_zero"], [0x80000000, 0])[1] == [0, 1]
    checked.append("signed_zero_bits_preserved_equal_keys")
    for bits in (0x7FC00000, 0x7F800000, 0xBF800000):
        try:
            float32_key(bits)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported float key admitted")
    checked.append("nonfinite_or_outside_keys_rejected")
    inputs = _synthetic_inputs()
    plan = {"generation_indices": [0]*10, "float_key_bits": [0]*5,
            "pool_indices": [0]*6}
    report = run_generation_case(inputs, plan)
    assert report["status"] == "complete", report["reason"]
    stage = report["before_first_init"]
    assert [stage["lists"][x]["version"] for x in ("unique_pool", "duplicate_pool", "must_include")] == [3, 5, 0]
    assert [stage["lists"]["current:"+x]["version"] for x in FACTIONS] == [4, 0, 1, 0]
    assert all(not stage["lists"]["saved:"+x]["items"] for x in FACTIONS)
    assert stage["lists"]["source_inline:villagers"]["kind"] == "list"
    assert stage["lists"]["source_inline:villagers"]["version"] == 0
    assert stage["lists"]["source_starting:villagers"]["kind"] == "array"
    assert stage["lists"]["source_starting:villagers"]["items"] != stage["lists"]["source_inline:villagers"]["items"]
    selected = stage["lists"]["selected_inline:villagers"]["items"][0]
    materialized = stage["lists"]["temporary_starting:villagers"]["items"][0]
    assert stage["occurrences"][materialized]["parent_occurrence"] == selected
    assert stage["occurrences"][selected]["parent_occurrence"] == stage["lists"]["source_inline:villagers"]["items"][0]
    checked.append("separate_orders_pool_versions_empty_saved_rosters")
    checked.append("inline_list_stored_array_and_selected_materialization_provenance")
    assert derived_choice_widths(inputs)["pool"] == [1, 10, 4, 3, 2, 1]
    checked.append("fallback_enumeration_width_derived_from_synthetic_inputs")
    assert len(stage["pending_obligations"]) == 10 and all(
        r["status"] == "pending_not_executed" and r["managed_class"]["build_id"] == "synthetic"
        and r["asset"]["namespace"] == "test" for r in stage["pending_obligations"])
    checked.append("typed_per_actor_pending_callback_manifest")
    omitted = stage["lists"]["unique_pool"]["items"][0]
    assert stage["occurrences"][omitted]["asset"]["path_id"] == 5
    contract = {"kind": "supplied_ordinary_minion", "actor_occurrence": 0,
                "no_intervening_pool_writers": True}
    registered = run_minion_selector(inputs, stage, 5, 0, contract)
    assert registered["status"] == "complete" and registered["registered"]
    assert registered["state"]["lists"]["unique_pool"] == stage["lists"]["unique_pool"]
    assert registered["state"]["lists"]["current:villagers"]["version"] == 5
    assert len(registered["selected_copy_obligations"]) == 2
    assert all(r["actor_occurrence"] == 0 and r["asset"]["path_id"] == 5
               for r in registered["selected_copy_obligations"])
    checked.append("unique_registration_new_asset_without_pool_consumption")
    repeated = run_minion_selector(inputs, stage, 5, 1, contract)
    assert repeated["status"] == "complete" and repeated["registration_attempted"] and not repeated["registered"]
    duplicate = run_minion_selector(inputs, stage, 1, 0, contract)
    assert duplicate["status"] == "complete" and not duplicate["registration_attempted"]
    checked.append("membership_suppression_distinct_from_no_registration_branch")
    incomplete = run_generation_case(inputs, {**plan, "pool_indices": [0]})
    assert incomplete["status"] == "incomplete" and incomplete["generation_stage"]
    assert incomplete["prefix_state"]["lists"]["unique_pool"]["version"] == 2
    checked.append("missing_choice_retains_completed_generation_and_pool_prefix")
    over = run_generation_case(inputs, {**plan, "generation_indices": [0]*11})
    assert over["status"] == "unsupported" and over["reason"] == "unused_recorded_choices"
    checked.append("unused_choices_rejected")
    limited = list(enumerate_minion_selectors(inputs, stage, 0, 1))
    assert len(limited) == 2 and limited[-1]["status"] == "capacity" and limited[-1]["emitted"] == 1
    checked.append("finite_enumeration_capacity_is_explicit")
    changed = copy.deepcopy(inputs)
    changed["profile"]["data"]["townsfolks"] = [[1, 5]]
    changed["input_seal"] = digest({k: v for k, v in changed.items() if k != "input_seal"})
    assert run_generation_case(changed, plan)["status"] == "unsupported"
    checked.append("foreign_asset_file_not_silently_dropped")
    malformed = copy.deepcopy(inputs)
    malformed["profile"]["data"]["possibleScripts"][0]["startingTownsfolks"][0] = []
    malformed["profile"]["data"]["startingTownsfolks"][0] = []
    malformed["input_seal"] = digest({k: v for k, v in malformed.items() if k != "input_seal"})
    rejected = run_generation_case(malformed, plan)
    assert rejected["status"] == "unsupported" and rejected["detail"] == "PPtr must be [file_id,path_id]"
    checked.append("malformed_starting_pptr_rejected_before_asset_dereference")
    return {"status": "complete", "tiny_json_checks": len(checked), "checks": checked,
            "native_imports": 0, "native_instructions": 0,
            "audited_report_comparisons": 0, "large_enumerations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--ascension-report", type=Path)
    parser.add_argument("--character-report", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any((args.ascension_report, args.character_report, args.plan, args.output)):
            parser.error("tiny self-test cannot read or write external artifacts")
        print(json.dumps(self_test(), sort_keys=True))
        return
    if not all((args.ascension_report, args.character_report, args.plan, args.output)):
        parser.error("one case requires both input reports, a recorded plan and a fresh output")
    output = args.output.resolve()
    output.parent.resolve(strict=True)
    if output.exists():
        raise ValueError("output must be fresh")
    inputs = load_inputs(args.ascension_report, args.character_report)
    raw_plan = args.plan.resolve(strict=True).read_bytes()
    plan = json.loads(raw_plan.decode("utf-8"))
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = run_generation_case(inputs, plan)
    report["provenance"] = {"source_sha256": source_hash,
                            "inputs": inputs["provenance"],
                            "plan_sha256": hashlib.sha256(raw_plan).hexdigest()}
    # Preserve the complete explicit boundary; unsupported histories are outputs.
    output.write_text(json.dumps(report, sort_keys=True, ensure_ascii=True, indent=2)+"\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "reason": report["reason"],
                      "output_bytes": output.stat().st_size}, sort_keys=True))


if __name__ == "__main__":
    main()

