from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIRECTORY))

import normalize_il2cpp_header as normalizer  # noqa: E402


BUILD_ID = "0123456789ab_abcdef012345"


class NormalizeIl2CppHeaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.header = self.root / "il2cpp.h"
        self.header_bytes = (
            b"struct Base_Fields {\r\n"
            b"\tuint32_t first;\r\n"
            b"};\r\n"
            b"struct Derived_Fields : Base_Fields {\r\n"
            b"\tint32_t second;\r\n"
            b"};\r\n"
            b"struct __declspec(align(8)) Aligned_Fields {\r\n"
            b"\tbool enabled;\r\n"
            b"};\r\n"
        )
        self.header.write_bytes(self.header_bytes)
        self.extraction = self.root / "extraction.json"
        self.extraction.write_text(
            json.dumps(
                {
                    "build_id": BUILD_ID,
                    "outputs": {
                        "il2cpp_h": {
                            "sha256": hashlib.sha256(self.header_bytes).hexdigest(),
                            "size": len(self.header_bytes),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_targets(self, name: str, signatures: list[str]) -> Path:
        path = self.root / name
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "build_id": BUILD_ID,
                    "functions": [
                        {"name": f"target-{index}", "signature": signature}
                        for index, signature in enumerate(signatures)
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_normalizes_and_unions_prototypes_deterministically(self) -> None:
        first = self.write_targets(
            "z_targets.json",
            ["void Shared (Derived_Fields* value);", "int32_t Zed (void);"],
        )
        second = self.write_targets(
            "a_targets.json",
            ["bool Alpha (void);", "void Shared (Derived_Fields* value);"],
        )
        output = self.root / "output"

        result = normalizer.build_outputs(
            self.header, [first, second], self.extraction, output
        )
        reversed_output = self.root / "reversed-output"
        reversed_result = normalizer.build_outputs(
            self.header, [second, first], self.extraction, reversed_output
        )

        normalized_bytes = result["normalized_header"].read_bytes()
        self.assertNotIn(b"\r", normalized_bytes)
        normalized = normalized_bytes.decode("utf-8")
        self.assertIn("typedef unsigned long long uintptr_t;", normalized)
        self.assertIn("struct Derived_Fields {\n\tstruct Base_Fields super;", normalized)
        self.assertNotIn("Derived_Fields : Base_Fields", normalized)

        prototypes = result["prototype_header"].read_text(encoding="utf-8")
        self.assertLess(prototypes.index("Alpha ("), prototypes.index("Shared ("))
        self.assertLess(prototypes.index("Shared ("), prototypes.index("Zed ("))
        self.assertEqual(prototypes.count("Shared ("), 1)

        manifest = json.loads(result["alignment_manifest"].read_text(encoding="utf-8"))
        self.assertEqual(manifest["names"], ["Aligned_Fields"])
        self.assertEqual(manifest["alignment_count"], 1)
        self.assertEqual(manifest["inheritance_rewrite_count"], 1)
        self.assertEqual(manifest["prototype_count"], 3)
        self.assertEqual(manifest["prototype_names"], ["Alpha", "Shared", "Zed"])
        self.assertEqual(
            result["prototype_header"].read_bytes(),
            reversed_result["prototype_header"].read_bytes(),
        )
        self.assertEqual(
            result["alignment_manifest"].read_bytes(),
            reversed_result["alignment_manifest"].read_bytes(),
        )
        self.assertEqual(
            result["normalization_summary"].read_bytes(),
            reversed_result["normalization_summary"].read_bytes(),
        )

    def test_rejects_conflicting_prototype_identifiers(self) -> None:
        first = self.write_targets("one.json", ["void Same (int32_t value);"])
        second = self.write_targets("two.json", ["void Same (bool value);"])
        with self.assertRaisesRegex(normalizer.NormalizationError, "Conflicting signatures"):
            normalizer.build_outputs(
                self.header, [first, second], self.extraction, self.root / "output"
            )

    def test_rejects_unmanifested_header(self) -> None:
        target = self.write_targets("target.json", ["void Valid (void);"])
        self.header.write_bytes(self.header_bytes + b"/* changed */\n")
        with self.assertRaisesRegex(normalizer.NormalizationError, "does not match"):
            normalizer.build_outputs(
                self.header, [target], self.extraction, self.root / "output"
            )

    def test_known_header_baseline_is_enforced(self) -> None:
        target = self.write_targets("target.json", ["void Valid (void);"])
        source_hash = hashlib.sha256(self.header_bytes).hexdigest().upper()
        normalizer.HEADER_BASELINES[source_hash] = {
            "alignment_count": 99,
            "inheritance_rewrite_count": 99,
        }
        try:
            with self.assertRaisesRegex(normalizer.NormalizationError, "baseline changed"):
                normalizer.build_outputs(
                    self.header, [target], self.extraction, self.root / "output"
                )
        finally:
            del normalizer.HEADER_BASELINES[source_hash]


if __name__ == "__main__":
    unittest.main()
