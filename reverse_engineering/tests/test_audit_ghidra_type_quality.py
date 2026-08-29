from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIRECTORY))

import audit_ghidra_type_quality as audit  # noqa: E402


class AuditGhidraTypeQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.baseline = self.root / "private-baseline"
        self.typed = self.root / "private-typed"
        self.baseline.mkdir()
        self.typed.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def summary(count: int) -> dict[str, int | bool]:
        return {
            "requested": count,
            "processed": count,
            "exported": count,
            "failed": 0,
            "cancelled": False,
        }

    def write_exports(
        self,
        directory: Path,
        bodies: dict[str, str],
        *,
        summary: dict[str, int | bool] | None = None,
    ) -> None:
        for name, body in bodies.items():
            (directory / name).write_text(body, encoding="utf-8", newline="\n")
        (directory / audit.SUMMARY_NAME).write_text(
            json.dumps(summary if summary is not None else self.summary(len(bodies))),
            encoding="utf-8",
        )

    def improvement_fixture(self) -> None:
        self.write_exports(
            self.baseline,
            {
                "Alpha.c": (
                    "undefined8 Alpha(longlong param_1) {\n"
                    "  /* WARNING: Could not recover jumptable */\n"
                    "  return *(undefined8 *)(param_1 + 0x18);\n"
                    "}\n"
                ),
                "Zulu.c": "void Zulu(undefined param_1) { FUN_180001000(param_1); }\n",
            },
        )
        self.write_exports(
            self.typed,
            {
                "Alpha.c": (
                    "int32_t Alpha(Alpha_o* __this, const MethodInfo* method) {\n"
                    "  return __this->fields.value;\n"
                    "}\n"
                ),
                "Zulu.c": "void Zulu(Zulu_o* __this, const MethodInfo* method) { }\n",
            },
        )

    def test_reports_deterministic_public_safe_improvements(self) -> None:
        self.improvement_fixture()

        report = audit.build_report(self.baseline, self.typed, check=True)
        repeated = audit.build_report(self.baseline, self.typed, check=True)
        encoded = audit.canonical_json(report)

        self.assertEqual(encoded, audit.canonical_json(repeated))
        self.assertEqual(
            [target["filename"] for target in report["targets"]],
            ["Alpha.c", "Zulu.c"],
        )
        self.assertEqual(report["target_count"], 2)
        self.assertTrue(report["check"]["passed"])
        self.assertGreater(report["aggregate"]["quality_delta"]["unresolved_type_tokens"], 0)
        self.assertGreater(report["aggregate"]["quality_delta"]["raw_field_offset_accesses"], 0)
        self.assertGreater(report["aggregate"]["typed"]["named_struct_field_accesses"], 0)

        self.assertNotIn(str(self.root), encoded)
        self.assertNotIn("Could not recover jumptable", encoded)
        self.assertNotIn("return __this->fields.value", encoded)

    def test_cli_check_writes_json_and_passes(self) -> None:
        self.improvement_fixture()
        output = self.root / "reports" / "quality.json"

        exit_code = audit.main(
            [
                "--baseline",
                str(self.baseline),
                "--typed",
                str(self.typed),
                "--output",
                str(output),
                "--check",
            ]
        )

        self.assertEqual(exit_code, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(report["check"]["enabled"])
        self.assertTrue(report["check"]["passed"])

    def test_rejects_filename_and_summary_mismatches(self) -> None:
        self.write_exports(self.baseline, {"Alpha.c": "void Alpha(void) {}\n"})
        self.write_exports(self.typed, {"Beta.c": "void Beta(void) {}\n"})
        with self.assertRaisesRegex(audit.AuditError, "filenames do not match"):
            audit.build_report(self.baseline, self.typed)

        (self.typed / "Beta.c").unlink()
        (self.typed / "Alpha.c").write_text("void Alpha(void) {}\n", encoding="utf-8")
        (self.typed / audit.SUMMARY_NAME).write_text(
            json.dumps(self.summary(2)), encoding="utf-8"
        )
        with self.assertRaisesRegex(audit.AuditError, "summaries do not match"):
            audit.build_report(self.baseline, self.typed)

    def test_check_detects_aggregate_regression(self) -> None:
        self.write_exports(self.baseline, {"Alpha.c": "void Alpha(void) {}\n"})
        self.write_exports(
            self.typed,
            {"Alpha.c": "undefined8 Alpha(undefined8 param_1) { return param_1; }\n"},
        )

        report = audit.build_report(self.baseline, self.typed, check=True)

        self.assertFalse(report["check"]["passed"])
        self.assertEqual(
            [regression["metric"] for regression in report["check"]["regressions"]],
            ["unresolved_type_tokens"],
        )

    def test_rejects_output_path_that_would_overwrite_an_export(self) -> None:
        self.improvement_fixture()
        protected = self.baseline / "Alpha.c"
        with self.assertRaisesRegex(audit.AuditError, "distinct .json"):
            audit.write_report(protected, "{}\n", [protected])


if __name__ == "__main__":
    unittest.main()
