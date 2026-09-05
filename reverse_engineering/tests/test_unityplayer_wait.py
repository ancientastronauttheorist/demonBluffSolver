import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("unity_wait", Path(__file__).parents[1] / "scripts" / "audit_unityplayer_wait.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class NativeWaitAuditTests(unittest.TestCase):
    def test_fingerprint_rejects_mutated_input(self):
        expected = "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"
        self.assertEqual(audit.verify_fingerprint(b"abc", expected.lower()), expected)
        with self.assertRaises(ValueError):
            audit.verify_fingerprint(b"abd", expected)

    def test_signed_rip_reference_uses_instruction_end(self):
        self.assertEqual(audit.relative_target(0x1000, 7, 0x80), 0x1087)
        self.assertEqual(audit.relative_target(0x1000, 7, -0x80), 0xF87)

    def test_reference_rejects_underflow_overflow_and_invalid_instruction(self):
        for args in [(0, 1, -2), (0xFFFFFFFF, 1, 0), (-1, 7, 0), (0, 0, 0)]:
            with self.assertRaises(ValueError):
                audit.relative_target(*args)
