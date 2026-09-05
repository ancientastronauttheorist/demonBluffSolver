import importlib.util
from pathlib import Path
import struct
import sys
import unittest

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("unity_phases", scripts / "audit_unityplayer_phases.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
sys.path.pop(0)


class PhaseJoinTests(unittest.TestCase):
    def setUp(self):
        self.buffer = bytearray(3 * 0x68)
        struct.pack_into("<Q", self.buffer, 0x28, 3)
        self.phases = [{"cell": 0x20, "cache": 0x60}, {"cell": 0x28, "cache": 0x68}]
        for index, phase in enumerate(self.phases, 1):
            struct.pack_into("<Q", self.buffer, index * 0x68 + 0x30, 0x5000 + phase["cache"])
            struct.pack_into("<Q", self.buffer, index * 0x68 + 0x58, 0x1000 + phase["cell"])

    def decode(self):
        return audit.find_phase_nodes(self.buffer, 0x1000, 0x5000, self.phases)

    def test_joins_distinct_slots_to_their_type_tags(self):
        self.assertEqual(self.decode(), (3, [1, 2]))

    def test_rejects_wrong_type_even_when_callback_cell_matches(self):
        struct.pack_into("<Q", self.buffer, 0x68 + 0x30, 0x5068)
        with self.assertRaisesRegex(ValueError, "wrong type-cache"):
            self.decode()

    def test_rejects_missing_or_duplicate_callback_cells(self):
        original = bytes(self.buffer)
        for replacement in (0, 0x1020):
            with self.subTest(replacement=replacement):
                self.buffer[:] = original
                if replacement:
                    struct.pack_into("<Q", self.buffer, 2 * 0x68 + 0x30, 0x5060)
                struct.pack_into("<Q", self.buffer, 2 * 0x68 + 0x58, replacement)
                with self.assertRaisesRegex(ValueError, "missing or duplicated"):
                    self.decode()

    def test_rejects_truncation_and_invalid_counts(self):
        for buffer in (b"", bytes(0x68), self.buffer[:-1]):
            with self.subTest(size=len(buffer)), self.assertRaises(ValueError):
                audit.find_phase_nodes(buffer, 0x1000, 0x5000, self.phases)
        struct.pack_into("<Q", self.buffer, 0x28, 257)
        with self.assertRaises(ValueError):
            self.decode()


if __name__ == "__main__":
    unittest.main()
