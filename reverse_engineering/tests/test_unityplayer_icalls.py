import importlib.util
from pathlib import Path
import struct
import sys
import unittest

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("unity_icalls", scripts / "audit_unityplayer_icalls.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
sys.path.pop(0)


class InternalCallTableTests(unittest.TestCase):
    def decode(self, functions=(0x1100, 0x1100), names=(0x1200, 0x1210), labels=None):
        labels = labels if labels is not None else {0x200: "Type::A", 0x210: "Type::B"}
        return audit.decode_pointer_pairs(struct.pack("<" + "Q" * len(functions), *functions),
                                           struct.pack("<" + "Q" * len(names), *names),
                                           0x1000, labels.__getitem__, lambda r: 0x100 <= r < 0x180)

    def test_shared_native_target_preserves_distinct_registered_names(self):
        self.assertEqual(self.decode(), {"Type::A": {"index": 0, "rva": 0x100},
                                        "Type::B": {"index": 1, "rva": 0x100}})

    def test_rejects_empty_mismatched_and_truncated_tables(self):
        for left, right in [(b"", b""), (b"12345678", b""), (b"x", b"x")]:
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                audit.decode_pointer_pairs(left, right, 0, lambda _: "Type::A", lambda _: True)

    def test_rejects_nonexecuting_and_out_of_image_pointers(self):
        for functions, names in [((0, 0x1100), (0x1200, 0x1210)),
                                 ((0x1200, 0x1100), (0x1200, 0x1210)),
                                 ((0x1100, 0x1100), (0, 0x1210)),
                                 ((0x100001000, 0x1100), (0x1200, 0x1210))]:
            with self.subTest(functions=functions, names=names), self.assertRaises(ValueError):
                self.decode(functions, names)

    def test_rejects_duplicate_or_invalid_names(self):
        for labels in [{0x200: "Type::A", 0x210: "Type::A"},
                       {0x200: "", 0x210: "Type::B"},
                       {0x200: "MissingSeparator", 0x210: "Type::B"}]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                self.decode(labels=labels)
