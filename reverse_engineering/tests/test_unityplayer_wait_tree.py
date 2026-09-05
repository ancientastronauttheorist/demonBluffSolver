import importlib.util
from pathlib import Path
import struct
import sys
import unittest

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("unity_wait_tree", scripts / "audit_unityplayer_wait_tree.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
sys.path.pop(0)


class TreeOracleTests(unittest.TestCase):
    def setUp(self):
        self.memory = bytearray(0x500)
        self.container, self.head, self.root, self.right = 0x20, 0x100, 0x200, 0x300
        self.records = {self.root: audit.make_record(-0.0, 7),
                        self.right: audit.make_record(0.0, 3)}
        self.write(self.container, struct.pack("<QQ", self.head, 2))
        self.node(self.head, self.root, self.root, self.right, 1, 1)
        self.node(self.root, self.head, self.head, self.right, 1, 0)
        self.node(self.right, self.head, self.root, self.head, 0, 0)
        for address, record in self.records.items():
            self.write(address + 0x20, record)

    def write(self, address, data):
        self.memory[address:address + len(data)] = data

    def node(self, address, left, parent, right, color, sentinel):
        self.write(address, struct.pack("<QQQBB", left, parent, right, color, sentinel))

    def validate(self, expected=(7, 3)):
        return audit.validate_tree(lambda a, n: self.memory[a:a + n], self.container,
                                   self.head, self.records, list(expected))

    def test_equal_signed_zero_keys_keep_occurrence_order_not_identity_order(self):
        self.assertEqual(self.validate(), 2)
        with self.assertRaisesRegex(ValueError, "stable deadline order"):
            self.validate((3, 7))

    def test_rejects_surviving_payload_corruption(self):
        self.memory[self.right + 0x5F] ^= 1
        with self.assertRaisesRegex(ValueError, "payload changed"):
            self.validate()

    def test_rejects_broken_balancing_parent_count_and_extrema(self):
        mutations = [(self.right + 0x18, b"\x01", "black height"),
                     (self.root + 0x18, b"\x00", "root is not black"),
                     (self.right + 8, struct.pack("<Q", self.head), "parent mismatch"),
                     (self.container + 8, struct.pack("<Q", 1), "count mismatch"),
                     (self.head, struct.pack("<Q", self.right), "minimum mismatch")]
        for address, data, message in mutations:
            with self.subTest(message=message):
                old = bytes(self.memory[address:address + len(data)])
                self.write(address, data)
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()
                self.write(address, old)

    def test_rejects_cycle_before_recursing_forever(self):
        self.write(self.root, struct.pack("<Q", self.root))
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.validate()

    def test_empty_sentinel_and_nonfinite_input_boundary(self):
        self.node(self.head, self.head, self.head, self.head, 1, 1)
        self.write(self.container + 8, bytes(8))
        self.records = {}
        self.assertEqual(self.validate(()), 1)
        for deadline in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                audit.make_record(deadline, 0)


if __name__ == "__main__":
    unittest.main()
