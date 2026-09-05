import importlib.util
from pathlib import Path
import struct
import sys
import unittest

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("unity_wait_consumer", scripts / "audit_unityplayer_wait_consumer.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
sys.path.pop(0)


class SyntheticWaitRecordTests(unittest.TestCase):
    def record(self, **overrides):
        values = dict(deadline=-0.0, frame=-(2**63), identity=0xFFFFFFFD,
                      generation=0xFFFFFFFF, mask=10, callback=0x1000, release=0x2000)
        values.update(overrides)
        return audit.wait_record(**values)

    def test_boundary_identity_cannot_turn_on_repeat_flag(self):
        record = self.record()
        self.assertEqual(len(record), 0x40)
        self.assertEqual(record[0x14], 0)
        self.assertEqual(struct.unpack_from("<Q", record, 0x10)[0], 0xFFFFFFFD)
        self.assertEqual(struct.unpack_from("<Q", record, 0x18)[0], 0xFFFFFFFD)
        self.assertEqual(struct.unpack_from("<q", record, 8)[0], -(2**63))
        self.assertEqual(struct.unpack_from("<III", record, 0x30), (0xFFFFFFFE, 10, 0xFFFFFFFF))

    def test_null_release_is_representable(self):
        self.assertEqual(struct.unpack_from("<Q", self.record(release=0), 0x28)[0], 0)

    def test_rejects_nonfinite_and_out_of_width_fields(self):
        for overrides in [dict(deadline=float("nan")), dict(deadline=float("inf")),
                          dict(frame=2**63), dict(frame=-(2**63) - 1),
                          dict(identity=2**32 - 2), dict(identity=-1),
                          dict(generation=2**32), dict(generation=-1),
                          dict(mask=2**32), dict(mask=-1)]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.record(**overrides)


if __name__ == "__main__":
    unittest.main()
