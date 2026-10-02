import importlib.util
import json
from pathlib import Path
import unittest

SCRIPT=Path(__file__).parents[1]/'scripts/audit_report_snapshots.py'
SPEC=importlib.util.spec_from_file_location('audit_report_snapshots',SCRIPT)
CODEC=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CODEC)


class SnapshotCodecTests(unittest.TestCase):
    def sample(self):
        state={'identity':2**64-1,'bytes':'00ff','nested':{'refs':[None,'same',2]},'ledger':[]}
        return {'initial':state,'events':[{'kind':'service','snapshot':state}],
                'final':state,'cases':[{'initial':{'identity':0,'bytes':'0000'}}]}

    def test_full_round_trip_retains_duplicates_order_and_numeric_bits(self):
        original=self.sample();pooled=CODEC.pool_snapshots(original)
        self.assertEqual(2,len(pooled['snapshot_blobs']))
        self.assertEqual(original,CODEC.expand_snapshots(json.loads(json.dumps(pooled))))
        self.assertEqual(pooled['initial'],pooled['events'][0]['snapshot'])
        self.assertEqual(original,self.sample())

    def test_expansion_has_independent_mutable_storage(self):
        expanded=CODEC.expand_snapshots(CODEC.pool_snapshots(self.sample()))
        expanded['initial']['nested']['refs'].append('changed')
        self.assertNotEqual(expanded['initial'],expanded['final'])
        self.assertEqual(3,len(expanded['events'][0]['snapshot']['nested']['refs']))

    def test_pooled_blob_mutation_does_not_change_input(self):
        original=self.sample();pooled=CODEC.pool_snapshots(original)
        digest=pooled['initial']['snapshot_sha256']
        pooled['snapshot_blobs'][digest]['nested']['refs'].append('changed')
        self.assertEqual(original,self.sample())

    def test_corrupt_blob_or_missing_reference_is_rejected(self):
        pooled=CODEC.pool_snapshots(self.sample())
        digest=pooled['initial']['snapshot_sha256']
        pooled['snapshot_blobs'][digest]['identity']=3
        with self.assertRaises(AssertionError):CODEC.expand_snapshots(pooled)
        pooled=CODEC.pool_snapshots(self.sample())
        pooled['initial']['snapshot_sha256']='missing'
        with self.assertRaises(KeyError):CODEC.expand_snapshots(pooled)
