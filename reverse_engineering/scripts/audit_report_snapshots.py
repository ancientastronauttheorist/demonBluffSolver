"""Lossless full-snapshot interning for authored offline audit JSON reports.

No native data is produced here. All semantic/prefix checks run before pooling;
the codec retains every field and supports exact expanded-value verification.
"""
import hashlib
import json
from copy import deepcopy


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')


def pool_snapshots(report):
    assert 'snapshot_blobs' not in report and 'snapshot_encoding' not in report
    blobs = {}
    def encode(value):
        if isinstance(value, list): return [encode(v) for v in value]
        if not isinstance(value, dict): return value
        result = {}
        for key, item in value.items():
            if key in ['initial', 'final', 'snapshot'] and isinstance(item, dict):
                digest = hashlib.sha256(canonical(item)).hexdigest()
                assert digest not in blobs or blobs[digest] == item
                if digest not in blobs: blobs[digest] = deepcopy(item)
                result[key] = {'snapshot_sha256': digest}
            else: result[key] = encode(item)
        return result
    pooled = encode(report)
    pooled['snapshot_encoding'] = 'sha256-full-authored-snapshot-v1'
    pooled['snapshot_blobs'] = blobs
    assert expand_snapshots(pooled) == report
    return pooled


def expand_snapshots(report):
    assert report['snapshot_encoding'] == 'sha256-full-authored-snapshot-v1'
    blobs = report['snapshot_blobs']
    for digest, value in blobs.items():
        assert isinstance(value, dict) and hashlib.sha256(canonical(value)).hexdigest() == digest
    def decode(value):
        if isinstance(value, list): return [decode(v) for v in value]
        if not isinstance(value, dict): return value
        if set(value) == {'snapshot_sha256'}:
            # Deep-copy the interned snapshot so two decoded service-entry
            # records do not accidentally share mutable diagnostic storage.
            return decode(blobs[value['snapshot_sha256']])
        return {k: decode(v) for k, v in value.items()}
    return decode({k: v for k,v in report.items() if k not in ['snapshot_encoding','snapshot_blobs']})
