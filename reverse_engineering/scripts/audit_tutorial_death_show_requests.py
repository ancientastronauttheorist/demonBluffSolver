"""Values-only native death MoveNext Show-entry and return observations."""
import argparse
import itertools
import json
from pathlib import Path
from audit_tutorial_death_generators import Machine as DeathMachine, TARGETS
from audit_saved_game_storage_join import compact_trace


class Machine(DeathMachine):
    def setup_death(self):
        super().setup_death()
        self.show_entries = []
        self.pending_show = None
        self.initial_storage = None

    def physical_storage(self):
        records = [(self.actor, 'actor', 0x1B8), (self.controller, 'controller', 0x40),
                   (self.icon, 'transform', 0x80), (self.pivot, 'transform', 0x80),
                   (self.statuses, 'statuses', 0x80), (self.gameplay_type, 'class', 0x180),
                   (self.gameplay_static, 'static', 0x100)]
        for p in [self.status_list, self.rq(self.statuses + 0x18)]:
            a = self.rq(p + 0x10)
            records.extend([(p, 'list', 0x28), (a, 'array', 0x20 + self.rq(a + 0x18) * 4)])
        records.extend((p, 'routine', 0x80) for p in self.death_routines)
        assert len({p for p, _, _ in records}) == len(records)
        return [{'identity': p, 'kind': kind, 'bytes': list(self.u.mem_read(p, size))}
                for p, kind, size in records]

    def hook(self, uc, address, size, data):
        rva = address - self.base
        if self.death_ready and rva == 0x3A9330 and self.initial_storage is None:
            self.initial_storage = self.physical_storage()
        if self.death_ready and rva == 0x38E1A0:
            assert self.pending_show is None
            x = self.x
            controller, kind, pivot, method = [self.reg(r) for r in
                [x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9]]
            assert controller == self.controller and kind in [100, 45] and method == 0
            row = {'controller': controller, 'type': kind, 'pivot': pivot, 'method_info': method,
                   'routines': [self.death_routine_state(p, k) for p, k in self.death_routines.items()],
                   'controller_before': list(self.u.mem_read(controller, 0x40))}
            self.show_entries.append(row)
            self.pending_show = row
        if self.death_ready and rva in [0x3A9424, 0x3AAE5D] and self.pending_show is not None:
            row = self.pending_show
            row['controller_after'] = list(self.u.mem_read(self.controller, 0x40))
            row['return_rva'] = hex(rva)
            self.pending_show = None
        return super().hook(uc, address, size, data)


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases = []
    state = {'key': 'Tutorials', 'completedTutorials': ['old-t'], 'unlockedCharactersId': ['c']}
    for warm, initialized, gameplay, statuses in itertools.product(
            [False, True], [False, True], [10, 50, 60], [[], [10], [15, 10, 10], [55]]):
        options = dict(warm=warm, class_initialized=initialized, gameplay_state=gameplay, statuses=statuses)
        r = m.run_death(state, options)
        assert r['returned'] and m.pending_show is None
        expected = [] if gameplay == 50 else [100] + ([45] if 10 in statuses else [])
        assert [v['type'] for v in m.show_entries] == expected
        assert all(v['controller_before'] == v['controller_after'] for v in m.show_entries)
        cases.append({'input': options, 'death': r['final']['death'], 'waits': r['final']['generator']['waits'],
                      'initial_storage': m.initial_storage, 'final_storage': m.physical_storage(),
                      'wait_storage': [{'identity': p, 'kind': 'wait', 'bytes': list(m.u.mem_read(p, 0x80))} for p in m.waits],
                      'gameplay_class': m.gameplay_type,
                      'status_contains_method': next(m.death_tokens[a] for a, (_, n) in m.death_bindings.items()
                          if n == 'Method$System.Collections.Generic.List<ECharacterStatus>.Contains()'),
                      'show_entries': m.show_entries, 'transform_calls': [e['args'] for e in r['events']
                          if e['kind'] == 'character_icon_transform_service'],
                      'status_calls': [e['args'] for e in r['events'] if e['kind'] == 'death_status_contains_service'],
                      'controller_storage_retained': True})
    return {'build_id': r['build_id'] if 'build_id' in r else 'f530404b0f3f_807de4a83df4',
            'targets': m.death_targets, 'case_count': len(cases), 'cases': cases,
            'scope': 'Actual native Show entry ABI and return observations for 48 normal empty-note profiles; controller bytes 0..0x40 retain identity and values. No general Show inertness, real scheduling, UI or persistence replay claim.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('game_root', type=Path)
    p.add_argument('dumper_root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r = audit(a.game_root, a.dumper_root)
    a.output.write_text(json.dumps(compact_trace(r), indent=2) + '\n', encoding='utf-8')
    print(r['case_count'])
