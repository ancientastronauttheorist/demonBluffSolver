"""Native Character hover entry and description-hide caller, explicit services."""
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path
from audit_character_assets import BUILD
from audit_character_presentation_helpers import Machine as PresentationMachine

TARGETS = {0x3674F0: ('OnHover', 'tdi5487.m0054', 0x3674F8),
           0x365340: ('HideDescription', 'tdi5487.m0062', 0x365458)}


class Machine(PresentationMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        self.description_ready = False
        super().__init__(game_root, dumper_root)
        self.description_targets, self.description_flags = [], set()
        self.description_instructions = {}
        for start, (name, method_id, end) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Name'] == 'Character$$'+name and r['Address'] == start]
            assert len(rows) == 1
            assert rows[0]['Signature'] == f'void Character__{name} (Character_o* __this, const MethodInfo* method);'
            assert rows[0]['TypeSignature'] == 'vii'
            self.description_targets.append(dict(rows[0], method_id=method_id))
            family = [(e.struct.BeginAddress, e.struct.EndAddress) for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress <= start < e.struct.EndAddress]
            assert family == ([(start, end+1)] if name == 'HideDescription' else [])
            following = min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > start)
            assert following == (0x365460 if name == 'HideDescription' else 0x367500)
            assert self.pe.get_data(end, following-end) == bytes([0xCC])*(following-end)
            decoded = list(self.cs.disasm(self.pe.get_data(start, end-start), start))
            assert sum(i.size for i in decoded) == end-start
            self.description_instructions.update({i.address:i for i in decoded})
            for i in decoded:
                for op in i.operands:
                    if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                        slot = i.address+i.size+op.mem.disp
                        row = [r for r in self.metadata['ScriptMetadata'] if r['Address'] == slot]
                        if row:
                            assert len(row) == 1
                            name = row[0]['Name']
                            if name not in self.bindings:
                                assert name == 'Characters_TypeInfo'
                                self.bindings[name] = self.arena+0x9000
                                self.ids[self.arena+0x9000] = name
                            token = self.bindings[name]
                            self.metadata_slots[self.base+slot] = token
                            self.q(self.base+slot, token)
                        elif i.mnemonic == 'cmp' and i.operands[0].size == 1:
                            self.description_flags.add(slot)
        self.instructions.update(self.description_instructions)
        fields = {
            'public class Character : MonoBehaviour, ICard // TypeDefIndex: 5487': ['bool leftAct; // 0xB0','Acted leftActed; // 0xB8','Acted acteds; // 0xA8','bool hover; // 0x190','string savedAct; // 0x198'],
            'public class Acted : MonoBehaviour // TypeDefIndex: 5477': ['ActedVersion acted; // 0x20'],
            'public class Characters : MonoBehaviour // TypeDefIndex: 5505': ['static Characters Instance; // 0x0'],
            'public static class UIEvents // TypeDefIndex: 5523': ['Action OnHideHint; // 0x40','Action OnHideCustomHint; // 0x98'],
        }
        for declaration, expected in fields.items():
            block = re.search(r'^'+re.escape(declaration)+r'\s*\{(.*?)(?=\n// Namespace:)', self.dump, re.M|re.S)
            assert block and all(f in block[1] for f in expected), declaration
        self.description_fields = fields
        self.game_services = {0x35DD10:'Acted$$Act',0x369D60:'Characters$$DisableHighlightAll'}
        for address,name in {**self.game_services,0x1C7F4A0:'UnityEngine.MonoBehaviour$$StopAllCoroutines',0x1C79FD0:'UnityEngine.Component$$get_gameObject',0x1C7D810:'UnityEngine.GameObject$$SetActive'}.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Name'] == name and r['Address'] == address]
            assert len(rows) == 1
        checks = {0x3674F0:('mov','byte ptr [rcx + 0x190], 1'),0x3674F7:('ret',''),
            0x365371:('cmp','byte ptr [rbx + 0xb0], 0'),0x36537F:('mov','rdi, qword ptr [rbx + 0xb8]'),
            0x365399:('mov','rcx, qword ptr [rdi + 0x20]'),0x3653D3:('mov','rdx, qword ptr [rbx + 0x198]'),
            0x36540D:('mov','r8, qword ptr [rax + 0x98]'),0x365425:('mov','rcx, qword ptr [rip + 0x2380154]'),
            0x365433:('mov','r8, qword ptr [rax + 0x40]'),0x365444:('call','qword ptr [r8 + 0x18]')}
        assert all((self.description_instructions[a].mnemonic,self.description_instructions[a].op_str)==v for a,v in checks.items())
        self.description_assertions = len(checks)
        extra = ['left_acted','left_version','left_game','acted','speech','characters_static','characters','hide_callback','hint_callback','hint_replacement','action_target','action_method']
        self.p.update({name:self.arena+0x30000+i*0x1000 for i,name in enumerate(extra)})
        self.ids.update({p:name for name,p in self.p.items()})
        self.action_gateway = self.stop+0x200

    def prepare(self, options):
        self.description_ready = False
        super().prepare(options)
        a = self.p['actor']
        self.u.mem_write(a+0xB0,bytes([options.get('left_bits',1)]))
        for offset,name in [(0xB8,'left_acted'),(0xA8,'acted'),(0x198,'speech')]:
            self.q(a+offset,0 if options.get('null_'+name) else self.p[name])
        self.q(self.p['left_acted']+0x20,0 if options.get('null_left_version') else self.p['left_version'])
        self.q(self.bindings['Characters_TypeInfo']+0xB8,self.p['characters_static'])
        self.q(self.p['characters_static'],0 if options.get('null_characters') else self.p['characters'])
        self.q(self.p['ui_static']+0x98, self.p['hide_callback'] if options.get('hide_callback',True) else 0)
        self.q(self.p['ui_static']+0x40, self.p['hint_callback'] if options.get('hint_callback',True) else 0)
        for name in ['hide_callback','hint_callback','hint_replacement']:
            p = self.p[name]
            self.q(p+0x18,self.action_gateway);self.q(p+0x28,self.p['action_method'])
            self.q(p+0x40,0 if options.get('null_action_target') else self.p['left_acted'] if options.get('alias_action_left') else self.p['action_target'])
        for flag in self.description_flags:
            self.u.mem_write(self.base+flag,bytes([options.get('warm_byte',1) if not options.get('cold') else 0]))
        self.hidden_game_active = True
        self.stopped_components,self.restored_speech,self.highlight_clears,self.action_calls = [],[],0,[]
        self.description_ready = True

    def snapshot(self):
        out = super().snapshot()
        if self.description_ready:
            a = self.p['actor']
            out['description'] = {'left_bits':self.byte(a+0xB0),
                'actor_bytes_hex':bytes(self.u.mem_read(a,0x200)).hex(),
                **{name:self.oid(self.rq(a+offset)) for name,offset in [('left_acted',0xB8),('acted',0xA8),('speech',0x198)]},
                'left_version':self.oid(self.rq(self.p['left_acted']+0x20)),
                'characters':self.oid(self.rq(self.p['characters_static'])),
                'characters_class_word':self.rd(self.bindings['Characters_TypeInfo']+0xE0),
                'metadata_flags':{hex(f):self.byte(self.base+f) for f in sorted(self.description_flags)},
                'hide_callback':self.oid(self.rq(self.p['ui_static']+0x98)),
                'hint_callback':self.oid(self.rq(self.p['ui_static']+0x40)),
                'supplied_game_active':self.hidden_game_active,'stopped_components':self.stopped_components.copy(),
                'restored_speech':self.restored_speech.copy(),'highlight_clears':self.highlight_clears,
                'action_calls':self.action_calls.copy()}
        return out

    def description_mutation(self, phase):
        if self.options.get('mutation_phase') != phase: return
        mutation = self.options['mutation']
        if mutation in ['clear_left','clear_acted','clear_speech']:
            offset = {'clear_left':0xB8,'clear_acted':0xA8,'clear_speech':0x198}[mutation]
            self.q(self.p['actor']+offset,0);self.authored_offsets.update(range(offset,offset+8))
        elif mutation == 'clear_hint':self.q(self.p['ui_static']+0x40,0)
        elif mutation == 'replace_hint':self.q(self.p['ui_static']+0x40,self.p['hint_replacement'])
        elif mutation == 'clear_hide':self.q(self.p['ui_static']+0x98,0)
        else:raise AssertionError(mutation)

    def hook(self, uc, address, size, data):
        rva,x = address-self.base,self.x
        if not self.description_ready:return super().hook(uc,address,size,data)
        self.executed.add(rva)
        if rva in self.description_instructions:return
        cx,dx,r8 = [self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8']]
        if rva == 0x1C7F4A0:
            assert cx == self.p['left_acted'] and dx == 0
            if self.event('supplied_stop_all_coroutines',[self.oid(cx)]):
                self.stopped_components.append(self.oid(cx));self.description_mutation('stop');self.ret()
        elif rva == 0x1C79FD0:
            assert cx == self.p['left_version'] and dx == 0
            result = 0 if self.options.get('null_game_result') else self.p['left_game']
            if self.event('supplied_game_object',[self.oid(cx),self.oid(result)]):self.ret(result)
        elif rva == 0x1C7D810:
            assert cx == self.p['left_game'] and dx == r8 == 0
            if self.event('supplied_set_active',[self.oid(cx),False]):
                self.hidden_game_active=False;self.description_mutation('active');self.ret()
        elif rva == 0x35DD10:
            assert cx == self.p['acted'] and dx in [0,self.p['speech']] and r8 == 0
            if self.event('supplied_acted_act',[self.oid(cx),self.oid(dx)]):
                self.restored_speech.append(self.oid(dx));self.ret()
        elif rva == 0x369D60:
            assert cx == self.p['characters'] and dx == 0
            if self.event('supplied_disable_highlight_all',[self.oid(cx)]):
                self.highlight_clears+=1;self.description_mutation('highlight');self.ret()
        elif address == self.action_gateway:
            # R8 is the actual loaded delegate in both native Action calls.
            cb = r8
            assert cb in [self.p[n] for n in ['hide_callback','hint_callback','hint_replacement']]
            assert cx == self.rq(cb+0x40) and dx == self.rq(cb+0x28)
            slot = 0x98 if cb == self.p['hide_callback'] else 0x40
            assert self.rq(self.p['ui_static']+slot) == cb
            kind = 'hide' if cb == self.p['hide_callback'] else 'hint'
            row = {'delegate':self.oid(cb),'target':self.oid(cx),'method':self.oid(dx)}
            if self.event('supplied_'+kind+'_action',[row]):
                self.action_calls.append(row);self.description_mutation(kind);self.ret()
        else:return super().hook(uc,address,size,data)

    def run_description(self,name,options=None,retained=False):
        if not retained:self.prepare(options or {})
        else:self.options,self.error=options or {},None
        self.authored_offsets=set();before=bytes(self.u.mem_read(self.p['actor'],0x200));initial=self.snapshot()
        old=len(self.events)
        address=next(a for a,(n,_,_) in TARGETS.items() if n==name)
        returned=self.invoke(address)
        after=bytes(self.u.mem_read(self.p['actor'],0x200))
        allowed=self.authored_offsets | ({0x190} if name=='OnHover' else set())
        assert all(i in allowed or before[i]==after[i] for i in range(0x200))
        if name=='OnHover':assert returned and self.byte(self.p['actor']+0x190)==1 and len(self.events)==old
        if returned and name=='HideDescription' and not self.options.get('mutation_phase'):
            left=initial['description']['left_bits']!=0
            assert self.hidden_game_active == (not left)
            assert self.highlight_clears == initial['description']['highlight_clears']+1
            assert self.restored_speech[len(initial['description']['restored_speech']):] == ([initial['description']['speech']] if left else [])
        return {'method':name,'options':self.options.copy(),'returned':returned,'error':self.error,
                'initial':initial,'events':self.events[old:].copy(),'final':self.snapshot(),'other_actor_bytes_retained':True}


def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root);cases,sequences,baselines,failures=[],[],[],[]
    for bits,hide,hint,cold in itertools.product([0,1,0x80,0xFF],[False,True],[False,True],[False,True]):
        cases.append(m.run_description('HideDescription',{'left_bits':bits,'hide_callback':hide,'hint_callback':hint,'cold':cold}))
    for hover in [0,1,0x80,0xFF]:cases.append(m.run_description('OnHover',{'hover':hover}))
    for options in [{'null_left_acted':True},{'null_left_version':True},{'null_game_result':True},{'null_acted':True},{'null_characters':True},
                    {'null_speech':True},{'left_bits':0,'null_left_acted':True,'null_acted':True},
                    {'null_action_target':True},{'alias_action_left':True},{'warm_byte':0x80,'class_word':0xDEADBEEF}]:
        r=m.run_description('HideDescription',options)
        guard=any(options.get(key) for key in ['null_left_acted','null_left_version','null_game_result','null_acted','null_characters']) and options.get('left_bits',1)!=0
        assert r['returned']==(not guard) and r['error']==('native_null_guard' if guard else None)
        if guard:
            assert r['final']['description']['highlight_clears']==0 and not r['final']['description']['action_calls']
            assert r['final']['description']['supplied_game_active']==(not(options.get('null_acted') or options.get('null_characters')))
        cases.append(r)
    for phase,mutation in [('stop','clear_left'),('active','clear_acted'),('active','clear_speech'),('highlight','clear_hide'),('hide','clear_hint'),('hide','replace_hint')]:
        r=m.run_description('HideDescription',{'mutation_phase':phase,'mutation':mutation})
        assert r['returned']==(mutation!='clear_acted')
        d=r['final']['description'];assert not d['supplied_game_active']
        if mutation=='clear_acted':
            assert r['error']=='native_null_guard' and not d['restored_speech'] and d['highlight_clears']==0 and not d['action_calls']
        elif mutation=='clear_left':assert d['left_acted'] is None and d['restored_speech']==['speech']
        elif mutation=='clear_speech':assert d['speech'] is None and d['restored_speech']==[None]
        elif mutation=='clear_hide':assert [a['delegate'] for a in d['action_calls']]==['hint_callback']
        elif mutation=='clear_hint':assert [a['delegate'] for a in d['action_calls']]==['hide_callback']
        elif mutation=='replace_hint':assert [a['delegate'] for a in d['action_calls']]==['hide_callback','hint_replacement']
        cases.append(r)
    for left in [0,0x80]:
        m.prepare({'left_bits':left,'cold':True})
        calls=[m.run_description('OnHover',retained=True),m.run_description('HideDescription',{'left_bits':left},retained=True)]
        sequences.append(calls)
    baseline=m.run_description('HideDescription',{'cold':True});assert baseline['returned'];baselines.append(baseline);counts={}
    for index,event in enumerate(baseline['events']):
        kind=event['kind'];counts[kind]=counts.get(kind,0)+1
        stopped=m.run_description('HideDescription',{'cold':True,'failure':[kind,counts[kind]]})
        assert not stopped['returned'] and stopped['events']==baseline['events'][:index+1]
        assert stopped['final']==event['snapshot']
        failures.append({'failure':[kind,counts[kind]],'exact_prefix_verified':True})
    missing=set(m.description_instructions)-m.executed
    assert not missing
    return {'build':BUILD,'targets':m.description_targets,'fields':m.description_fields,'case_count':len(cases),'cases':cases,
            'retained_sequences':sequences,'failure_baselines':baselines,'failure_case_count':len(failures),'failures':failures,
            'instruction_assertions':m.description_assertions,'caller_instructions_executed':len(m.description_instructions),
            'scope':'Actual OnHover and complete HideDescription callers only. Acted.Act, Characters.DisableHighlightAll, coroutine stopping, Unity object/active, metadata and Action effects are named supplied services; runtime class/static roots are authored valid storage. No game-owned service body, scheduler, scene ordering, rendering, or exception unwinding is inferred.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=audit(args.game_root,args.dumper_root);args.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:r[k] for k in ['case_count','failure_case_count','caller_instructions_executed']}))
