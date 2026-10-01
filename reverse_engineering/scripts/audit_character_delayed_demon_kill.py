"""Execute a native captured Demon-kill iterator and explicit controlled resumes."""
import argparse
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_character_publication_entries import Machine as EntryMachine


TARGETS = {
    0x3757F0: ('DelayedDemonKill.MoveNext', 'Character.<DelayedDemonKill>d__103$$MoveNext',
               'bool Character__DelayedDemonKill_d__103__MoveNext (Character__DelayedDemonKill_d__103_o* __this, const MethodInfo* method);'),
    0x363AA0: ('AddStatus', 'CharacterStatuses$$AddStatus',
               'void CharacterStatuses__AddStatus (CharacterStatuses_o* __this, int32_t newStatus, Character_o* sourceRef, Character_o* targetRef, const MethodInfo* method);'),
    0x3645C0: ('Act', 'Character$$Act', 'void Character__Act (Character_o* __this, int32_t trigger, const MethodInfo* method);'),
    0x397750: ('CheckLying', 'CharacterHelper$$CheckLying',
               'bool CharacterHelper__CheckLying (Character_o* c, const MethodInfo* method);'),
    0x369470: ('UpdateUI', 'Character$$UpdateUI', 'void Character__UpdateUI (Character_o* __this, const MethodInfo* method);'),
    0x3B24C0: ('BaseCheckIfCanBeKilled', 'Role$$CheckIfCanBeKilled',
               'bool Role__CheckIfCanBeKilled (Role_o* __this, Character_o* charRef, const MethodInfo* method);'),
    0x1C961F0: ('WaitForSeconds.ctor', 'UnityEngine.WaitForSeconds$$.ctor',
               'void UnityEngine_WaitForSeconds___ctor (UnityEngine_WaitForSeconds_o* __this, float seconds, const MethodInfo* method);'),
}
FIELDS = {'data': (0x50, 8), 'bluff': (0x58, 8), 'register_as': (0x60, 8),
          'trailer': (0x68, 8), 'runtime': (0x70, 8), 'created_dead': (0x98, 8),
          'revealed': (0xD8, 1), 'uses': (0xDC, 4), 'previous': (0xE0, 4),
          'state': (0xE4, 4), 'killed_hidden': (0xEC, 1), 'killed_demon': (0xED, 1),
          'statuses': (0xF0, 8), 'alignment': (0xF8, 4), 'id': (0x118, 4),
          'started': (0x11C, 1), 'history': (0x148, 8), 'hover': (0x150, 8),
          'order': (0x160, 4), 'role': (0x168, 8), 'copied_role': (0x170, 8),
          'state_callback': (0x180, 8), 'saved_act': (0x198, 8), 'act': (0x1A1, 1)}


class Machine(EntryMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root, dumper_root)
        self.kill_targets, self.kill_addresses = [], set()
        for address, (label, name, signature) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
            assert len(rows) == 1 and rows[0]['Signature'] == signature
            self.kill_targets.extend(rows)
            chunks = []
            for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = entry
                while root.unwindinfo.Flags & 4: root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == address:
                    chunks.append((entry.struct.BeginAddress, entry.struct.EndAddress))
            if address == 0x3B24C0:
                assert not chunks
                chunks = [(address, address + 3)]
            assert chunks
            self.ranges[hex(address)] = [[hex(a), hex(b)] for a, b in chunks]
            for a, b in chunks:
                ins = list(self.cs.disasm(self.pe.get_data(a, b-a), a))
                assert sum(i.size for i in ins) == b-a
                self.instructions.update({i.address: i for i in ins})
                self.kill_addresses.update(i.address for i in ins)
        assert [(self.instructions[a].mnemonic, self.instructions[a].op_str) for a in [0x3B24C0, 0x3B24C2]] == [('mov', 'al, 1'), ('ret', '')]
        empty_base=list(self.cs.disasm(self.pe.get_data(0x33ED50,3),0x33ED50))
        assert [(i.mnemonic,i.op_str) for i in empty_base]==[('ret','0')]
        base_died=[r for r in self.metadata['ScriptMethod'] if r['Name']=='Role$$ActOnDied']
        assert len(base_died)==1 and base_died[0]['Address']==0x33ED50
        assert base_died[0]['Signature']=='void Role__ActOnDied (Role_o* __this, Character_o* charRef, const MethodInfo* method);'
        declarations = {
            ('Character', 5487): ['public CharacterData dataRef; // 0x50', 'public CharacterStatuses statuses; // 0xF0',
                                  'private int order; // 0x160', 'public Action onStateChange; // 0x180',
                                  'public bool killedByDemon; // 0xED', 'public Role role; // 0x168',
                                  'public Role bluffRole; // 0x170', 'public ECharacterState prevState; // 0xE0',
                                  'public ECharacterState state; // 0xE4'],
            ('CharacterStatuses', 5488): ['public List<ECharacterStatus> statuses; // 0x10',
                                          'public List<ECharacterStatus> resistances; // 0x18',
                                          'public Character targetCharacter; // 0x20'],
            ('Role', 5853): ['public Action<ActedInfo> onActed; // 0x28'],
            ('CharacterData',5845): ['public Role role; // 0x140'],
            ('Gameplay',5604): ['public static int CurrentReveal; // 0x38'],
            ('UIEvents',5523): ['public static Action OnUIUpdate; // 0x0'],
            ('GameplayEvents',5519): ['public static Action<Character> OnCharacterKilled; // 0x48'],
            ('WaitForSeconds', 6799): ['internal float m_Seconds; // 0x10'],
            ('Character.<DelayedDemonKill>d__103', 5483): ['private int <>1__state; // 0x10',
                'private object <>2__current; // 0x18', 'public Character <>4__this; // 0x20', 'public Character evilRef; // 0x28'],
        }
        for (name, index), fields in declarations.items():
            match = re.search(r'^[^\n]*class ' + re.escape(name) + r'(?: :[^\n]*)? // TypeDefIndex: ' + str(index)
                              + r'\s*\{(.*?)^\}', self.dump, re.M | re.S)
            assert match and all(field in match[1] for field in fields)
        status_enum = re.search(r'^public enum ECharacterStatus // TypeDefIndex: 5491\s*\{(.*?)^\}', self.dump, re.M | re.S)
        assert status_enum and 'MessedUpByEvil = 50;' in status_enum[1] and 'KilledByEvil = 55;' in status_enum[1]
        for name,index,field in [('ECharacterState',5489,'Dead = 20;'),('ETriggerPhase',5605,'OnDied = 50;')]:
            declaration=re.search(r'^public enum '+name+r' // TypeDefIndex: '+str(index)+r'\s*\{(.*?)^\}',self.dump,re.M|re.S)
            assert declaration and field in declaration[1]
        references = set()
        for address in self.kill_addresses:
            i = self.instructions[address]
            for operand in i.operands:
                if operand.type == capstone.CS_OP_MEM and operand.mem.base == capstone.x86.X86_REG_RIP:
                    references.add(i.address + i.size + operand.mem.disp)
                    if i.mnemonic == 'cmp' and operand.size == 1: self.flags.add(i.address + i.size + operand.mem.disp)
        for row in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if row['Address'] in references:
                if row['Name'] not in self.bindings:
                    self.bindings[row['Name']] = self.arena + 0x4000 + len(self.bindings)*0x200
                pointer = self.bindings[row['Name']]
                self.metadata_slots[self.base + row['Address']] = pointer
                self.q(self.base + row['Address'], pointer)
        for row in self.metadata['ScriptString']:
            if row['Address'] in references and self.base + row['Address'] not in self.metadata_slots:
                pointer = self.arena + 0xB0000 + len(self.strings)*0x200
                self.make_string(pointer, row['Value'], 'kill_literal:' + row['Value'])
                self.metadata_slots[self.base + row['Address']] = pointer
                self.q(self.base + row['Address'], pointer)
        required = ['UIEvents_TypeInfo', 'GameplayEvents_TypeInfo', 'Gameplay_TypeInfo', 'UnityEngine.WaitForSeconds_TypeInfo',
                    'Method$System.Collections.Generic.List<ECharacterStatus>.Add()',
                    'Method$System.Collections.Generic.List<ECharacterStatus>.Contains()']
        assert all(name in self.bindings for name in required)
        wait = self.instructions[0x37584F]
        slot = wait.address + wait.size + wait.operands[1].mem.disp
        self.wait_bits = struct.unpack('<I', self.pe.get_data(slot, 4))[0]
        assert self.wait_bits == 0x3EE66666
        self.kill_checks = {
            0x37583C: ('mov', 'dword ptr [rdi + 0x10], 0xffffffff'),
            0x37586C: ('mov', 'qword ptr [rcx], rbx'),
            0x375874: ('mov', 'al, 1'), 0x375876: ('mov', 'dword ptr [rdi + 0x10], 1'),
            0x375891: ('mov', 'dword ptr [rdi + 0x10], 0xffffffff'),
            0x3758DF: ('call', 'rax'), 0x3758E1: ('test', 'al, al'),
            0x3758EF: ('mov', 'dword ptr [rbx + 0xe0], eax'),
            0x3758FC: ('mov', 'byte ptr [rbx + 0xed], 1'),
            0x375903: ('mov', 'dword ptr [rbx + 0xe4], 0x14'),
            0x37592D: ('mov', 'r8, qword ptr [rdi + 0x28]'),
            0x375931: ('xor', 'r9d, r9d'), 0x375956: ('mov', 'r8, qword ptr [rdi + 0x28]'),
            0x3759A8: ('mov', 'rcx, qword ptr [rcx + 0x140]'),
            0x3759F1: ('call', '0x369470'), 0x3759FB: ('xor', 'al, al'),
            0x363AB6: ('mov', 'rsi, r9'), 0x363B33: ('mov', 'qword ptr [rcx], rsi'),
            0x3694BE: ('mov', 'dword ptr [rbx + 0x160], ecx'),
            0x1C961FD: ('movaps','xmm6, xmm1'),0x1C96203: ('call','0x33ed50'),
            0x1C96208: ('movss','dword ptr [rbx + 0x10], xmm6'),
            0x1C9620D: ('movaps','xmm6, xmmword ptr [rsp + 0x20]')}
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == expected for a, expected in self.kill_checks.items())
        self.status = self.arena + 0xD0000
        self.active_list, self.resistance_list, self.alternate_status, self.alternate_list = [self.arena+n for n in [0xD1000,0xD2000,0xD3000,0xD4000]]
        self.ui_static, self.kill_static, self.phase_static = [self.arena+n for n in [0xD5000,0xD6000,0xD7000]]
        self.evil, self.alternate_evil, self.alternate_data, self.alternate_role, self.copied, self.old_current = [self.arena+n for n in [0xD8000,0xD9000,0xDA000,0xDB000,0xDC000,0xDD000]]
        self.delegates_by_kind = {kind: self.arena + 0xE0000 + n*0x1000 for n,kind in enumerate(['state','ui','killed','trigger'])}
        self.gateways = {kind: self.stop + 0x600 + n*0x10 for n,kind in enumerate(['predicate','died','state','ui','killed','trigger','role_act','role_bluff'])}
        self.static_ids.update({p:name for name,p in [('statuses',self.status),('active_statuses',self.active_list),
            ('resistances',self.resistance_list),('alternate_statuses',self.alternate_status),('alternate_active',self.alternate_list),
            ('evil',self.evil),('alternate_evil',self.alternate_evil),('alternate_data',self.alternate_data),
            ('alternate_role',self.alternate_role),('copied_role',self.copied),('old_current',self.old_current),
            *[(name+'_delegate',p) for name,p in self.delegates_by_kind.items()]]})
        for role,label in [(self.role,'role'),(self.alternate_role,'alternate_role'),(self.copied,'copied_role')]:
            self.static_ids.update({role+0x500:label+'_kill_method',role+0x580:label+'_died_method'})
        self.phase, self.phase_frames, self.resume_index = 'Unprepared', [], 0

    def snapshot(self):
        def lst(pointer):
            count = self.rd(pointer+0x18)
            assert count <= 16
            return {'identity':self.object_id(pointer),'backing':self.rq(pointer+0x10),
                    'count':count,'version':self.rd(pointer+0x1C),
                    'values':[self.rd(pointer+0x120+i*4) for i in range(count)],
                    'storage':[self.rd(pointer+0x120+i*4) for i in range(8)]}
        return {'actor':{name:int.from_bytes(self.u.mem_read(self.actor+off,size),'little') for name,(off,size) in FIELDS.items()},
                'statuses':[{'identity':self.object_id(p),'active':self.object_id(self.rq(p+0x10)),
                    'resistances':self.object_id(self.rq(p+0x18)),'target':self.object_id(self.rq(p+0x20))} for p in [self.status,self.alternate_status]],
                'lists':[lst(p) for p in [self.active_list,self.resistance_list,self.alternate_list]],
                'iterators':[self.factory_object(p) for p,name in self.objects.items() if name.startswith('kill_factory')],
                'waits':{name:self.rd(p+0x10) for p,name in self.objects.items() if name.startswith('wait')},
                'role_callbacks':{self.object_id(p):self.object_id(self.rq(p+0x28)) for p in [self.role,self.alternate_role,self.copied]},
                'allocated_objects':list(self.objects.values()),'role_calls':self.role_call_log.copy(),
                'ui_updates':self.ui_updates,'killed_notifications':self.killed_notifications,
                'metadata_flags':{hex(p):self.u.mem_read(self.base+p,1)[0] for p in sorted(self.flags)}}

    def event(self, kind, args):
        key = self.phase,kind
        self.phase_counts[key] = self.phase_counts.get(key,0)+1
        self.counts[kind] = self.counts.get(kind,0)+1
        self.events.append({'phase':self.phase,'kind':kind,'args':args,'snapshot':self.snapshot()})
        self.actor_prefixes.append(bytes(self.u.mem_read(self.actor,0x1B8)))
        if self.options.get('failure') == [self.phase,kind,self.phase_counts[key]]:
            self.error=kind; self.u.emu_stop(); return False
        return True

    def prepare(self, options):
        self.phase_counts, self.actor_prefixes = {},[]
        self.role_call_log, self.ui_updates, self.killed_notifications = [],0,0
        super().prepare(options)
        for p in [self.status,self.active_list,self.resistance_list,self.alternate_status,self.alternate_list,
                  self.ui_static,self.kill_static,self.phase_static,self.evil,self.alternate_evil,self.alternate_data,self.alternate_role,self.copied,
                  *self.delegates_by_kind.values()]: self.u.mem_write(p,bytes(0x800))
        self.q(self.actor+0xF0,0 if options.get('null_status') else self.status)
        self.q(self.actor+0x168,0 if options.get('null_action_role') else self.role)
        self.q(self.actor+0x170,{'alias':self.role,'copied':self.copied}.get(options.get('copied'),0))
        self.d(self.actor+0xE4,options.get('state',5));self.d(self.actor+0xE0,options.get('previous',10))
        self.d(self.actor+0xF8,options.get('alignment',10));self.d(self.actor+0x160,options.get('order',73))
        self.u.mem_write(self.actor+0xEC,bytes([int(options.get('killed_hidden',True))]))
        for pointer,values in [(self.active_list,options.get('statuses',[])),(self.resistance_list,options.get('resistances',[])),
                               (self.alternate_list,options.get('alternate_statuses',[30]))]:
            assert len(values)<=8
            self.q(pointer+0x10,pointer+0x100);self.d(pointer+0x118,16)
            self.d(pointer+0x18,len(values));self.d(pointer+0x1C,options.get('list_version',9))
            for i,value in enumerate(values):self.d(pointer+0x120+i*4,value)
        for status,active in [(self.status,self.active_list),(self.alternate_status,self.alternate_list)]:
            self.q(status+0x10,0 if options.get('null_active') else active)
            self.q(status+0x18,0 if options.get('null_resistances') else
                   active if options.get('alias_lists') else self.resistance_list)
            self.q(status+0x20,self.evil)
        for data,role in [(self.fixtures['data'],self.role),(self.alternate_data,self.alternate_role)]:
            self.q(data+0x140,0 if options.get('null_data_role') else role)
        for role in [self.role,self.alternate_role,self.copied]:
            self.q(role,role+0x600)
            self.q(role+0x600+0x298,self.base+0x3B24C0 if options.get('predicate','base')=='base' else self.gateways['predicate'])
            self.q(role+0x600+0x2A0,role+0x500)
            self.q(role+0x600+0x218,self.base+0x33ED50 if options.get('died','base')=='base' else self.gateways['died'])
            self.q(role+0x600+0x220,role+0x580)
            self.q(role+0x600+0x208,self.gateways['role_act']);self.q(role+0x600+0x210,role+0x540)
            self.q(role+0x600+0x258,self.gateways['role_bluff']);self.q(role+0x600+0x260,role+0x560)
        for name,p in self.delegates_by_kind.items():
            self.q(p+0x18,self.gateways[name]);self.q(p+0x28,p+0x80);self.q(p+0x40,p)
        for field,name in [(0x180,'state'),(0x110,'trigger')]:
            self.q(self.actor+field,self.delegates_by_kind[name] if options.get(name+'_callback',True) else 0)
        self.q(self.bindings['UIEvents_TypeInfo']+0xB8,self.ui_static)
        self.q(self.ui_static,self.delegates_by_kind['ui'] if options.get('ui_callback',True) else 0)
        self.q(self.bindings['GameplayEvents_TypeInfo']+0xB8,self.kill_static)
        self.q(self.kill_static+0x48,self.delegates_by_kind['killed'] if options.get('killed_callback',True) else 0)
        self.q(self.bindings['Gameplay_TypeInfo']+0xB8,self.phase_static)
        self.d(self.phase_static+0x28,50);self.d(self.phase_static+0x38,options.get('current_reveal',101))
        for p in self.bindings.values(): self.d(p+0xE0,0 if options.get('cold') else 1)
        self.q(self.actor+0x58,self.fixtures['bluff'] if options.get('bluff_live') else 0)
        if 'metadata_flag' in options:
            for p in self.flags:self.u.mem_write(self.base+p,bytes([options['metadata_flag']]))
        self.iterator_pointer=None

    def ret(self, value=0):
        # All authored services can destroy caller-saved state. RAX retains its
        # supplied pointer/AL result; integer and XMM nonvolatiles are untouched.
        x=self.x
        for name in ['RCX','RDX','R8','R9','R10','R11']:self.u.reg_write(getattr(x,'UC_X86_REG_'+name),0xCA11000000000000)
        for i in range(6):self.u.reg_write(getattr(x,f'UC_X86_REG_XMM{i}'),0xBAD00000+i)
        super().ret(value)

    def hook(self, uc, address, size, data):
        rva,x=address-self.base,self.x
        cx,dx,r8,r9=[self.reg(reg) for reg in [x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8,x.UC_X86_REG_R9]]
        while self.phase_frames and address==self.phase_frames[-1][0]:
            _,self.phase=self.phase_frames.pop()
        if rva in TARGETS or rva in [0x364A90,0x368790,0x33ED50]:
            label=TARGETS[rva][0] if rva in TARGETS else {0x364A90:'Factory',0x368790:'RoleAct',0x33ED50:'EmptyBase'}[rva]
            if rva==0x33ED50 and self.rq(self.reg(x.UC_X86_REG_RSP))==self.base+0x3759C7:label='BaseActOnDied'
            if rva!=self.invoke_entry:
                self.phase_frames.append((self.rq(self.reg(x.UC_X86_REG_RSP)),self.phase))
                self.phase=f'{label}#{self.resume_index}'
            args=[label]
            if rva==0x363AA0:
                assert cx in [self.status,self.alternate_status] and dx & 0xFFFFFFFF in [50,55]
                assert r8 in [0,self.evil,self.alternate_evil,self.actor] and r9==0 and self.rq(self.reg(x.UC_X86_REG_RSP)+0x28)==0
                args += [self.object_id(cx),dx & 0xFFFFFFFF,{'source_ref':self.object_id(r8),'target_ref':None,'method_info':None}]
            elif rva==0x3B24C0 or label=='BaseActOnDied':
                assert cx in [self.role,self.alternate_role] and dx==self.actor
                assert r8==cx+(0x500 if rva==0x3B24C0 else 0x580)
                args += [self.object_id(cx),self.object_id(dx),self.object_id(r8)]
            elif rva==0x3645C0:
                assert cx==self.actor and dx & 0xFFFFFFFF==50 and r8==0
            elif rva in [0x397750,0x369470]:assert cx==self.actor and dx==0
            elif rva==0x1C961F0:
                assert self.objects[cx].startswith('wait') and r8==0
                assert self.reg(x.UC_X86_REG_XMM1)&0xFFFFFFFF==self.wait_bits
                args += [self.object_id(cx),self.wait_bits]
            if not self.event('native_entry',args):return
        self.executed.add(rva)
        if rva==0xB45070:
            assert cx in [self.active_list,self.resistance_list,self.alternate_list]
            assert r8==self.bindings['Method$System.Collections.Generic.List<ECharacterStatus>.Contains()']
            if self.event('status_membership_service',[self.object_id(cx),dx & 0xFFFFFFFF]):
                values=[self.rd(cx+0x120+i*4) for i in range(self.rd(cx+0x18))]
                self.ret(0xA500000000000000|int(dx & 0xFFFFFFFF in values))
        elif rva==0x41A0:
            assert cx in [self.active_list,self.alternate_list] and r8==self.bindings['Method$System.Collections.Generic.List<ECharacterStatus>.Add()']
            if self.event('status_list_add_service',[self.object_id(cx),dx & 0xFFFFFFFF]):
                count=self.rd(cx+0x18);assert count<16
                self.d(cx+0x120+count*4,dx&0xFFFFFFFF);self.d(cx+0x18,count+1);self.d(cx+0x1C,(self.rd(cx+0x1C)+1)&0xFFFFFFFF)
                if dx & 0xFFFFFFFF==50:
                    effect=self.options.get('first_append_effect')
                    if effect=='replace_status':self.q(self.actor+0xF0,self.alternate_status)
                    elif effect=='clear_status':self.q(self.actor+0xF0,0)
                    elif effect=='replace_evil':self.q(self.iterator_pointer+0x28,self.alternate_evil)
                self.ret(0xCA11000000000000)
        elif rva==0x363C40:
            assert cx in [self.status,self.alternate_status] and dx & 0xFFFFFFFF in [10,30] and r8==0
            if self.event('active_membership_service',[self.object_id(cx),dx & 0xFFFFFFFF]):
                active=self.rq(cx+0x10);assert active
                values=[self.rd(active+0x120+i*4) for i in range(self.rd(active+0x18))]
                self.ret(0xA500000000000000|int(dx & 0xFFFFFFFF in values))
        elif address in self.gateways.values():
            kind=next(name for name,p in self.gateways.items() if p==address)
            if kind in ['predicate','died']:
                assert cx in [self.role,self.alternate_role] and dx==self.actor and r8==cx+(0x500 if kind=='predicate' else 0x580)
            elif kind in ['role_act','role_bluff']:
                assert cx in [self.role,self.alternate_role,self.copied] and dx & 0xFFFFFFFF==50 and r8==self.actor
                assert r9==cx+(0x540 if kind=='role_act' else 0x560)
            elif kind=='killed':
                assert cx==self.delegates_by_kind[kind] and dx==self.actor and r8==cx+0x80
            elif kind=='trigger':
                assert cx==self.delegates_by_kind[kind] and dx==self.actor and r8 & 0xFFFFFFFF==50 and r9==cx+0x80
            else:assert cx==self.delegates_by_kind[kind] and dx==cx+0x80
            if self.event(kind+'_service',[self.object_id(cx)]):
                effect=self.options.get(kind+'_effect')
                if effect=='replace_status':self.q(self.actor+0xF0,self.alternate_status)
                elif effect=='clear_status':self.q(self.actor+0xF0,0)
                elif effect=='replace_data':self.q(self.actor+0x50,self.alternate_data)
                elif effect=='clear_data':self.q(self.actor+0x50,0)
                elif effect=='change_state':self.d(self.actor+0xE4,30)
                elif effect=='change_reveal':self.d(self.phase_static+0x38,0xFFFFFFFF)
                elif effect=='replace_evil':self.q(self.iterator_pointer+0x28,self.alternate_evil)
                if kind=='ui':self.ui_updates+=1
                elif kind=='killed':self.killed_notifications+=1
                elif kind in ['role_act','role_bluff']:self.role_call_log.append({'role':self.object_id(cx),'route':kind})
                self.ret(0xA500000000000000|int(self.options.get('predicate_result',False)) if kind=='predicate' else 0xCA11000000000000)
        elif rva==0x282580:
            assert cx==self.bindings['ETriggerPhase_TypeInfo'] and self.rd(dx)==50
            if self.event('box_service',[50]):self.ret(self.arena+0xEF000)
        elif rva==0xF74DF0:
            assert dx==self.arena+0xEF000 and r8==0
            if self.event('trigger_format_service',[self.object_id(cx)]):self.ret(self.string_pointers['log'])
        elif rva==0x1C4B450:
            assert cx==self.string_pointers['log'] and dx==0
            if self.event('trigger_log_service',[]):self.ret()
        elif rva==0x1C82480:
            assert cx in [0,self.fixtures['bluff']] and dx==r8==0
            if self.event('bluff_liveness_service',[self.object_id(cx)]):self.ret(0xA500000000000000|int(bool(cx) and self.options.get('bluff_live')))
        elif rva==0x1C961F0:
            # The decoded constructor executes, including its empty base call,
            # float store and XMM6 restoration. Do not enter the inherited stub.
            pass
        else:
            super().hook(uc,address,size,data)

    def invoke_kill(self, entry, receiver, phase):
        self.invoke_entry=entry;self.phase=phase;self.phase_frames=[]
        returned=self.invoke(entry,receiver,0,0,0)
        assert not self.phase_frames or not returned
        return {'phase':phase,'returned':returned,'al':self.reg(self.x.UC_X86_REG_RAX)&0xFF if returned else None,
                'snapshot':self.snapshot(),'error':self.error}

    def run_kill(self, options=None):
        options=options or {};self.resume_index=0;self.phase='Factory'
        self.prepare(options)
        # Factory receiver/argument are set explicitly; no iterator replacement
        # is authored after its native physical allocation and field captures.
        self.invoke_entry=0x364A90
        evil={'null':0,'self':self.actor,'alternate':self.alternate_evil}.get(options.get('evil'),self.evil)
        self.phase_frames=[]
        factory=self.invoke(0x364A90,0 if options.get('null_actor') else self.actor,evil,0,0)
        self.stage_actor_bytes=[bytes(self.u.mem_read(self.actor,0x1B8))]
        stages=[{'phase':'Factory','returned':factory,'snapshot':self.snapshot(),'error':self.error}]
        if factory:
            self.iterator_pointer=self.reg(self.x.UC_X86_REG_RAX)
            assert self.objects[self.iterator_pointer].startswith('kill_factory')
            if options.get('iterator_state') is not None:self.d(self.iterator_pointer+0x10,options['iterator_state']&0xFFFFFFFF)
            if options.get('old_current'):self.q(self.iterator_pointer+0x18,self.old_current)
            for n in range(options.get('resumes',3)):
                self.resume_index=n+1
                stage=self.invoke_kill(0x3757F0,self.iterator_pointer,f'MoveNext#{n+1}')
                stages.append(stage)
                self.stage_actor_bytes.append(bytes(self.u.mem_read(self.actor,0x1B8)))
                if not stage['returned']:break
        return {'input':options.copy(),'stages':stages,'events':self.events.copy(),'final':self.snapshot(),
                'returned':all(stage['returned'] for stage in stages),'error':self.error}


def verify_normal(m,result):
    options=result['input'];assert result['returned']
    initial=result['stages'][0]['snapshot'];final=result['final'];actor=final['actor']
    assert m.stage_actor_bytes[0]==m.actor_prefixes[0]
    entry_state=options.get('iterator_state',0)&0xFFFFFFFF
    takes_kill=entry_state in [0,1] and options.get('state',5)!=20 and (
        options.get('predicate','base')=='base' or options.get('predicate_result',False))
    expected_actor=initial['actor'].copy()
    if takes_kill:
        expected_actor.update(previous=options.get('state',5),state=20,killed_demon=1,
                              order=options.get('current_reveal',101))
    assert actor==expected_actor
    # Full object retention includes padding and fields absent from the public
    # projection. Only the four verified inert-path writes may differ.
    expected_bytes=bytearray(m.stage_actor_bytes[0])
    for name in ['previous','state','killed_demon','order']:
        off,width=FIELDS[name]
        expected_bytes[off:off+width]=expected_actor[name].to_bytes(width,'little')
    assert m.stage_actor_bytes[-1]==bytes(expected_bytes)
    assert final['ui_updates']==int(takes_kill and options.get('ui_callback',True))
    assert final['killed_notifications']==int(takes_kill and options.get('killed_callback',True))
    values=list(options.get('statuses',[]));original_count=len(values)
    resisted=values.copy() if options.get('alias_lists') else options.get('resistances',[])
    accepted=False
    for status in [50,55]:
        if takes_kill and status not in resisted:
            accepted=True
            if status not in values:values.append(status)
    assert final['lists'][0]['values']==values
    assert final['lists'][0]['version']==(options.get('list_version',9)+len(values)-original_count)&0xFFFFFFFF
    assert final['statuses'][0]['target']==(None if accepted else 'evil')
    assert final['lists'][1]==initial['lists'][1] and final['lists'][2]==initial['lists'][2]
    assert final['statuses'][1]==initial['statuses'][1]
    iterator=final['iterators'][0]
    expected_evil={'null':None,'self':'actor','alternate':'alternate_evil'}.get(options.get('evil'),'evil')
    assert iterator['owner']=='actor' and iterator['argument']==expected_evil
    assert iterator['state']==(0xFFFFFFFF if entry_state in [0,1] else entry_state)
    if entry_state==0:
        assert [s['al'] for s in result['stages'][1:]]==[1,0,0]
        first=result['stages'][1]['snapshot']
        assert m.stage_actor_bytes[1]==m.stage_actor_bytes[0]
        for key in ['actor','statuses','lists','role_callbacks','role_calls','ui_updates','killed_notifications']:
            assert first[key]==initial[key]
        assert first['iterators'][0]['state']==1
        assert list(first['waits'].values())==[m.wait_bits]
        assert first['iterators'][0]['current']==next(iter(first['waits']))
        assert iterator['current']==first['iterators'][0]['current']
        assert result['stages'][2]['snapshot']==result['stages'][3]['snapshot']
        assert m.stage_actor_bytes[2]==m.stage_actor_bytes[3]
    else:
        assert [s['al'] for s in result['stages'][1:]]==[0,0,0]
        assert iterator['current']==('old_current' if options.get('old_current') else None)
        assert not final['waits']
    expected_roles=[]
    if takes_kill:
        lying=(options.get('bluff_live',False) or options.get('alignment',10)==20)
        if 30 in values:lying=False
        if 10 in values:lying=True
        real_bluff=lying and not (options.get('alignment',10)==20 and options.get('copied'))
        expected_roles.append({'role':'role','route':'role_bluff' if real_bluff else 'role_act'})
        if options.get('copied'):
            expected_roles.append({'role':'role' if options['copied']=='alias' else 'copied_role',
                                   'route':'role_bluff' if lying else 'role_act'})
    assert final['role_calls']==expected_roles
    for role in ['role','copied_role','alternate_role']:
        touched=any(r['role']==role for r in expected_roles)
        assert (final['role_callbacks'][role]!=initial['role_callbacks'][role])==touched
    entries=[e['args'][0] for e in result['events'] if e['kind']=='native_entry' and
             e['args'][0] not in ['Factory','EmptyBase','DelayedDemonKill.MoveNext','WaitForSeconds.ctor']]
    expected_entries=[]
    if entry_state in [0,1] and options.get('state',5)!=20:
        if options.get('predicate','base')=='base':expected_entries.append('BaseCheckIfCanBeKilled')
    if takes_kill:
        expected_entries+=['AddStatus','AddStatus','Act','CheckLying']+['RoleAct']*len(expected_roles)+['BaseActOnDied','UpdateUI']
    assert entries==expected_entries
    calls=[e for e in result['events'] if e['kind']=='native_entry' and e['args'][0]=='AddStatus']
    assert [e['args'][2] for e in calls]==([50,55] if takes_kill else [])
    assert all(e['args'][3]=={'source_ref':expected_evil,'target_ref':None,'method_info':None} for e in calls)
    result['full_actor_retention_and_normal_projection_verified']=True


def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root)
    cases=[]
    for state,cold,evil,predicate,copy,statuses,resistances in itertools.product(
            [5,20,30],[False,True],['null','self','evil'],['base','service'],[None,'copied','alias'],
            [[],[50,55],[10],[30]],[[],[50],[55],[50,55]]):
        result=m.run_kill({'state':state,'cold':cold,'evil':evil,'predicate':predicate,'copied':copy,
                           'statuses':statuses,'resistances':resistances})
        verify_normal(m,result)
        cases.append({**result,'events':[{'phase':e['phase'],'kind':e['kind'],'args':e['args']} for e in result['events']]})
        if len(cases)%288==0:print(json.dumps({'normal_cases_verified':len(cases)}),flush=True)
    retained=[]
    for options in [{'old_current':True},{'iterator_state':-1,'old_current':True},
                    {'iterator_state':7,'old_current':True},{'iterator_state':1,'old_current':True},
                    {'predicate':'service','predicate_result':True,'list_version':0xFFFFFFFF},
                    {'state_callback':False,'ui_callback':False,'killed_callback':False,'trigger_callback':False}]:
        result=m.run_kill(options);verify_normal(m,result);retained.append(result)
    policies=[]
    for alignment,bluff,copy,statuses in itertools.product([10,20],[False,True],[None,'copied','alias'],[[],[10],[30],[10,30]]):
        result=m.run_kill({'alignment':alignment,'bluff_live':bluff,'copied':copy,'statuses':statuses})
        verify_normal(m,result);policies.append(result)
    for options in [{'alias_lists':True},{'alias_lists':True,'statuses':[50,55]},
                    {'metadata_flag':2},{'metadata_flag':255},{'state':0xFFFFFFFF,'current_reveal':0xFFFFFFFF}]:
        result=m.run_kill(options);verify_normal(m,result);policies.append(result)
    mutations=[]
    for kind,effect in [('predicate','change_state'),('predicate','replace_data'),('state','replace_status'),
                        ('state','clear_status'),('state','replace_data'),('state','clear_data'),
                        ('state','replace_evil'),('ui','replace_data'),('killed','change_reveal')]:
        options={kind+'_effect':effect,'predicate':'service' if kind=='predicate' else 'base','predicate_result':True,'died':'service'}
        result=m.run_kill(options)
        assert result['returned']==(effect not in ['clear_status','clear_data'])
        a=result['final']['actor']
        assert a['state']==20 and a['killed_demon']==1
        assert a['previous']==(30 if kind=='predicate' and effect=='change_state' else 5)
        assert a['order']==(0xFFFFFFFF if effect=='change_reveal' else 101 if result['returned'] else 73)
        if effect=='replace_status':
            assert a['statuses']==m.alternate_status and result['final']['lists'][0]['values']==[]
            assert result['final']['lists'][2]['values']==[30,50,55]
        if effect=='replace_data':
            assert a['data']==m.alternate_data
            died=[e for e in result['events'] if e['kind']=='died_service']
            assert died[-1]['args']==['alternate_role']
        if effect=='replace_evil':
            calls=[e for e in result['events'] if e['kind']=='native_entry' and e['args'][0]=='AddStatus']
            assert all(e['args'][3]['source_ref']=='alternate_evil' for e in calls)
            assert result['final']['statuses'][0]['target'] is None
        mutations.append(result)
    for effect in ['replace_status','clear_status','replace_evil']:
        result=m.run_kill({'first_append_effect':effect})
        assert result['returned']==(effect!='clear_status')
        assert result['final']['actor']['state']==20 and result['final']['actor']['killed_demon']==1
        calls=[e for e in result['events'] if e['kind']=='native_entry' and e['args'][0]=='AddStatus']
        assert calls[0]['args'][1:]==['statuses',50,{'source_ref':'evil','target_ref':None,'method_info':None}]
        assert result['final']['statuses'][0]['target'] is None
        if effect=='replace_status':
            assert calls[1]['args'][1:]==['alternate_statuses',55,{'source_ref':'evil','target_ref':None,'method_info':None}]
            assert result['final']['lists'][0]['values']==[50] and result['final']['lists'][2]['values']==[30,55]
        elif effect=='replace_evil':
            assert calls[1]['args'][3]['source_ref']=='alternate_evil'
            assert result['final']['lists'][0]['values']==[50,55]
        else:
            assert len(calls)==1 and result['final']['lists'][0]['values']==[50]
            assert result['final']['ui_updates']==0 and result['final']['actor']['order']==73
        mutations.append(result)
    malformed=[]
    for options in [{'null_actor':True},{'null_data':True},{'null_data_role':True},{'null_status':True},
                    {'null_resistances':True},{'null_active':True},{'null_action_role':True}]:
        result=m.run_kill(options)
        assert not result['returned'] and result['error']=='native_null_guard'
        assert result['stages'][1]['returned'] and result['stages'][1]['al']==1
        assert result['final']['iterators'][0]['state']==0xFFFFFFFF
        assert result['final']['iterators'][0]['current']==next(iter(result['final']['waits']))
        after_kill=not any(options.get(k) for k in ['null_actor','null_data','null_data_role'])
        expected_actor=result['stages'][0]['snapshot']['actor'].copy()
        if after_kill:expected_actor.update(previous=5,state=20,killed_demon=1)
        assert result['final']['actor']==expected_actor
        expected_bytes=bytearray(m.stage_actor_bytes[0])
        for name in ['previous','state','killed_demon']:
            off,width=FIELDS[name];expected_bytes[off:off+width]=expected_actor[name].to_bytes(width,'little')
        assert m.stage_actor_bytes[-1]==bytes(expected_bytes)
        assert result['final']['ui_updates']==int(bool(options.get('null_action_role')))
        assert result['final']['killed_notifications']==0
        result['full_actor_partial_prefix_verified']=True
        malformed.append(result)
    baselines,failures=[],[]
    for options in [{'cold':True,'died':'service','predicate':'service','predicate_result':True,'copied':'alias'},
                    {'cold':True,'statuses':[50,55],'resistances':[50]},
                    {'cold':True,'predicate':'service','predicate_result':False}]:
        baseline=m.run_kill(options);assert baseline['returned'];ordinal=len(baselines);baselines.append(baseline)
        prefixes=m.actor_prefixes.copy();counts={}
        for index,event in enumerate(baseline['events']):
            key=event['phase'],event['kind'];counts[key]=counts.get(key,0)+1
            failure=[*key,counts[key]]
            stopped=m.run_kill(dict(options,failure=failure))
            assert not stopped['returned'] and stopped['events']==baseline['events'][:index+1]
            assert stopped['final']==event['snapshot']
            assert bytes(m.u.mem_read(m.actor,0x1B8))==prefixes[index]
            failures.append({'baseline':ordinal,'failure':failure,'prefix_length':index+1,'exact_snapshot_and_actor_bytes_verified':True})
    return {'build':BUILD,'targets':m.kill_targets,'factory_target':next(t for t in m.entry_targets if t['Name']=='Character$$DelayedDemonKill'),
            'instruction_assertions':len(m.kill_checks),'wait_f32_bits':m.wait_bits,
            'case_count':len(cases),'cases':cases,'retained_sequences':retained,'policy_cases':policies,'mutation_cases':mutations,
            'malformed_cases':malformed,'failure_baselines':baselines,'failure_case_count':len(failures),'failure_cases':failures,
            'native_instructions_executed':len(m.executed&m.kill_addresses),'native_instructions_decoded':len(m.kill_addresses),
            'native_execution_addresses':len(m.executed),
            'scope':'Actual DelayedDemonKill factory and manually ordered MoveNext calls execute with WaitForSeconds constructor, base kill predicate, AddStatus, Act(OnDied), CheckLying, RoleAct, empty base ActOnDied and UpdateUI bodies. Runtime/delegate/GC/List membership and bounded append, concrete role actions, alternate predicates and callbacks remain authored services. No scheduler readiness, HP/score/dead-roster subscribers, real Unity lifetime or managed unwinding is inferred.'}


def serialize_report(report):
    """Keep every projection; one compact normal fixture per reviewable line."""
    fields=[]
    for key,value in report.items():
        if key=='cases' and value:
            payload='[\n'+',\n'.join('    '+json.dumps(row,separators=(',',':')) for row in value)+'\n  ]'
        else:
            lines=json.dumps(value,indent=2).splitlines()
            payload=lines[0]+''.join('\n  '+line for line in lines[1:])
        fields.append('  '+json.dumps(key)+': '+payload)
    return '{\n'+',\n'.join(fields)+'\n}\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root',type=Path);parser.add_argument('dumper_root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=audit(args.game_root,args.dumper_root)
    args.output.write_text(serialize_report(report),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['case_count','failure_case_count','native_instructions_executed','native_execution_addresses']}))
