"""Exact ShowDescription caller with explicit supplied helper/runtime services."""
import argparse
import hashlib
import itertools
import json
import re
import struct
from copy import deepcopy
from pathlib import Path

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine
from audit_character_oracle_reveal_join import pool_memory
from audit_report_snapshots import pool_snapshots

BODY = (0x368C20, 0x369235, 0x369240)
POISON = 0xFACE123456789090
ITEM_MI = 'Method$System.Collections.Generic.List<ActedInfo>.get_Item()'
LITERALS = {'Can not reveal, something is blocking me!': 'literal_block',
            '': 'literal_empty', 'Killed by the demon\ncan not be revealed': 'literal_killed'}
SERVICES = {0x35DE70: 'get_acted', 0x35DDC0: 'act', 0x35DF00: 'hide',
            0xF76390: 'string_empty', 0x3B4BF0: 'get_description', 0xB22150: 'history_item',
            0x36CAF0: 'highlight', 0x364EC0: 'hidden_count', 0x3BC200: 'hint_constructor',
            0x1C79FD0: 'component_game_object', 0x1C7DC50: 'active_self',
            0x1C82480: 'unity_inequality', 0x1C822C0: 'unity_equality',
            0x2B7B40: 'metadata', 0x281D90: 'class_initialization', 0x2B7D40: 'allocate_hint',
            0x2B6FF0: 'saved_act_barrier', 0x2B7D90: 'native_null_guard'}
FIELDS = {
 'public class Character : MonoBehaviour, ICard // TypeDefIndex: 5487':
 ['public Transform hintPivot; // 0x38', 'public CharacterData dataRef; // 0x50', 'public CharacterData bluff; // 0x58',
  'public Acted acteds; // 0xA8', 'public bool leftAct; // 0xB0', 'public Acted leftActed; // 0xB8',
  'public ECharacterState prevState; // 0xE0', 'public ECharacterState state; // 0xE4', 'public bool killedByDemon; // 0xED',
  'public List<ActedInfo> actedInfos; // 0x148', 'private string savedAct; // 0x198', 'public bool showDisguise; // 0x1A0'],
 'public class CharacterData : ScriptableObject, ICharacterLocData, ICardData // TypeDefIndex: 5845': ['public Role role; // 0x140'],
 'public class ActedInfo // TypeDefIndex: 5498': ['public List<Character> characters; // 0x18'],
 'public class Characters : MonoBehaviour // TypeDefIndex: 5505': ['public static Characters Instance; // 0x0'],
 'public class GameData : ScriptableObject // TypeDefIndex: 5928': ['public static EGameState GameState; // 0x4'],
 'public class PlayerController : MonoBehaviour // TypeDefIndex: 5535': ['public static PlayerInfo PlayerInfo; // 0x0'],
 'public class PlayerInfo // TypeDefIndex: 5536': ['public Block blocks; // 0x20'],
 'public abstract class Resource // TypeDefIndex: 5537': ['public Value value; // 0x10'],
 'public static class UIEvents // TypeDefIndex: 5523': ['public static Action<HintInfo, Transform> OnShowHint; // 0x10',
  'public static Action<CharacterData, Transform> OnShowCharacterDataHint; // 0x20', 'public static Action<Character, Transform> OnShowCharacterHint; // 0x28',
  'public static Action<Character, CharacterData> OnShowCustomHint; // 0x90'],
 'public abstract class Delegate : ICloneable, ISerializable // TypeDefIndex: 419': ['private IntPtr invoke_impl; // 0x18',
  'private object m_target; // 0x20', 'private IntPtr method; // 0x28', 'private IntPtr method_code; // 0x40'],
 'public class HintInfo // TypeDefIndex: 5800': ['public string title; // 0x10', 'public string text; // 0x18',
  'public string hints; // 0x20', 'public string flavor; // 0x28', 'public Sprite img; // 0x30', 'public Color borderColor; // 0x38']}


def decode_snapshot(memory, slots, flags, state, ids):
    def q(n, o): return int.from_bytes(memory[n][o:o+8], 'little')
    def d(n, o): return int.from_bytes(memory[n][o:o+4], 'little')
    def oid(v): return ids[v] if v else None
    return dict(memory={n: bytes(b).hex() for n, b in memory.items()}, slots={n: oid(v) for n, v in slots.items()}, flags=flags.copy(),
        actor=dict(data=oid(q('actor', 0x50)), bluff=oid(q('actor', 0x58)), acteds=oid(q('actor', 0xA8)), left_acteds=oid(q('actor', 0xB8)),
                   history=oid(q('actor', 0x148)), saved_act=oid(q('actor', 0x198)), pivot=oid(q('actor', 0x38)),
                   state_bits=d('actor', 0xE4), prev_state_bits=d('actor', 0xE0), killed=memory['actor'][0xED],
                   left=memory['actor'][0xB0], show_disguise=memory['actor'][0x1A0]),
        class_words={n: d(n, 0xE0) for n in ['object_class', 'other_object_class', 'game_data_class', 'other_game_data_class', 'ui_class']},
        history_count_bits={n: d(n, 0x18) for n in ['history', 'other_history']},
        info_characters={n: oid(q(n, 0x18)) for n in ['info0', 'info1', 'info2']},
        characters_count_bits={n: d(n, 0x18) for n in ['characters_list', 'other_characters_list']},
        delegates={hex(o): oid(q('ui_static', o)) for o in [0x10, 0x20, 0x28, 0x90]},
        supplied_state=deepcopy(state))


class Machine(NativeMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root)
        manifest = json.loads((Path(__file__).parents[1]/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(name, key):
            raw = (Path(dumper_root)/name).read_bytes()
            assert hashlib.sha256(raw).hexdigest().upper() == manifest['outputs'][key]['sha256'].upper()
            return raw.decode('utf-8-sig')
        self.metadata = json.loads(pin('script.json', 'script_json')); dump = pin('dump.cs', 'dump_cs')
        self.target = [r for r in self.metadata['ScriptMethod'] if r['Address'] == BODY[0]]
        assert len(self.target) == 1 and self.target[0]['Name'] == 'Character$$ShowDescription'
        assert self.target[0]['Signature'] == 'void Character__ShowDescription (Character_o* __this, const MethodInfo* method);'
        assert self.target[0]['TypeSignature'] == 'vii'
        assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > BODY[0]) == BODY[2]
        chunks = []
        for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
            root = entry
            while root.unwindinfo.Flags & 4: root = root.unwindinfo._chained_entry
            if root.struct.BeginAddress == BODY[0]: chunks.append((entry.struct.BeginAddress, entry.struct.EndAddress))
        assert chunks == [(BODY[0], BODY[1])], chunks
        section = self.pe.get_section_by_rva(BODY[0]); assert section and BODY[2] <= section.VirtualAddress+section.SizeOfRawData
        raw = self.pe.get_data(BODY[0], BODY[2]-BODY[0]); assert len(raw) == BODY[2]-BODY[0] and raw[BODY[1]-BODY[0]:] == b'\xCC'*11
        ins = list(self.cs.disasm(raw[:BODY[1]-BODY[0]], BODY[0])); assert sum(i.size for i in ins) == BODY[1]-BODY[0]
        assert (ins[-1].address,ins[-1].size,ins[-1].mnemonic)==(0x369234,1,'int3')
        self.instructions = {i.address: i for i in ins}; refs = set()
        for i in ins:
            for op in i.operands:
                if op.type == capstone.x86.X86_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                    refs.add(i.address+i.size+op.mem.disp)
            if i.mnemonic == 'cmp' and i.operands[0].type == capstone.x86.X86_OP_MEM and i.operands[0].size == 1 and i.operands[0].mem.base == capstone.x86.X86_REG_RIP:
                self.flag = i.address+i.size+i.operands[0].mem.disp
        self.slot_names = {r['Address']: r['Name'] for category in ['ScriptMetadata', 'ScriptMetadataMethod'] for r in self.metadata[category] if r['Address'] in refs}
        self.literal_values = {r['Address']: r['Value'] for r in self.metadata['ScriptString'] if r['Address'] in refs}
        assert set(self.literal_values.values()) == set(LITERALS)
        self.slot_names.update({a: LITERALS[value] for a, value in self.literal_values.items()})
        self.slots = {n: a for a, n in self.slot_names.items()}
        roots = {'Characters_TypeInfo':'characters_class', 'UIEvents_TypeInfo':'ui_class', 'GameData_TypeInfo':'game_data_class',
                 'HintInfo_TypeInfo':'hint_class', 'UnityEngine.Object_TypeInfo':'object_class', 'PlayerController_TypeInfo':'player_class',
                 ITEM_MI:'item_method', 'Method$System.Collections.Generic.List<ActedInfo>.get_Count()':'count_method',
                 'Method$System.Collections.Generic.List<Character>.get_Count()':'characters_count_method',
                 **{n:n for n in LITERALS.values()}}
        assert set(roots) == set(self.slots); self.root_records = roots
        names = ['actor', 'other_actor', 'data', 'bluff', 'other_data', 'role', 'acteds', 'other_acteds', 'left_acteds', 'other_left',
                 'game_object', 'other_game_object', 'history', 'other_history', 'info0', 'info1', 'info2', 'characters_list', 'other_characters_list',
                 'characters_class', 'characters_static', 'characters_instance', 'other_characters_instance', 'object_class', 'other_object_class',
                 'ui_class', 'ui_static', 'game_data_class', 'other_game_data_class', 'game_data_static', 'other_game_data_static',
                 'player_class', 'player_static', 'player_info', 'blocks', 'value', 'value_class', 'value_method', 'hint_class', 'hint0', 'hint1',
                 'action_hint', 'action_data', 'action_character', 'action_custom', 'replacement_action', 'action_target', 'action_method',
                 'pivot', 'other_pivot', 'acted_text', 'other_text', 'description', 'other_description', 'literal_empty', 'literal_block', 'literal_killed',
                 'item_method', 'count_method', 'characters_count_method']
        self.p = {n:self.arena+0x20000+i*0x1000 for i,n in enumerate(names)}; self.ids = {v:n for n,v in self.p.items()}
        self.sizes = {n:0x200 if n in ['actor','other_actor','value_class'] else 0x180 if n in ['data','bluff','other_data'] else
                      0x100 if n.endswith('_class') or n=='ui_static' else 0x80 if n.startswith('hint') or n in ['acted_text','other_text','description','other_description',*LITERALS.values()] else 0x60 for n in names}
        self.action_gateway, self.value_gateway = self.stop+0x300, self.stop+0x400
        for declaration, required in FIELDS.items():
            m = re.search('^'+re.escape(declaration)+r'\s*\{(.*?)\n\}', dump, re.M|re.S)
            assert m and all(f in m[1] for f in required), declaration
        for declaration, required in [('public enum ECharacterState // TypeDefIndex: 5489','public const ECharacterState Hidden = 5;'),
                                      ('public enum EGameState // TypeDefIndex: 5930','public const EGameState Gameplay = 30;'),
                                      ('public abstract class Value // TypeDefIndex: 5542','// RVA: -1 Offset: -1 Slot: 7\n\tpublic abstract int GetValue();'),
                                      ('public class Block : Resource // TypeDefIndex: 5541','public void .ctor()')]:
            m=re.search('^'+re.escape(declaration)+r'\s*\{(.*?)\n\}',dump,re.M|re.S); assert m and required in m[1]
        self.supplied_targets=[]
        expected_names={0x35DE70:'Acted$$GetActed',0x35DDC0:'Acted$$Act',0x35DF00:'Acted$$Hide',0xF76390:'System.String$$IsNullOrEmpty',
                        0x3B4BF0:'CharacterData$$GetDescription',0xB22150:'System.Collections.Generic.List<object>$$get_Item',
                        0x36CAF0:'Characters$$HighlightCharacters',0x364EC0:'Character$$GetHiddenCardsAmount',0x3BC200:'HintInfo$$.ctor',
                        0x1C79FD0:'UnityEngine.Component$$get_gameObject',0x1C7DC50:'UnityEngine.GameObject$$get_activeSelf',
                        0x1C82480:'UnityEngine.Object$$op_Inequality',0x1C822C0:'UnityEngine.Object$$op_Equality'}
        for a,n in expected_names.items():
            rows=[r for r in self.metadata['ScriptMethod'] if r['Address']==a and r['Name']==n];assert len(rows)==1
            assert rows[0]['TypeSignature']=={0x35DDC0:'viifi',0x35DF00:'vii',0xB22150:'iiii',0x36CAF0:'viii',
                                              0x3BC200:'viiiiiiii',0x1C82480:'iiii',0x1C822C0:'iiii'}.get(a,'iii')
            self.supplied_targets+=rows
        assert self.supplied_targets[[r['Address'] for r in self.supplied_targets].index(0x3BC200)]['Signature']=='void HintInfo___ctor (HintInfo_o* __this, System_String_o* txt, UnityEngine_Sprite_o* img, System_String_o* hints, System_String_o* flavor, System_String_o* title, UnityEngine_Color_o borderColor, const MethodInfo* method);'
        self.delay_bits = struct.unpack('<I', self.pe.get_data(0x1F34B10,4))[0]; assert self.delay_bits==0x3E4CCCCD
        self.checks={0x368CD8:('cmp','dword ptr [rdi + 0xe4], 5'),0x368CEF:('cmp','dword ptr [rdi + 0xe0], 5'),
          0x368D3E:('mov','rbx, qword ptr [rdi + 0x58]'),0x368D5C:('movzx','ecx, byte ptr [rdi + 0xb0]'),
          0x368DEB:('mov','qword ptr [rcx], rax'),0x368E11:('test','rbx, rbx'),0x368E1A:('movss','xmm2, dword ptr [rip + 0x1bcbcee]'),
          0x368E54:('cmp','qword ptr [rcx + 0x140], rbp'),0x368F2A:('mov','qword ptr [rcx], rax'),
          0x368FC0:('mov','r8, qword ptr [rdi + 0x38]'),0x368FDB:('cmp','dword ptr [rcx + 0x18], ebp'),
          0x369008:('mov','rcx, qword ptr [rdi + 0x148]'),0x369036:('test','rax, rax'),0x36903F:('cmp','dword ptr [rax + 0x18], ebp'),
          0x369052:('mov','rbx, qword ptr [rcx]'),0x369088:('mov','rdx, qword ptr [rax + 0x18]'),
          0x3690B3:('mov','rbx, qword ptr [rdi + 0x58]'),0x3690F3:('mov','r8, qword ptr [rdi + 0x58]'),
          0x369127:('mov','r8, qword ptr [rdi + 0x50]'),0x369144:('mov','rax, qword ptr [rip + 0x238ee75]'),
          0x369152:('cmp','dword ptr [rax + 4], 0x1e'),0x369169:('mov','ebx, eax'),
          0x369198:('mov','rdx, qword ptr [rcx]'),0x36919B:('mov','rax, qword ptr [rdx + 0x1a8]'),
          0x3691A2:('mov','rdx, qword ptr [rdx + 0x1b0]'),0x3691AB:('cmp','ebx, eax'),
          0x3691EE:('mov','qword ptr [rsp + 0x38], rbp'),0x3691FD:('xor','r8d, r8d'),
          0x369200:('mov','qword ptr [rsp + 0x28], r9'),0x369208:('mov','qword ptr [rsp + 0x20], r9'),
          0x36920D:('movdqa','xmmword ptr [rsp + 0x40], xmm6'),0x36921F:('mov','r8, qword ptr [rdi + 0x38]')}
        assert all((self.instructions[a].mnemonic,self.instructions[a].op_str)==v for a,v in self.checks.items())
        self.tracking=False;self.u.hook_add(self.unicorn.UC_HOOK_MEM_WRITE,self.observe_write)

    def oid(self,v): return self.ids[v] if v else None
    def observe_write(self,uc,access,address,size,value,user_data):
        if not self.tracking:return
        pc=self.reg(self.x.UC_X86_REG_RIP)-self.base
        if address==self.base+self.flag:
            assert size==value==1 and pc==0x368CCA;self.flag_writes=True
        elif address==self.p['actor']+0x198:
            assert size==8 and pc in [0x368DEB,0x368F2A];self.allowed.setdefault('actor',set()).update(range(0x198,0x1A0))

    def snapshot(self):
        return decode_snapshot({n:bytes(self.u.mem_read(p,self.sizes[n])) for n,p in self.p.items()},
                               {n:self.rq(self.base+a) for n,a in self.slots.items()},
                               {hex(self.flag):self.u.mem_read(self.base+self.flag,1)[0]},self.state,self.ids)

    def write_effect(self,n,o,value,size=8):
        if isinstance(value,str):value=self.p[value]
        self.u.mem_write(self.p[n]+o,value.to_bytes(size,'little'));self.allowed.setdefault(n,set()).update(range(o,o+size))

    def mutate(self,kind):
        ordinal=self.completed_counts.get(kind,0)+1;self.completed_counts[kind]=ordinal
        for target,offset,size,value in self.options.get('mutations',{}).get(kind+':'+str(ordinal),[]):
            if target.startswith('slot:'):
                n=target[5:];self.q(self.base+self.slots[n],self.p[value] if isinstance(value,str) else value);self.slot_writes[n]=value
            else:self.write_effect(target,offset,value,size)

    def prepare(self,options):
        self.options=deepcopy(options);self.events=[];self.counts={};self.error=None
        self.state={n:[] for n in ['entries','metadata','classes','allocations','acted','empty','description','items','unity','highlight','hidden','value','hint_constructors','actions','barriers']}
        for n,p in self.p.items():self.u.mem_write(p,bytes([0xA5])*self.sizes[n])
        for n,record in self.root_records.items():self.q(self.base+self.slots[n],self.p[record])
        for cls,static in [('characters_class','characters_static'),('ui_class','ui_static'),('game_data_class','game_data_static'),
                           ('other_game_data_class','other_game_data_static'),('player_class','player_static')]:self.q(self.p[cls]+0xB8,self.p[static])
        for cls in ['object_class','game_data_class','ui_class']:self.d(self.p[cls]+0xE0,options.get('class_word',0))
        for cls in ['other_object_class','other_game_data_class']:self.d(self.p[cls]+0xE0,0xFFFFFFFF)
        self.d(self.p['game_data_static']+4,options.get('game_state_bits',30));self.d(self.p['other_game_data_static']+4,20)
        self.q(self.p['characters_static'],0 if options.get('null_characters') else self.p['characters_instance'])
        self.q(self.p['player_static'],0 if options.get('null_player') else self.p['player_info'])
        self.q(self.p['player_info']+0x20,0 if options.get('null_blocks') else self.p['blocks'])
        self.q(self.p['blocks']+0x10,0 if options.get('null_value') else self.p['value'])
        self.q(self.p['value'],self.p['value_class']);self.q(self.p['value_class']+0x1A8,self.value_gateway);self.q(self.p['value_class']+0x1B0,self.p['value_method'])
        for n in ['data','bluff','other_data']:self.q(self.p[n]+0x140,0 if options.get('null_role') else self.p['role'])
        for o,n in [(0x38,'pivot'),(0x50,'data'),(0x58,'data' if options.get('alias_data') else 'bluff'),(0xA8,'acteds'),
                    (0xB8,'acteds' if options.get('alias_acteds') else 'left_acteds'),(0x148,'history'),(0x198,'other_text')]:
            self.q(self.p['actor']+o,0 if options.get('null_'+{0x38:'pivot',0x50:'data',0x58:'bluff',0xA8:'acteds',0xB8:'left',0x148:'history',0x198:'saved'}[o]) else self.p[n])
        self.d(self.p['actor']+0xE4,options.get('state_bits',10));self.d(self.p['actor']+0xE0,options.get('prev_state_bits',10))
        for o,key,default in [(0xB0,'left',0),(0xED,'killed',0),(0x1A0,'show_disguise',0)]:self.u.mem_write(self.p['actor']+o,bytes([options.get(key,default)]))
        for n in ['history','other_history']:self.d(self.p[n]+0x18,options.get('history_count_bits',1))
        for n in ['characters_list','other_characters_list']:self.d(self.p[n]+0x18,options.get('characters_count_bits',1))
        for n in ['info0','info1','info2']:self.q(self.p[n]+0x18,0 if options.get('null_info_characters') else self.p['characters_list'])
        for o,n in [(0x10,'action_hint'),(0x20,'action_data'),(0x28,'action_character'),(0x90,'action_custom')]:self.q(self.p['ui_static']+o,0 if o in options.get('null_actions',[]) else self.p[n])
        for n in ['action_hint','action_data','action_character','action_custom','replacement_action']:
            self.q(self.p[n]+0x18,self.action_gateway);self.q(self.p[n]+0x28,self.p['action_method'])
            target=options.get('action_target','action_target');self.q(self.p[n]+0x40,self.p[target] if target else 0)
            self.q(self.p[n]+0x20,self.p['other_actor'])
        for n,text in dict(acted_text='authored speech',other_text='replacement speech',description='authored description',other_description='replacement description',
                           **{name:text for text,name in LITERALS.items()}).items():
            encoded=text.encode('utf-16le');self.d(self.p[n]+0x10,len(encoded)//2);self.u.mem_write(self.p[n]+0x14,encoded+b'\0\0')
        self.u.mem_write(self.base+self.flag,bytes([options.get('warm_byte',0)]))

    def event(self,kind,args):
        okay=super().event(kind,args);e=self.events[-1]
        e.update(raw_args=[self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']],
                 caller=hex(self.rq(self.reg(self.x.UC_X86_REG_RSP))-self.base),native_phase='ShowDescription')
        if kind=='act':e['xmm2_bits']=self.reg(self.x.UC_X86_REG_XMM2)
        if kind=='hint_constructor':
            sp=self.reg(self.x.UC_X86_REG_RSP);e['stack_args']=[self.rq(sp+o) for o in [0x28,0x30,0x38,0x40]]
            e['color_bits']=list(struct.unpack('<IIII',self.u.mem_read(e['stack_args'][2],16)))
        return okay

    def ret(self,value=0):
        for n in ['RCX','RDX','R8','R9','R10','R11']:self.u.reg_write(getattr(self.x,'UC_X86_REG_'+n),POISON)
        for i in range(6):self.u.reg_write(getattr(self.x,f'UC_X86_REG_XMM{i}'),(1<<127)|i)
        super().ret(value)

    def hook(self,uc,address,size,user_data):
        rva=address-self.base;x=self.x;self.executed.add(rva)
        cx,dx,r8,r9=[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']]
        if rva==BODY[0]:self.state['entries'].append([cx,dx,r8,r9])
        if rva in self.instructions:return
        caller=self.rq(self.reg(x.UC_X86_REG_RSP))-self.base
        kind='action' if address==self.action_gateway else 'value' if address==self.value_gateway else SERVICES.get(rva)
        assert kind,(hex(rva),hex(caller))
        result=0;effect=None
        if kind=='metadata':
            assert cx-self.base in self.slot_names;args=[self.slot_names[cx-self.base]];result=self.rq(cx);effect=('metadata',args)
        elif kind=='class_initialization':
            assert cx in [self.p[n] for n in ['object_class','game_data_class','other_object_class','other_game_data_class']] and self.rd(cx+0xE0)==0
            args=[self.oid(cx)];effect=('classes',args)
        elif kind=='allocate_hint':
            assert cx==self.p['hint_class'];result=self.p[self.options.get('allocation_hint','hint0')];args=['hint_class',self.oid(result)];effect=('allocations',args)
        elif kind in ['unity_inequality','unity_equality']:
            assert self.oid(cx) in [None,'data','bluff','other_data'] and dx==r8==0
            default=int(bool(cx) and not self.options.get('destroyed_bluff')) if kind=='unity_inequality' else int(not cx or bool(self.options.get('destroyed_bluff')))
            result=self.options.get(kind+'_bits',0xABCD123456789000|default);args=[self.oid(cx),0,0,result];effect=('unity',[kind,*args])
        elif kind=='component_game_object':
            assert self.oid(cx) in ['acteds','other_acteds','left_acteds'] and dx==0
            result=self.p[self.options.get('game_object','game_object')] if self.options.get('game_object','game_object') else 0;args=[self.oid(cx),0,self.oid(result)]
        elif kind=='active_self':
            assert self.oid(cx) in ['game_object','other_game_object'] and dx==0
            result=self.options.get('active_bits',0xABCD123456789001);args=[self.oid(cx),0,result]
        elif kind=='get_acted':
            assert self.oid(cx) in ['acteds','other_acteds','left_acteds'] and dx==0
            values=self.options.get('get_acted_results',['acted_text']);ordinal=self.counts.get(kind,0);value=values[min(ordinal,len(values)-1)];result=self.p[value] if value else 0
            args=[self.oid(cx),0,self.oid(result)];effect=('acted',[kind,*args])
        elif kind=='string_empty':
            assert dx==0;role='acted' if caller in [0x368DC6,0x368F05] else 'description'
            result=self.options.get(role+'_empty_bits',0xABCD123456789000|int(cx==0 or self.rd(cx+0x10)==0));args=[self.oid(cx),0,result];effect=('empty',args)
        elif kind=='saved_act_barrier':
            assert cx==self.p['actor']+0x198 and self.rq(cx)==dx;args=['actor',0x198,self.oid(dx)];effect=('barriers',args)
        elif kind=='act':
            assert self.oid(cx) in ['left_acteds','acteds','other_left'] and r9==0 and self.reg(x.UC_X86_REG_XMM2)==self.delay_bits
            args=[self.oid(cx),self.oid(dx),self.delay_bits,0];effect=('acted',[kind,*args])
        elif kind=='hide':
            assert self.oid(cx) in ['acteds','other_acteds','left_acteds'] and dx==0;args=[self.oid(cx),0];effect=('acted',[kind,*args])
        elif kind=='get_description':
            assert self.oid(cx) in ['data','bluff','other_data'] and dx==0;value=self.options.get('description_result','description');result=self.p[value] if value else 0
            args=[self.oid(cx),0,self.oid(result)];effect=('description',args)
        elif kind=='history_item':
            assert self.oid(cx) in ['history','other_history'] and r8==self.p['item_method'] and dx<=0xFFFFFFFF
            values=self.options.get('item_results',['info0']);ordinal=self.counts.get(kind,0);value=values[min(ordinal,len(values)-1)];result=self.p[value] if value else 0
            args=[self.oid(cx),dx,'item_method',self.oid(result)];effect=('items',args)
        elif kind=='highlight':
            assert self.oid(cx) in ['characters_instance','other_characters_instance'] and self.oid(dx) in [None,'characters_list','other_characters_list'] and r8==0
            args=[self.oid(cx),self.oid(dx),0];effect=('highlight',args)
        elif kind=='hidden_count':
            assert cx==self.p['actor'] and dx==0;result=self.options.get('hidden_bits',1);args=['actor',0,result];effect=('hidden',args)
        elif kind=='value':
            assert cx==self.p['value'] and dx==self.p['value_method'];result=self.options.get('value_bits',1);args=['value','value_method',result];effect=('value',args)
        elif kind=='hint_constructor':
            assert self.oid(cx) in ['hint0','hint1'] and r8==0
            sp=self.reg(x.UC_X86_REG_RSP);flavor,title,color,mi=[self.rq(sp+o) for o in [0x28,0x30,0x38,0x40]]
            assert flavor==title==r9 and mi==0 and bytes(self.u.mem_read(color,16))==bytes(16)
            args=[self.oid(cx),self.oid(dx),None,self.oid(r9),self.oid(flavor),self.oid(title),[0]*4,0];effect=('hint_constructors',args)
        elif kind=='action':
            cb=self.reg(x.UC_X86_REG_RSI) if caller==0x36922A else self.reg(x.UC_X86_REG_RAX)
            assert self.oid(cb) in ['action_hint','action_data','action_character','action_custom','replacement_action']
            assert cx==self.rq(cb+0x40) and r9==self.rq(cb+0x28)
            args=[self.oid(cb),self.oid(cx),self.oid(dx),self.oid(r8),self.oid(r9)];effect=('actions',args)
        else:args=[]
        if self.event(kind,args):
            if kind=='native_null_guard':self.error=kind;uc.emu_stop();return
            if effect:self.state[effect[0]].append(effect[1])
            if kind=='class_initialization':self.write_effect(self.oid(cx),0xE0,1,4)
            if kind=='hint_constructor':
                for o,v in [(0x18,dx),(0x30,r8),(0x20,r9),(0x28,flavor),(0x10,title)]:self.write_effect(self.oid(cx),o,v)
                self.write_effect(self.oid(cx),0x38,0,16)
            self.mutate(kind);self.ret(result)

    def run(self,options=None,retained=False):
        if not retained:self.prepare(options or {})
        else:self.options=deepcopy(options or {});self.error=None;self.counts={}
        self.allowed,self.slot_writes,self.flag_writes,self.completed_counts={},{},False,{}
        initial=self.snapshot();old=len(self.events);x=self.x;sp=self.stack+0x18008;self.q(sp,self.stop)
        for i,n in enumerate(['RBX','RBP','RSI','RDI','R12','R13','R14','R15']):self.u.reg_write(getattr(x,'UC_X86_REG_'+n),0xFAB0000000000000+i)
        for i in range(6,16):self.u.reg_write(getattr(x,f'UC_X86_REG_XMM{i}'),(1<<125)|i)
        args=[self.p['actor'],0,self.options.get('unused_r8',0xCAFE123400000008),self.options.get('unused_r9',0xCAFE123400000009)]
        for n,v in zip(['RCX','RDX','R8','R9'],args):self.u.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        self.u.reg_write(x.UC_X86_REG_RSP,sp);self.tracking=True
        try:self.u.emu_start(self.base+BODY[0],self.stop,timeout=10000000,count=10000)
        finally:self.tracking=False
        returned=self.reg(x.UC_X86_REG_RIP)==self.stop;assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP)==sp+8
            assert all(self.reg(getattr(x,'UC_X86_REG_'+n))==0xFAB0000000000000+i for i,n in enumerate(['RBX','RBP','RSI','RDI','R12','R13','R14','R15']))
            assert all(self.reg(getattr(x,f'UC_X86_REG_XMM{i}'))==(1<<125)|i for i in range(6,16))
        final=self.snapshot()
        for n,raw in initial['memory'].items():
            a,b=bytes.fromhex(raw),bytes.fromhex(final['memory'][n]);assert all(i in self.allowed.get(n,set()) or value==b[i] for i,value in enumerate(a)),n
        assert final['slots']=={**initial['slots'],**self.slot_writes}
        assert final['flags']==({hex(self.flag):1} if self.flag_writes else initial['flags'])
        row=dict(options=self.options,entry_raw_args=args,returned=returned,error=self.error,initial=initial,events=self.events[old:].copy(),final=final,
                 completed_memory_write_offsets={n:sorted(v) for n,v in self.allowed.items()},completed_slot_writes=self.slot_writes.copy(),
                 reached_native_flag_write=self.flag_writes,normal_abi_verified=returned,unrelated_storage_retained=True)
        verify(row,self.p,self.ids,self.slots,self.flag,self.base,sp);row['independent_ordered_state_verified']=True
        return row


def verify(row,p,ids,slot_addresses,flag,base,entry_sp):
    """Independent branch/order model from complete input bytes and supplied plans."""
    raw={n:bytearray.fromhex(v) for n,v in row['initial']['memory'].items()}
    slots={n:p[v] for n,v in row['initial']['slots'].items()};flags=row['initial']['flags'].copy();state=deepcopy(row['initial']['supplied_state'])
    options=row['options'];regs=row['entry_raw_args'].copy();expected=[];counts={};completed_counts={};error=None;returned=False
    state['entries'].append(row['entry_raw_args'])
    def q(n,o):return int.from_bytes(raw[n][o:o+8],'little')
    def d(n,o):return int.from_bytes(raw[n][o:o+4],'little')
    def signed(v):return (v&0x7FFFFFFF)-(v&0x80000000)
    def oid(v):return ids[v] if v else None
    def ptr(n):return p[n] if n else 0
    def write(n,o,value,size=8):raw[n][o:o+size]=(p[value] if isinstance(value,str) else value).to_bytes(size,'little')
    def actor(o):return oid(q('actor',o))
    class Stop(Exception):pass
    def mutate(kind):
        occurrence=completed_counts.get(kind,0)+1;completed_counts[kind]=occurrence
        for target,offset,size,value in options.get('mutations',{}).get(kind+':'+str(occurrence),[]):
            if target.startswith('slot:'):slots[target[5:]]=p[value] if isinstance(value,str) else value
            else:write(target,offset,value,size)
    def emit(kind,args,caller,category=None,record=None,result=0,extra=None):
        e=dict(kind=kind,args=args,raw_args=regs.copy(),caller=hex(caller),native_phase='ShowDescription',
               snapshot=decode_snapshot(raw,slots,flags,state,ids))
        if extra:e.update(extra)
        expected.append(e);counts[kind]=counts.get(kind,0)+1
        if options.get('failure')==[kind,counts[kind]]:raise Stop
        if category:state[category].append(args if record is None else record)
        if kind=='class_initialization':write(args[0],0xE0,1,4)
        if kind=='hint_constructor':
            for o,n in [(0x18,args[1]),(0x30,args[2]),(0x20,args[3]),(0x28,args[4]),(0x10,args[5])]:write(args[0],o,ptr(n))
            write(args[0],0x38,0,16)
        if kind=='native_null_guard':raise Stop
        mutate(kind);regs[:]=[POISON]*4
        return result
    def guard():emit('native_null_guard',[],0x369234)
    def initialize(root,caller):
        cls=oid(slots[root])
        if d(cls,0xE0)==0:
            regs[0]=p[cls];emit('class_initialization',[cls],caller,'classes')
    def get_acted(caller):
        n=actor(0xA8);regs[0]=ptr(n)
        if n is None:guard()
        regs[1]=0;values=options.get('get_acted_results',['acted_text']);n=values[min(counts.get('get_acted',0),len(values)-1)]
        return emit('get_acted',[oid(regs[0]),0,n],caller,'acted',['get_acted',oid(regs[0]),0,n],ptr(n))
    def empty(value,role,caller):
        regs[0:2]=[value,0];bits=options.get(role+'_empty_bits',0xABCD123456789000|int(value==0 or d(oid(value),0x10)==0))
        return emit('string_empty',[oid(value),0,bits],caller,'empty',result=bits)&255
    def action(cb,dx,r8,caller):
        regs[:]=[q(cb,0x40),dx,r8,q(cb,0x28)]
        emit('action',[cb,oid(regs[0]),oid(dx),oid(r8),oid(regs[3])],caller,'actions')
    def hint(text_name,allocation_caller):
        ui=oid(slots['UIEvents_TypeInfo']);regs[0]=q(ui,0xB8);cb=oid(q(oid(regs[0]),0x10))
        if not cb:return
        regs[0]=slots['HintInfo_TypeInfo'];n=options.get('allocation_hint','hint0')
        emit('allocate_hint',['hint_class',n],allocation_caller,'allocations',result=p[n])
        text=oid(slots[text_name]);empty_name=oid(slots['literal_empty']);color=entry_sp-0x28
        regs[:]=[p[n],ptr(text),0,ptr(empty_name)]
        args=[n,text,None,empty_name,empty_name,empty_name,[0]*4,0]
        emit('hint_constructor',args,0x369218,'hint_constructors',extra=dict(stack_args=[ptr(empty_name),ptr(empty_name),color,0],color_bits=[0]*4))
        action(cb,p[n],q('actor',0x38),0x36922A)
    try:
        if flags[hex(flag)]==0:
            ordered=['Characters_TypeInfo','GameData_TypeInfo','HintInfo_TypeInfo','Method$System.Collections.Generic.List<Character>.get_Count()',
                     'Method$System.Collections.Generic.List<ActedInfo>.get_Count()',ITEM_MI,'UnityEngine.Object_TypeInfo','PlayerController_TypeInfo',
                     'UIEvents_TypeInfo','literal_block','literal_killed','literal_empty']
            for n,caller in zip(ordered,[0x368C46,0x368C52,0x368C5E,0x368C6A,0x368C76,0x368C82,0x368C8E,0x368C9A,0x368CA6,0x368CB2,0x368CBE,0x368CCA]):
                regs[0]=base+slot_addresses[n];emit('metadata',[n],caller,'metadata',result=slots[n])
            flags[hex(flag)]=1
        if d('actor',0xE4)==5:
            initialize('GameData_TypeInfo',0x369144);cls=oid(slots['GameData_TypeInfo']);static=oid(q(cls,0xB8))
            if d(static,4)==30:
                regs[0:2]=[p['actor'],0];hidden=options.get('hidden_bits',1)
                emit('hidden_count',['actor',0,hidden],0x369162,'hidden',result=hidden)
                cls=oid(slots['PlayerController_TypeInfo']);regs[0]=p[cls];regs[1]=q(cls,0xB8)
                regs[0]=q(oid(regs[1]),0)
                if not regs[0]:guard()
                regs[0]=q(oid(regs[0]),0x20)
                if not regs[0]:guard()
                regs[0]=q(oid(regs[0]),0x10)
                if not regs[0]:guard()
                cls=oid(q(oid(regs[0]),0));regs[1]=q(cls,0x1B0);value=options.get('value_bits',1)
                emit('value',['value','value_method',value],0x3691AB,'value',result=value)
                if signed(hidden&0xFFFFFFFF)<=signed(value&0xFFFFFFFF):hint('literal_block',0x3691DD)
        elif d('actor',0xE0)==5 and raw['actor'][0xED]!=0:hint('literal_killed',0x368D2B)
        else:
            captured=actor(0x58);regs[0]=slots['UnityEngine.Object_TypeInfo'];initialize('UnityEngine.Object_TypeInfo',0x368D4F)
            regs[:3]=[ptr(captured),0,0]
            bits=options.get('unity_inequality_bits',0xABCD123456789000|int(bool(captured) and not options.get('destroyed_bluff')))
            emit('unity_inequality',[captured,0,0,bits],0x368D5C,'unity',['unity_inequality',captured,0,0,bits],bits)
            regs[0]=raw['actor'][0xB0];bluff_live=bool(bits&255)
            if regs[0]:
                n=actor(0xA8);regs[0]=ptr(n)
                if not n:guard()
                regs[1]=0;game=options.get('game_object','game_object')
                emit('component_game_object',[n,0,game],0x368EC9 if bluff_live else 0x368D8A,result=ptr(game))
                if not game:guard()
                regs[0:2]=[ptr(game),0];active=options.get('active_bits',0xABCD123456789001)
                emit('active_self',[game,0,active],0x368EDC if bluff_live else 0x368D9D,result=active)
                if active&255:
                    text=get_acted(0x368EFB if bluff_live else 0x368DBC)
                    if not empty(text,'acted',0x368F05 if bluff_live else 0x368DC6):
                        text=get_acted(0x368F20 if bluff_live else 0x368DE1);regs[:2]=[p['actor']+0x198,text];write('actor',0x198,text)
                        emit('saved_act_barrier',['actor',0x198,oid(text)],0x368F32 if bluff_live else 0x368DF3,'barriers')
                        left=actor(0xB8);text=get_acted(0x368F50 if bluff_live else 0x368E11)
                        if not left:guard()
                        regs[0:2]=[p[left],text];regs[3]=0
                        emit('act',[left,oid(text),0x3E4CCCCD,0],0x368F6F if bluff_live else 0x368E30,'acted',
                             ['act',left,oid(text),0x3E4CCCCD,0],extra=dict(xmm2_bits=0x3E4CCCCD))
                        n=actor(0xA8);regs[0]=ptr(n)
                        if not n:guard()
                        regs[1]=0;emit('hide',[n,0],0x368F86 if bluff_live else 0x368E47,'acted',['hide',n,0])
            eligible=True
            if not bluff_live:
                n=actor(0x50);regs[0]=ptr(n)
                if not n:guard()
                eligible=q(n,0x140)!=0
            if eligible:
                if raw['actor'][0x1A0]:
                    cls=oid(slots['UIEvents_TypeInfo']);regs[0]=q(cls,0xB8);cb=oid(q(oid(regs[0]),0x20))
                    if cb:action(cb,q('actor',0x58),q('actor',0x38),0x368FCB)
                else:
                    n=actor(0x58 if bluff_live else 0x50);regs[0]=ptr(n)
                    if not n:guard()
                    regs[1]=0;description=options.get('description_result','description')
                    emit('get_description',[n,0,description],0x368E75,'description',result=ptr(description))
                    if not empty(ptr(description),'description',0x368E7F):
                        cls=oid(slots['UIEvents_TypeInfo']);regs[0]=q(cls,0xB8);cb=oid(q(oid(regs[0]),0x28))
                        if cb:action(cb,p['actor'],q('actor',0x38),0x368FCB)
            n=actor(0x148);regs[0]=ptr(n)
            if not n:guard()
            def item(caller):
                n=actor(0x148);regs[0]=ptr(n)
                if not n:guard()
                regs[1]=(d(n,0x18)-1)&0xFFFFFFFF;regs[2]=slots[ITEM_MI]
                values=options.get('item_results',['info0']);value=values[min(counts.get('history_item',0),len(values)-1)]
                emit('history_item',[n,regs[1],'item_method',value],caller,'items',result=ptr(value))
                if not value:guard()
                return value
            if signed(d(n,0x18))>0:
                info=item(0x368FF5)
                if q(info,0x18):
                    info=item(0x369029);characters=oid(q(info,0x18))
                    if not characters:guard()
                    if signed(d(characters,0x18))>0:
                        cls=oid(slots['Characters_TypeInfo']);regs[0]=q(cls,0xB8);instance=oid(q(oid(regs[0]),0))
                        info=item(0x369076)
                        if not instance:guard()
                        regs[:3]=[p[instance],q(info,0x18),0]
                        emit('highlight',[instance,oid(regs[1]),0],0x369097,'highlight')
            n=actor(0x148)
            if not n:guard()
            if signed(d(n,0x18))>0:
                captured=actor(0x58);regs[0]=slots['UnityEngine.Object_TypeInfo'];initialize('UnityEngine.Object_TypeInfo',0x3690C4)
                regs[:3]=[ptr(captured),0,0];bits=options.get('unity_equality_bits',0xABCD123456789000|int(not captured or bool(options.get('destroyed_bluff'))))
                emit('unity_equality',[captured,0,0,bits],0x3690D1,'unity',['unity_equality',captured,0,0,bits],bits)
                cls=oid(slots['UIEvents_TypeInfo']);regs[0]=q(cls,0xB8);cb=oid(q(oid(regs[0]),0x90))
                if cb:action(cb,p['actor'],q('actor',0x50 if bits&255 else 0x58),0x369101)
        returned=True
    except Stop:error=expected[-1]['kind']
    assert row['returned']==returned and row['error']==error
    assert len(row['events'])==len(expected),(options,[e['kind'] for e in row['events']],[e['kind'] for e in expected])
    for actual,predicted in zip(row['events'],expected):assert actual==predicted,(options,actual['kind'],{k:(actual.get(k),v) for k,v in predicted.items() if actual.get(k)!=v})
    assert row['final']==decode_snapshot(raw,slots,flags,state,ids),options


def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root);cases=[];sequences=[];baselines=[];stops=[]
    for state,prev,killed,left,show,bluff in itertools.product([5,10],[5,10],[0,0x80],[0,0x80],[0,0x80],['live','absent','destroyed']):
        cases.append(m.run(dict(state_bits=state,prev_state_bits=prev,killed=killed,left=left,show_disguise=show,null_bluff=bluff=='absent',destroyed_bluff=bluff=='destroyed')))
    for count,characters,neq,eq in itertools.product([0,1,0xFFFFFFFF],[0,1,0xFFFFFFFF],[0,0x80],[0,0x80]):
        cases.append(m.run(dict(history_count_bits=count,characters_count_bits=characters,unity_inequality_bits=0xABCD123456789000|neq,unity_equality_bits=0xABCD123456789000|eq)))
    for game,hidden,value in itertools.product([10,30,0x8000001E],[0,1,0xFFFFFFFF],[0,1,0x80000000]):
        cases.append(m.run(dict(state_bits=5,game_state_bits=game,hidden_bits=0xABCD123400000000|hidden,value_bits=0x1234567800000000|value)))
    for active,acted_empty,description_empty in itertools.product([0,0x80],[0,0xFF],[0,0x80]):
        for neq in [0,0x80]:cases.append(m.run(dict(left=0x80,active_bits=0xABCD123456789000|active,acted_empty_bits=0xABCD123456789000|acted_empty,description_empty_bits=0xABCD123456789000|description_empty,unity_inequality_bits=0xABCD123456789000|neq)))
    for options in [dict(null_acteds=True,left=1),dict(game_object=None,left=1),dict(null_left=True,left=1),dict(null_data=True,null_bluff=True),
                    dict(null_history=True),dict(item_results=[None]),dict(item_results=['info0',None]),dict(item_results=['info0','info1',None]),
                    dict(null_info_characters=True),dict(null_characters=True),dict(state_bits=5,null_player=True),dict(state_bits=5,null_blocks=True),dict(state_bits=5,null_value=True),
                    dict(description_result=None),dict(get_acted_results=[None],left=1),dict(alias_data=True,alias_acteds=True,left=1),
                    dict(warm_byte=0xFE,class_word=0xFFFFFFFF,unused_r8=0xFEDCBA9876543210,unused_r9=0x123456789ABCDEF0),
                    dict(state_bits=5,warm_byte=0xFE,class_word=0xFFFFFFFF),dict(prev_state_bits=5,killed=0x80,warm_byte=0xFE),dict(null_actions=[0x10,0x20,0x28,0x90]),
                    dict(action_target=None),dict(action_target='actor'),dict(action_target='other_actor'),
                    dict(null_role=True,null_bluff=True),dict(null_role=True,null_bluff=True,show_disguise=0x80),
                    dict(state_bits=0x80000005,prev_state_bits=5,killed=0x80),dict(prev_state_bits=0x80000005,killed=0x80),
                    dict(left=1,get_acted_results=['acted_text','other_text','description']),
                    dict(left=1,get_acted_results=['acted_text',None,'other_text']),dict(left=1,get_acted_results=['acted_text','other_text',None]),
                    dict(item_results=['info0','info1','info2'],mutations={'history_item:2':[['info1',0x18,8,'other_characters_list']]}),
                    dict(mutations={'history_item:2':[['actor',0x148,8,0]]}),dict(mutations={'history_item:1':[['history',0x18,4,0]]}),
                    dict(show_disguise=0x80,mutations={'unity_inequality:1':[['actor',0x58,8,0]]}),
                    dict(mutations={'unity_equality:1':[['ui_static',0x90,8,0],['actor',0x58,8,0],['actor',0x50,8,0]]}),
                    dict(state_bits=5,allocation_hint='hint1'),dict(prev_state_bits=5,killed=0x80,allocation_hint='hint1'),
                    dict(state_bits=5,null_actions=[0x10]),dict(prev_state_bits=5,killed=0x80,null_actions=[0x10]),
                    dict(description_result='literal_empty'),dict(left=1,get_acted_results=['literal_empty']),
                    dict(show_disguise=0x80,null_actions=[0x20]),dict(state_bits=5,action_target=None)]:cases.append(m.run(options))
    plans=[('class_initialization:1',[['actor',0x58,8,'other_data']]),('unity_inequality:1',[['actor',0x58,8,0]]),
           ('component_game_object:1',[['actor',0xA8,8,0]]),('get_acted:1',[['actor',0xA8,8,'other_acteds']]),
           ('get_acted:2',[['actor',0xB8,8,'other_left']]),('get_acted:3',[['actor',0xB8,8,0]]),
           ('saved_act_barrier:1',[['actor',0x198,8,'other_text'],['actor',0xB8,8,'other_left']]),
           ('act:1',[['actor',0xA8,8,0]]),('get_description:1',[['actor',0x38,8,'other_pivot']]),
           ('action:1',[['actor',0x148,8,'other_history']]),('history_item:1',[['actor',0x148,8,'other_history'],['other_history',0x18,4,2]]),
           ('history_item:1',[['info0',0x18,8,0]]),('history_item:2',[['info0',0x18,8,0]]),
           ('history_item:3',[['characters_static',0,8,0],['info0',0x18,8,0]]),
           ('highlight:1',[['actor',0x148,8,0]]),('unity_equality:1',[['actor',0x58,8,'other_data'],['actor',0x50,8,'other_data']]),
           ('allocate_hint:1',[['ui_static',0x10,8,'replacement_action'],['actor',0x38,8,'other_pivot']]),
           ('allocate_hint:1',[['slot:literal_block',0,8,'other_text'],['slot:literal_killed',0,8,'description'],['slot:literal_empty',0,8,'other_description']]),
           ('hint_constructor:1',[['hint0',0x18,8,'other_text']]),('class_initialization:1',[['slot:GameData_TypeInfo',0,8,'other_game_data_class']]),
           ('class_initialization:1',[['slot:UnityEngine.Object_TypeInfo',0,8,'other_object_class']]),
           ('highlight:1',[['object_class',0xE0,4,0],['actor',0x58,8,'other_data']]),
           ('metadata:12',[['actor',0xE4,4,5]])]
    for phase,writes in plans:
        for options in [dict(left=1),dict(state_bits=5),dict(prev_state_bits=5,killed=0x80),dict(history_count_bits=0,left=0,warm_byte=0xFE,class_word=0xFFFFFFFF)]:
            cases.append(m.run(dict(options,mutations={phase:writes})))
    for options in [{},dict(left=1),dict(state_bits=5),dict(prev_state_bits=5,killed=0x80)]:
        rows=[m.run(options),m.run(dict(options,mutations={'action:1':[['actor',0x1A0,1,0x80]]}),True),m.run(options,True)]
        assert all(prior['final']==following['initial'] for prior,following in zip(rows,rows[1:]))
        assert all(r['returned'] for r in rows);sequences.append(rows)
    profiles=[{},dict(left=1),dict(null_bluff=True,left=1),dict(show_disguise=0x80),dict(state_bits=5),dict(prev_state_bits=5,killed=0x80),
              dict(left=1,mutations={'saved_act_barrier:1':[['actor',0x198,8,'other_text'],['actor',0xB8,8,'other_left']]}),
              dict(mutations={'history_item:2':[['info0',0x18,8,0]]}),dict(state_bits=5,mutations={'allocate_hint:1':[['ui_static',0x10,8,'replacement_action']]}),
              dict(mutations={'highlight:1':[['object_class',0xE0,4,0]],'class_initialization:2':[['actor',0x58,8,0]]}),
              dict(state_bits=5,null_actions=[0x10]),dict(prev_state_bits=5,killed=0x80,null_actions=[0x10])]
    for options in profiles:
        baseline=m.run(options);bid=len(baselines);baselines.append(baseline);counts={}
        for index,e in enumerate(baseline['events']):
            kind=e['kind'];counts[kind]=counts.get(kind,0)+1;stopped=m.run(dict(options,failure=[kind,counts[kind]]))
            assert not stopped['returned'] and stopped['events']==baseline['events'][:index+1] and stopped['final']==e['snapshot']
            stops.append(dict(baseline=bid,prefix_length=index+1,result=stopped))
    missing=sorted(set(m.instructions)-m.executed);assert missing==[0x369234],[hex(a) for a in missing]
    return dict(schema='character_show_description_native_v1',build=BUILD,symbol_key='tdi5487.m0056',target=m.target[0],
                body_bounds=dict(start=hex(BODY[0]),end_exclusive=hex(BODY[1]),next_managed=hex(BODY[2]),padding_bytes=11),
                body_identity=dict(byte_length=BODY[1]-BODY[0],instruction_count=len(m.instructions),sha256=hashlib.sha256(m.pe.get_data(BODY[0],BODY[1]-BODY[0])).hexdigest()),
                scope='Actual exact ShowDescription caller only; complete Acted/GetDescription/String/List/Unity/Action/Highlight/hidden-count/Value/allocation/HintInfo/runtime helpers supplied explicitly; no helper alias/native implementation promotion',
                fields=FIELDS,supplied_targets=m.supplied_targets,metadata_slots={n:hex(a) for n,a in m.slots.items()},metadata_flag=hex(m.flag),
                pinned_literals={hex(a):text for a,text in m.literal_values.items()},delay_bits=m.delay_bits,diagnostic_windows_not_object_extents=m.sizes,
                instruction_assertions=len(m.checks),decoded_instructions=len(m.instructions),covered_instructions=len(set(m.instructions)&m.executed),
                unexecuted_traps=[hex(a) for a in missing],cases=cases,retained_sequences=sequences,baselines=baselines,failure_stops=stops,
                summary=dict(cases=len(cases),sequences=len(sequences),baselines=len(baselines),stops=len(stops)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--game-root',required=True);parser.add_argument('--dumper-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=pool_snapshots(pool_memory(audit(args.game_root,args.dumper_root)))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=True)+'\n',encoding='utf-8')
    print(json.dumps(report['summary'],sort_keys=True))
