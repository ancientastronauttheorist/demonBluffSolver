"""Audit managed Mutant separately from the public Mutant/Skinwalker asset.

Execute the native selector with explicit list/filter/status/RNG gateways. The
execution evidence covers caller order and arguments; gateways are authored
models, not native execution of those callees or Unity RNG state recovery.
"""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from audit_spy import ASSET_HASHES, BUILD


def audit(game_root, dumper_root):
    import capstone
    import pefile
    import unicorn
    import UnityPy
    from unicorn import x86_const as x
    if unicorn.__version__ != "2.1.4":
        raise ValueError("requires Unicorn 2.1.4")
    repo = Path(__file__).parents[1]; root = Path(game_root); dumper = Path(dumper_root)
    manifest = json.loads((repo/f"manifests/builds/{BUILD}.json").read_text(encoding="utf-8"))
    extraction = json.loads((repo/f"manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json").read_text(encoding="utf-8"))
    def pinned(path, digest):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest().upper() != digest.upper(): raise ValueError(f"hash mismatch: {path.name}")
        return data
    native = pinned(root/'GameAssembly.dll', manifest['inputs']['game_assembly']['sha256'])
    pinned(root/'Demon Bluff_Data/il2cpp_data/Metadata/global-metadata.dat', manifest['inputs']['global_metadata']['sha256'])
    dump = pinned(dumper/'dump.cs', extraction['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    script = json.loads(pinned(dumper/'script.json', extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    block = re.search(r'public class Mutant : Role // TypeDefIndex: 5903.*?\n\}', dump, re.S)
    expected_rvas = [0x3E4DB0, 0x3E4D50, 0x3E4CF0, 0x33ED50, 0x3E4BB0, 0x3CFFF0]
    if not block or [int(v,16) for v in re.findall(r'// RVA: (0x[0-9A-Fa-f]+)', block[0])] != expected_rvas:
        raise ValueError('managed Mutant declaration boundary changed')
    pe = pefile.PE(data=native, fast_load=True)
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); cs.detail = True
    instructions = {}
    for start, size in ((0x3E4BB0,0x130),(0x3E4CF0,0x5B),(0x3E4D50,0x5B),(0x3E4DB0,0x2C),(0x3C4CA0,0x20)):
        instructions.update({i.address:i for i in cs.disasm(pe.get_data(start,size),start)})
    checks = [(0x3E4C33,'call','0x37dc00'), (0x3E4C58,'lea','r8d, [r9 + 0xa]'),
              (0x3E4C5C,'call','0x369eb0'), (0x3E4C7D,'call','0x36a550'),
              (0x3E4CA2,'mov','r8, rbx'), (0x3E4CA5,'lea','edx, [r9 + 0x14]'),
              (0x3E4CA9,'call','0x363aa0'), (0x3E4CAE,'test','rdi, rdi'),
              (0x3E4CB3,'mov','edx, dword ptr [rdi + 0x18]'),
              (0x3E4CBB,'call','0x1c86600'), (0x3E4CD6,'jmp','0xb22150'),
              (0x3E4D31,'xor','r9d, r9d'), (0x3E4D34,'xor','r8d, r8d'), (0x3E4D3D,'call','0x35d5d0'),
              (0x3E4D91,'xor','r9d, r9d'), (0x3E4D94,'xor','r8d, r8d'), (0x3E4D9D,'call','0x35d5d0'),
              (0x3C4CA3,'mov','r9, qword ptr [rax + 0x210]'), (0x3C4CAA,'jmp','qword ptr [rax + 0x208]')]
    for address, mnemonic, operands in checks:
        i = instructions[address]
        if (i.mnemonic,i.op_str)!=(mnemonic,operands): raise ValueError(f'native relationship changed: {address:x}')
    strings = {s['Address']:s['Value'] for s in script['ScriptString']}
    for address in (0x3E4D2A,0x3E4D8A,0x3E4DD0):
        i = instructions[address]
        if strings.get(i.address+i.size+i.operands[1].mem.disp) != '': raise ValueError('expected empty role string')
    assets_path = root/'Demon Bluff_Data/sharedassets0.assets'
    pinned(assets_path, ASSET_HASHES['sharedassets0.assets'])
    asset = next(o for o in UnityPy.load(str(assets_path)).objects if o.path_id == 21592)
    raw = asset.get_raw_data(); asset_hash = hashlib.sha256(raw).hexdigest().upper()
    if asset_hash != 'A3A703FA630D1A3288E4C031C2EDB4C91C88924F59FC7B957960AAE575470653': raise ValueError('public Mutant asset changed')
    if asset.read_typetree(check_read=False)['m_Name'] != 'Mutant' or raw[0x284:0x28E] != b'Skinwalker':
        raise ValueError('public Mutant binding changed')
    # Native body is warmed: metadata and class initialization are explicit inputs.
    base = pe.OPTIONAL_HEADER.ImageBase
    uc = unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64)
    uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095); uc.mem_write(base,pe.get_memory_mapped_image())
    arena, stack, stop = 0x200000000,0x300000000,0x400000000
    uc.mem_map(arena,0x10000); uc.mem_map(stack,0x10000); uc.mem_map(stop,0x1000)
    def q(address,value): uc.mem_write(address,struct.pack('<Q',value))
    def d(address,value): uc.mem_write(address,struct.pack('<I',value))
    def slot(address):
        i=instructions[address]; return base+i.address+i.size+i.operands[1].mem.disp
    gameplay_type,characters_type,gameplay_static,characters_static=arena+0x100,arena+0x300,arena+0x500,arena+0x600
    gameplay,characters,actor,statuses=arena+0x700,arena+0x800,arena+0x1000,arena+0x1300
    combined,good,final=arena+0x2000,arena+0x2100,arena+0x2200
    q(slot(0x3E4BF9),gameplay_type); q(slot(0x3E4C38),characters_type)
    q(gameplay_type+0xB8,gameplay_static); d(gameplay_type+0xE0,1); q(gameplay_static+0x10,gameplay)
    q(characters_type+0xB8,characters_static); q(characters_static,characters); q(actor+0xF0,statuses)
    uc.mem_write(base+0x288C5E4,b'\x01')
    gateways={0x37DC00:'script',0x369EB0:'alignment',0x36A550:'bluffable',0x363AA0:'mad',0x1C86600:'range',0xB22150:'item'}
    state={}; corpus=[]
    def ret(value):
        rsp=uc.reg_read(x.UC_X86_REG_RSP); destination=struct.unpack('<Q',uc.mem_read(rsp,8))[0]
        uc.reg_write(x.UC_X86_REG_RAX,value); uc.reg_write(x.UC_X86_REG_RSP,rsp+8); uc.reg_write(x.UC_X86_REG_RIP,destination)
    def hook(_,address,size,__):
        rva=address-base
        if rva in gateways:
            event=gateways[rva]; state['events'].append(event)
            rcx,rdx,r8,r9=[uc.reg_read(r) for r in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8,x.UC_X86_REG_R9)]
            if event=='script':
                assert rcx==gameplay and rdx==0; ret(combined)
            elif event=='alignment':
                assert rcx==characters and rdx==combined and r8==10 and r9==0
                if None in state['occurrences']:
                    state['result']={'kind':'null_asset','occurrence_index':state['occurrences'].index(None)}; uc.emu_stop(); return
                state['good']=[(i,k) for i,k in enumerate(state['occurrences']) if state['assets'][k]['alignment']==10]; ret(good)
            elif event=='bluffable':
                assert rcx==characters and rdx==good and r8==0
                state['eligible']=[(i,k) for i,k in state['good'] if state['assets'][k]['bluffable']]
                d(final+0x18,len(state['eligible'])); ret(final)
            elif event=='mad':
                assert rcx==statuses and rdx==20 and r8==actor and r9==0
                assert struct.unpack('<Q',uc.mem_read(uc.reg_read(x.UC_X86_REG_RSP)+0x28,8))[0]==0
                ret(0)
            elif event=='range':
                assert rcx==0 and rdx==len(state['eligible']) and r8==0; ret(state['draw_index'])
            else:
                assert rcx==final and rdx==state['draw_index']
                if not state['eligible']:
                    state['result']={'kind':'empty_support'}; uc.emu_stop(); return
                i,k=state['eligible'][rdx]; state['result']={'kind':'selected','asset_id':k,'occurrence_index':i}; ret(arena+0x4000+k*0x100)
        elif rva not in instructions or not 0x3E4BB0<=rva<0x3E4CE0:
            raise ValueError(f'execution left audited selector/gateways at {rva:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE,hook)
    asset_fields=[{'asset_id':1,'alignment':10,'bluffable':True},{'asset_id':2,'alignment':20,'bluffable':True},
                  {'asset_id':3,'alignment':10,'bluffable':False},{'asset_id':4,'alignment':10,'bluffable':True}]
    lists=[[],[1],[1,1,4],[2,3],[1,2,3,1,4],[None],[1,None],[2,None,1],[4,1,4,1]]
    for occurrences in lists:
        eligible=[] if None in occurrences else [i for i,k in enumerate(occurrences) if k in (1,4)]
        for draw_index in range(max(1,len(eligible))):
            state.clear(); state.update(occurrences=occurrences,assets={a['asset_id']:a for a in asset_fields},events=[],draw_index=draw_index)
            rsp=stack+0x8008; q(rsp,stop)
            uc.reg_write(x.UC_X86_REG_RSP,rsp); uc.reg_write(x.UC_X86_REG_RCX,arena+0x3000); uc.reg_write(x.UC_X86_REG_RDX,actor)
            uc.reg_write(x.UC_X86_REG_R8,0); uc.reg_write(x.UC_X86_REG_RBX,0x11223344); uc.reg_write(x.UC_X86_REG_RDI,0x55667788)
            uc.emu_start(base+0x3E4BB0,stop,timeout=1_000_000,count=200)
            result=state.get('result')
            expected_events=['script','alignment'] if None in occurrences else ['script','alignment','bluffable','mad','range','item']
            if not result or state['events']!=expected_events: raise ValueError('native selector sequence incomplete')
            if result['kind']=='selected':
                assert uc.reg_read(x.UC_X86_REG_RIP)==stop and uc.reg_read(x.UC_X86_REG_RSP)==rsp+8
                assert uc.reg_read(x.UC_X86_REG_RBX)==0x11223344 and uc.reg_read(x.UC_X86_REG_RDI)==0x55667788
            corpus.append({'context':{'rule_version':'managed_mutant_selector_native_v1','assets':asset_fields,
                'script_occurrences':occurrences,'statuses':{'values':[10],'resistance':[],'target_position':7}},
                'draw_index':draw_index,'expected_selection':result,'mad_attempted':'mad' in state['events'],
                'rng_draw_count':int('range' in state['events']),'events':state['events']})
    return {'schema_version':1,'build_id':BUILD,'type_def_index':5903,'managed_method_count':6,
            'game_assembly_sha256':manifest['inputs']['game_assembly']['sha256'],'method_rvas':[hex(r) for r in expected_rvas],
            'native_relationships_verified':len(checks)+3,'description_and_both_clues':'empty text; clue reference lists are null',
            'native_caller_cases':len(corpus),'cases':corpus,
            'public_mutant_asset':{'path_id':21592,'size':len(raw),'object_sha256':asset_hash,'managed_role':'Skinwalker','role_name_offset':'0x284'},
            'emulation_scope':'Warm native Mutant.GetBluffIfAble caller; list/filter/status/RNG/index services are explicit gateways; exact argument and call ordering verified',
            'unresolved':['managed Mutant shipped-asset binding','native execution of gateway callees in this harness','Unity RNG state','public Skinwalker rules and broader Reveal composition']}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('game_root',type=Path)
    parser.add_argument('--dumper-root',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); report=audit(args.game_root,args.dumper_root)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"Verified {report['managed_method_count']} managed methods, {report['native_relationships_verified']} relationships and {report['native_caller_cases']} native caller cases")
