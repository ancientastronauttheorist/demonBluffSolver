"""Read the pinned core CharacterData prefix and resolve its managed role RID.

Serialized Boolean fields each occupy one byte plus alignment padding. Runtime
IL2CPP offsets are not serialized-file offsets. Managed-reference payloads after
registration headers remain opaque; full object deserialization is not claimed.
"""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from audit_spy import ASSET_HASHES, BUILD

FIELDS = ['characterId','localization_key','characterName','iWasName','gender','roguelikeInfo',
          'bundledCharacters','description','descriptionPL','descriptionCHN','flavorText','additionalFlavorTexts',
          'hints','ifLies','notes','art','art_cute','art_nice','art_animated','randomArt','backgroundArt','currentSkin',
          'skins','achievements','color','artBgColor','cardBgColor','cardBorderColor','additionalStatuses','tags',
          'canAppearIf','type','startingAlignment','abilityUsage','bluffable','usuallyDisguised','picking','role','translation']

class Cursor:
    def __init__(self, data, offset=28): self.data=data; self.offset=offset
    def read(self, fmt):
        size=struct.calcsize('<'+fmt)
        if self.offset+size>len(self.data): raise ValueError('truncated serialized value')
        value=struct.unpack_from('<'+fmt,self.data,self.offset);self.offset+=size
        return value[0] if len(value)==1 else value
    def align(self):
        end=(self.offset+3)&~3
        if end>len(self.data) or any(self.data[self.offset:end]): raise ValueError('unexpected alignment padding')
        self.offset=end
    def string(self):
        size=self.read('i')
        if not 0<=size<=len(self.data)-self.offset: raise ValueError('invalid serialized string length')
        value=self.data[self.offset:self.offset+size].decode('utf-8');self.offset+=size;self.align();return value
    def pointer(self): return self.read('iq')
    def array(self, reader):
        count=self.read('i')
        if not 0<=count<=10000: raise ValueError('invalid array count')
        return [reader() for _ in range(count)]
    def boolean(self):
        value=self.read('B')
        if value not in (0,1): raise ValueError('invalid Boolean value')
        self.align();return bool(value)


def parse_prefix(data, role_types):
    cursor=Cursor(data); result={}; offsets={}
    def field(name, reader):
        offsets[name]=hex(cursor.offset);value=reader();result[name]=value;return value
    for name in ('name','characterId','localization_key','characterName','iWasName'): field(name,cursor.string)
    field('gender',lambda:cursor.read('i'));field('roguelikeInfo',lambda:cursor.read('ifi'))
    field('bundledCharacters',lambda:cursor.array(cursor.pointer))
    # Read all variable-length fields to establish following offsets, but do
    # not publish authored descriptions, artwork or other asset content.
    for name in ('description','descriptionPL','descriptionCHN','flavorText'):field(name,cursor.string)
    field('additionalFlavorTexts',lambda:cursor.array(cursor.string))
    for name in ('hints','ifLies','notes'):field(name,cursor.string)
    for name in ('art','art_cute','art_nice','art_animated','randomArt','backgroundArt','currentSkin'):field(name,cursor.pointer)
    for name in ('skins','achievements'):field(name,lambda:cursor.array(cursor.pointer))
    for name in ('color','artBgColor','cardBgColor','cardBorderColor'):field(name,lambda:cursor.read('ffff'))
    for name in ('additionalStatuses','tags'):field(name,lambda:cursor.array(lambda:cursor.read('i')))
    field('canAppearIf',lambda:cursor.array(cursor.pointer))
    for name in ('type','startingAlignment','abilityUsage'):field(name,lambda:cursor.read('i'))
    for name in ('bluffable','usuallyDisguised','picking'):field(name,cursor.boolean)
    field('role_rid',lambda:cursor.read('q'));field('translation_rid',lambda:cursor.read('q'))
    registry_offset=cursor.offset
    version,count=cursor.read('ii')
    if version!=2 or not 1<=count<=128: raise ValueError('unexpected managed-reference registry header')
    # Join the explicit role RID to exactly one valid registration header,
    # rather than guessing a role from any string occurring in the object.
    pattern=struct.pack('<q',result['role_rid']);matches=[]
    for match in re.finditer(re.escape(pattern),data[cursor.offset:]):
        entry_offset=cursor.offset+match.start();candidate=Cursor(data,entry_offset+8)
        try:
            name,namespace,assembly=[candidate.string() for _ in range(3)]
            full_name=(namespace+'.' if namespace else '')+name
            if assembly=='Assembly-CSharp' and full_name in role_types:
                matches.append((entry_offset,full_name,role_types[full_name]))
        except (ValueError,UnicodeDecodeError,struct.error):
            continue
    if len(matches)!=1: raise ValueError('ambiguous or missing role RID registration')
    entry_offset,role_type,role_type_index=matches[0]
    published=('name','characterId','characterName','gender','roguelikeInfo','bundledCharacters','additionalStatuses',
               'tags','canAppearIf','type','startingAlignment','abilityUsage','bluffable','usuallyDisguised','picking',
               'role_rid','translation_rid')
    return {**{k:result[k] for k in published},'field_offsets':{k:offsets[k] for k in published},
            'registry_offset':hex(registry_offset),'registry_version':version,'registry_entry_count':count,
            'role_type':role_type,'role_type_def_index':role_type_index,'role_registration_offset':hex(entry_offset),
            'packed_byte_interpretation':{'usuallyDisguised':bool(data[int(offsets['bluffable'],16)+1]),
                                          'picking':bool(data[int(offsets['bluffable'],16)+2])}}


def audit(game_root,dumper_root):
    import UnityPy
    import capstone
    import pefile
    root=Path(game_root)/'Demon Bluff_Data';dumper=Path(dumper_root);repo=Path(__file__).parents[1]
    def pinned(path,expected):
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest().upper()!=expected.upper():raise ValueError(f'fingerprint changed: {path.name}')
        return raw
    extraction=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    build=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    native=pinned(Path(game_root)/'GameAssembly.dll',build['inputs']['game_assembly']['sha256'])
    pe=pefile.PE(data=native,fast_load=True)
    decoder=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64)
    flag_reads=[]
    for start,end,site,operand in [(0x3B7530,0x3B8640,0x3B7E36,'byte ptr [rdx + 0x13d], 0'),
            (0x3B63B0,0x3B7520,0x3B6A67,'byte ptr [r14 + 0x13d], 0'),
            (0x3DD410,0x3DD7B0,0x3DD643,'byte ptr [rdx + 0x13d], 0')]:
        instructions={i.address:i for i in decoder.disasm(pe.get_data(start,end-start),start)}
        instruction=instructions.get(site)
        if instruction is None or (instruction.mnemonic,instruction.op_str)!=('cmp',operand):
            raise ValueError('native usuallyDisguised field access changed')
        flag_reads.append({'entry_rva':hex(start),'field_read_rva':hex(site),'runtime_field_offset':'0x13D'})
    dump=pinned(dumper/'dump.cs',extraction['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    block=re.search(r'^public class CharacterData .*? // TypeDefIndex: 5845\n\{(.*?)\n\}',dump,re.M|re.S)
    if not block:raise ValueError('CharacterData declaration missing')
    fields=re.findall(r'^\s*public [^;\n]+ (\w+); // 0x[0-9A-F]+$',block[1].split('// Methods')[0],re.M)
    if fields!=FIELDS:raise ValueError('CharacterData serialized field order changed')
    types=json.loads((repo/f'symbols/{BUILD}/assembly_csharp_types.json').read_text(encoding='utf-8'))['types']
    role_types={(t['namespace']+'.' if t['namespace'] else '')+t['name']:t['type_def_index'] for t in types if 5853<=t['type_def_index']<=5923}
    for name in ('sharedassets0.assets','globalgamemanagers.assets'):pinned(root/name,ASSET_HASHES[name])
    manager=UnityPy.load(str(root/'globalgamemanagers.assets'))
    script=next(o for o in manager.objects if o.path_id==1576)
    mono=script.read_typetree()
    if script.type.name!='MonoScript' or [mono[k] for k in ('m_ClassName','m_Namespace','m_AssemblyName')]!=['CharacterData','','Assembly-CSharp']:
        raise ValueError('CharacterData MonoScript binding changed')
    environment=UnityPy.load(str(root/'sharedassets0.assets'))
    source=next(iter(environment.files.values()))
    if source.externals[0].path!='globalgamemanagers.assets':raise ValueError('script external-file binding changed')
    records=[]
    for obj in environment.objects:
        if obj.type.name!='MonoBehaviour':continue
        header=obj.read_typetree(check_read=False)
        if header.get('m_Script')!={'m_FileID':1,'m_PathID':1576}:continue
        raw=obj.get_raw_data();record=parse_prefix(raw,role_types)
        if record['name']!=header['m_Name']:raise ValueError('independent MonoBehaviour name disagreement')
        record.update(path_id=obj.path_id,object_size=len(raw),object_sha256=hashlib.sha256(raw).hexdigest().upper())
        records.append(record)
    records.sort(key=lambda r:r['path_id'])
    if [r['path_id'] for r in records]!=list(range(21590,21636)):raise ValueError('core CharacterData inventory changed')
    disguised=[r['name'] for r in records if r['usuallyDisguised']]
    picking=[r['name'] for r in records if r['picking']]
    return {'schema_version':1,'build_id':BUILD,'asset_sha256':ASSET_HASHES['sharedassets0.assets'],
            'game_assembly_sha256':build['inputs']['game_assembly']['sha256'],'native_flag_reads':flag_reads,
            'script_source_sha256':ASSET_HASHES['globalgamemanagers.assets'],'mono_script_path_id':1576,
            'record_count':len(records),'usually_disguised_count':len(disguised),'usually_disguised_assets':disguised,
            'picking_count':len(picking),'picking_assets':picking,'records':records,
            'scope':'Complete fixed CharacterData prefix and exact role-RID header join for pinned sharedassets0 core records; managed-reference payloads remain opaque',
            'correction':'Each serialized Boolean aligns independently to four bytes. The adjacent-byte interpretation reads padding and falsely reports usuallyDisguised/picking as false.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('game_root',type=Path)
    parser.add_argument('--dumper-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.game_root,args.dumper_root)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f"Verified {result['record_count']} core assets: {result['usually_disguised_count']} usually disguised, {result['picking_count']} picking")
