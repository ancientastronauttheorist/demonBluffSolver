"""Native Character filter tail and actual captured Unity-object predicate."""
import argparse,hashlib,itertools,json,struct
from pathlib import Path
from audit_character_assets import BUILD

def audit(game_root,dumper_root):
 import capstone,pefile,unicorn
 from unicorn import x86_const as x
 repo=Path(__file__).parents[1];lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'));ext=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'));assert unicorn.__version__=='2.1.4'
 def pin(p,h):
  b=p.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==h.upper();return b
 raw=pin(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256']);meta=json.loads(pin(Path(dumper_root)/'script.json',ext['outputs']['script_json']['sha256']).decode('utf-8-sig'))
 dump=pin(Path(dumper_root)/'dump.cs',ext['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
 char_decl=dump.split('public class Character ',1)[1].split('// Namespace:',1)[0]
 for field in ('public CharacterData dataRef; // 0x50','public ECharacterState state; // 0xE4','public CharacterStatuses statuses; // 0xF0'):assert field in char_decl
 status_decl=dump.split('public class CharacterStatuses ',1)[1].split('// Namespace:',1)[0];assert 'public List<ECharacterStatus> statuses; // 0x10' in status_decl
 methods={'Characters$$FilterCharacterContainsStatus':0x36A6D0,'Characters$$FilterRevealedCharacters':0x36BCD0,'Characters$$FilterNotInPlayCharactersUnique':0x36B3F0,'Characters.<>c__DisplayClass50_0$$<FilterNotInPlayCharactersUnique>b__0':0x377510}
 exact=[]
 for n,a in methods.items():
  rows=[r for r in meta['ScriptMethod'] if r['Name']==n];assert len(rows)==1 and rows[0]['Address']==a;exact+=rows
 expected_signatures=[
 'System_Collections_Generic_List_Character__o* Characters__FilterCharacterContainsStatus (Characters_o* __this, System_Collections_Generic_List_Character__o* inpuCharacters, int32_t status, const MethodInfo* method);',
 'System_Collections_Generic_List_Character__o* Characters__FilterRevealedCharacters (Characters_o* __this, System_Collections_Generic_List_Character__o* inputCharacters, const MethodInfo* method);',
 'System_Collections_Generic_List_CharacterData__o* Characters__FilterNotInPlayCharactersUnique (Characters_o* __this, System_Collections_Generic_List_CharacterData__o* inpuCharacters, const MethodInfo* method);',
 'bool Characters___c__DisplayClass50_0___FilterNotInPlayCharactersUnique_b__0 (Characters___c__DisplayClass50_0_o* __this, CharacterData_o* cd, const MethodInfo* method);']
 assert [r['Signature'] for r in exact]==expected_signatures
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
 ends={0x36A6D0:0x36A8B9,0x36BCD0:0x36BE44,0x36B3F0:0x36B65B,0x377510:0x37756D,0x33ED50:0x33ED53};decoded={}
 for a,b in ends.items():
  buf=pe.get_data(a,b-a);assert len(buf)==b-a;ins=list(cs.disasm(buf,a));assert ins[-1].address+ins[-1].size==b;decoded.update({i.address:i for i in ins})
 assert (decoded[0x377568].mnemonic,decoded[0x377568].op_str)==('jmp','0x1c822c0')
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image());arena,stack,stop=0x200000000,0x300000000,0x400000000
 uc.mem_map(arena,0x100000);uc.mem_map(stack,0x20000);uc.mem_map(stop,0x1000)
 def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
 def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
 def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
 def rd(a):return struct.unpack('<I',uc.mem_read(a,4))[0]
 def reg(r):return uc.reg_read(r)
 def ret(v=0):
  sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
 # Bind exact named metadata referred to by these bodies, including shared generic tokens.
 refs=set()
 for i in decoded.values():
  for o in i.operands:
   if o.type==capstone.CS_OP_MEM and o.mem.base==capstone.x86.X86_REG_RIP:refs.add(i.address+i.size+o.mem.disp)
 bindings={};rows=meta['ScriptMetadata']+meta['ScriptMetadataMethod']
 for r in rows:
  if r['Address'] in refs:
   p=arena+0x1000+len(bindings)*0x300;bindings[r['Name']]=p;q(base+r['Address'],p);d(p+0xE0,1)
 for n in ('Gameplay_TypeInfo','UnityEngine.Object_TypeInfo','System.Collections.Generic.List<Character>_TypeInfo','System.Collections.Generic.List<CharacterData>_TypeInfo','Characters.<>c__DisplayClass50_0_TypeInfo','System.Predicate<CharacterData>_TypeInfo'):assert n in bindings
 for i in decoded.values():
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
 game_static,source,board=arena+0x10000,arena+0x11000,arena+0x12000;q(bindings['Gameplay_TypeInfo']+0xB8,game_static)
 chars={i:arena+0x20000+i*0x200 for i in range(4)};data={i:arena+0x30000+i*0x200 for i in range(4)};labels={0:None}|{p:i for i,p in chars.items()}|{p:i for i,p in data.items()}
 lists={};versions={};predicates={};opt={};state={};visited=set()
 def event(n,**kw):
  state['counts'][n]=state['counts'].get(n,0)+1;state['events'].append({'event':n,**kw})
  if opt.get('fail')==[n,state['counts'][n]]:state['error']=n;uc.emu_stop();return False
  return True
 def hook(_,a,size,__):
  r=a-base;c,t,m=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
  if a in (stop,stop+0x100):uc.emu_stop();return
  if r==0x2B7D40:
   if event('allocate'):
    state['alloc']+=1;p=arena+0x50000+state['alloc']*0x100;q(p,c);lists[p]=[];versions[p]=0;state['allocated'].append(p);ret(p)
  elif r==0xB02160:
   assert t==bindings['Method$System.Collections.Generic.List<Character>..ctor()'];ret()
  elif r==0xB610A0:
   assert m==bindings['Method$System.Collections.Generic.List<CharacterData>..ctor()']
   if event('copy'):
    if t==0:state['error']='collection_null';uc.emu_stop()
    else:lists[c]=lists[t][:];ret()
  elif r==0xB16640:
   assert m==bindings['Method$System.Collections.Generic.List<Character>.GetEnumerator()']
   if event('enumerator'):
    q(c,t);d(c+8,0);d(c+12,versions[t]);q(c+16,0);ret(c)
  elif r==0x9693D0:
   assert t==bindings['Method$System.Collections.Generic.List.Enumerator<Character>.MoveNext()']
   if not event('move_next'):return
   l=rq(c)
   if rd(c+12)!=versions[l]:state['error']='version';uc.emu_stop();return
   i=rd(c+8);values=lists[l]
   if i<len(values):q(c+16,values[i]);d(c+8,i+1);ret(1)
   else:q(c+16,0);ret(0)
  elif r==0x2EB0:
   assert m==bindings['Method$System.Collections.Generic.List<Character>.Add()']
   if event('add',value=labels[t]):
    lists[c].append(t);versions[c]+=1
    if opt.get('mutate_source'):versions[source]+=1
    ret()
  elif r==0xB45070:
   assert m==bindings['Method$System.Collections.Generic.List<ECharacterStatus>.Contains()']
   if event('status_contains',status=t&0xffffffff):ret(int((t&0xffffffff) in lists[c]))
  elif r==0xB55950:
   assert m==bindings['Method$System.Collections.Generic.List<CharacterData>.Contains()']
   if event('managed_contains',value=labels[t]):ret(int(opt.get('contains_override',t in lists[c])))
  elif r==0xC8B620:
   assert m==bindings['Method$Characters.<>c__DisplayClass50_0.<FilterNotInPlayCharactersUnique>b__0()'];predicates[c]=t;ret()
  elif r==0xB59980:
   assert m==bindings['Method$System.Collections.Generic.List<CharacterData>.RemoveAll()']
   if event('remove_all'):state['pending']=(c,predicates[t]);uc.emu_stop()
  elif r==0x1C822C0:
   if event('unity_equal',left=labels[c],right=labels[t]):
    def canonical(p):return 0 if opt.get('unity_null_alias') and p==data[1] else p
    ret(int(canonical(c)==canonical(t)))
  elif r==0x281D90:
   if event('class_init'):d(c+0xE0,1);ret()
  elif r==0x2B6FF0:ret()
  elif r==0x2B7D90:state['error']='null';uc.emu_stop()
  elif r in decoded:visited.add(r)
  else:raise AssertionError(hex(r))
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 def execute(entry):
  uc.emu_start(base+entry,stop,count=10000)
  while state.get('pending') and not state['error']:
   target,capture=state.pop('pending');saved=uc.context_save();before=lists[target][:];keep=[]
   for value in before:
    sp=stack+0x18008;q(sp,stop+0x100);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,capture);uc.reg_write(x.UC_X86_REG_RDX,value);uc.reg_write(x.UC_X86_REG_R8,0)
    uc.emu_start(base+0x377510,stop+0x100,count=1000)
    if state['error']:break
    assert reg(x.UC_X86_REG_RIP)==stop+0x100 and reg(x.UC_X86_REG_RSP)==sp+8
    if not(reg(x.UC_X86_REG_RAX)&0xff):keep.append(value)
   uc.context_restore(saved)
   if state['error']:break
   lists[target]=keep;versions[target]+=int(len(keep)!=len(before));ret(len(before)-len(keep));uc.emu_start(reg(x.UC_X86_REG_RIP),stop,count=10000)
 cases=[]
 def run(name,values,options=None):
  opt.clear();opt.update(options or {});lists.clear();versions.clear();predicates.clear();state.clear();state.update(error=None,events=[],counts={},alloc=0,allocated=[])
  unique=name=='unique';lists[source]=[(data if unique else chars)[i] if i is not None else 0 for i in (values or [])];versions[source]=0
  lists[board]=[chars[i] if i is not None else 0 for i in opt.get('board',[0])];versions[board]=0;q(game_static+0x18,0 if opt.get('null_board') else board)
  for i,p in chars.items():
   d(p+0xE4,opt.get('states',[5,0,10,20])[i]);q(p+0x50,0 if opt.get('null_data') else data[i]);wrapper=arena+0x40000+i*0x100;statuslist=arena+0x41000+i*0x100;q(p+0xF0,0 if opt.get('missing')=='wrapper' else wrapper);q(wrapper+0x10,0 if opt.get('missing')=='status_list' else statuslist);lists[statuslist]=[10,i];versions[statuslist]=0
  d(bindings['Gameplay_TypeInfo']+0xE0,0 if opt.get('cold') else 1);d(bindings['UnityEngine.Object_TypeInfo']+0xE0,0 if opt.get('cold') else 1)
  before=lists[source][:];sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,0);uc.reg_write(x.UC_X86_REG_RDX,0 if values is None else source);uc.reg_write(x.UC_X86_REG_R8,opt.get('status',10)&0xffffffff)
  keep=[x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
  for rr in keep:uc.reg_write(rr,0xABC000+rr)
  execute({'status':0x36A6D0,'revealed':0x36BCD0,'unique':0x36B3F0}[name])
  assert lists[source]==before
  if state['error'] is None:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8 and all(reg(rr)==0xABC000+rr for rr in keep)
  output=lists[state['allocated'][0]] if state['allocated'] else []
  result={'method':name,'input':values,'options':dict(opt),'events':state['events'][:],'output_prefix':[labels[p] for p in output],'error':state['error']};cases.append(result);return result
 for values in ([],[0,1,2,3],[1,1,0,2],[None],[1,None,2]):
  r=run('revealed',values);prefix=values[:values.index(None)] if None in values else values;assert r['output_prefix']==[i for i in prefix if i!=0] and r['error']==('null' if None in values else None)
  for status in (-1,0,1,10,2147483647):
   r=run('status',values,{'status':status});assert r['output_prefix']==[i for i in prefix if status in (10,i)] and r['error']==('null' if None in values else None)
 for value in (-2147483648,-1,0,1,5,10,20,2147483647):
  r=run('revealed',[0,1,0],{'states':[value]*4});assert r['output_prefix']==([] if value==5 else [0,1,0]) and r['error'] is None
 for name in ('revealed','status'):
  assert run(name,None)['error']=='null'
  r=run(name,[1,2],{'mutate_source':True});assert r['error']=='version' and r['output_prefix']==[1]
  for fail in (['allocate',1],['move_next',2],['add',2]):assert run(name,[1,2],{'fail':fail})['error']==fail[0]
 for missing in ('wrapper','status_list'):assert run('status',[0],{'missing':missing})['error']=='null'
 for values,board_values in itertools.product(([],[0,1,0,2],[None,1,2],[1,1]),([], [0],[1],[0,1],[None],[0,None])):
  r=run('unique',values,{'board':board_values});out=values[:]
  for b in board_values:
   if b is None:break
   if b in out:out=[v for v in out if v!=b]
  assert r['output_prefix']==out and r['error']==('null' if None in board_values else None)
 assert run('unique',None)['error']=='collection_null';assert run('unique',[0],{'null_board':True})['error']=='null'
 r=run('unique',[None,1,2],{'null_data':True,'unity_null_alias':True});assert r['output_prefix']==[2]
 r=run('unique',[1,2],{'null_data':True,'unity_null_alias':True});assert r['output_prefix']==[1,2] # managed Contains gate is separate
 for fail in (['copy',1],['managed_contains',1],['remove_all',1],['unity_equal',2],['class_init',1]):
  r=run('unique',[0,1,0],{'cold':True,'fail':fail});assert r['error']==fail[0]
  assert r['output_prefix']==([] if fail[0]=='copy' else [0,1,0])
 return {'schema_version':1,'build_id':BUILD,'exact_declarations':exact,'cases_passed':len(cases),'distinct_native_instructions':len(visited),'metadata_bindings':sorted(bindings),'cases':cases,'scope':'All3 native callers plus captured predicate execute. Enumerators enforce supplied version, output Add versions increment; inputs otherwise stable. Contains and Unity-object equality are distinct gateways. RemoveAll invokes actual native predicate for every occurrence, committing authored removal only after all predicates succeed; native collection-internal partial writes/unwinding are not claimed.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(f"Passed {r['cases_passed']} filter-tail cases")
