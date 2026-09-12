"""Native generic Character real-role filter with explicit runtime type services."""
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
 assert 'public CharacterData dataRef; // 0x50' in char_decl
 methods={'Characters$$FilterRealCharacterRole<object>':0x6028C0}
 exact=[r for r in meta['ScriptMethod'] if r['Name'] in methods];assert len(exact)==1 and exact[0]['Address']==0x6028C0
 assert exact[0]['Signature']=='System_Collections_Generic_List_Character__o* Characters__FilterRealCharacterRole_object_ (Characters_o* __this, System_Collections_Generic_List_Character__o* inpuCharacters, const MethodInfo_6028C0* method);'
 data_decl=dump.split('public class CharacterData ',1)[1].split('// Namespace:',1)[0];assert 'public Role role; // 0x140' in data_decl
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
 assert min(r['Address'] for r in meta['ScriptMethod'] if r['Address']>0x6028C0)==0x602AA0
 ends={0x6028C0:0x602A92,0x33ED50:0x33ED53};decoded={}
 for a,b in ends.items():
  buf=pe.get_data(a,b-a);assert len(buf)==b-a;ins=list(cs.disasm(buf,a));assert ins[-1].address+ins[-1].size==b;decoded.update({i.address:i for i in ins})
 for a,expected in {0x6028D3:('mov','r15, r8'),0x6029D3:('mov','rax, qword ptr [rdi + 0x50]'),0x6029E0:('mov','r14, qword ptr [rax + 0x140]'),0x6029EB:('mov','rax, qword ptr [rax]'),0x602A05:('call','0x2b7010'),0x33ED50:('ret','0')}.items():assert a in decoded and (decoded[a].mnemonic,decoded[a].op_str)==expected
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
 required=('System.Collections.Generic.List<Character>_TypeInfo','Method$System.Collections.Generic.List<Character>..ctor()','Method$System.Collections.Generic.List<Character>.GetEnumerator()','Method$System.Collections.Generic.List.Enumerator<Character>.MoveNext()','Method$System.Collections.Generic.List.Enumerator<Character>.Dispose()','Method$System.Collections.Generic.List<Character>.Add()')
 for n in required:assert n in bindings,n
 source,method,rgctx,target=arena+0x10000,arena+0x11000,arena+0x12000,arena+0x13000
 chars={i:arena+0x20000+i*0x200 for i in range(4)};data={i:arena+0x30000+i*0x200 for i in range(4)};roles={i:arena+0x40000+i*0x100 for i in range(4)};labels={0:None}|{p:i for i,p in chars.items()}|{p:i for i,p in roles.items()}
 lists={};versions={};opt={};state={};visited=set()
 def event(n,**kw):
  state['counts'][n]=state['counts'].get(n,0)+1;state['events'].append({'event':n,**kw})
  if opt.get('fail')==[n,state['counts'][n]]:state['error']=n;uc.emu_stop();return False
  return True
 def hook(_,a,size,__):
  r=a-base;c,t,m=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
  if a==stop:uc.emu_stop();return
  if r==0x2B7B40:
   assert c-base in refs
   if event('metadata'):ret()
  elif r==0x29C910:
   assert c==method
   if event('rgctx'):q(method+0x38,rgctx);ret()
  elif r==0x29C890:
   assert c==target
   if event('type_ready'):uc.mem_write(target+0x135,b'\1');ret(target)
  elif r==0x2B7D40:
   assert c==bindings['System.Collections.Generic.List<Character>_TypeInfo']
   if event('allocate'):
    p=arena+0x50000;lists[p]=[];versions[p]=0;state['allocated']=p;ret(0 if opt.get('null_allocation') else p)
  elif r==0xB02160:
   assert t==bindings['Method$System.Collections.Generic.List<Character>..ctor()']
   if event('constructor'):ret()
  elif r==0xB16640:
   assert m==bindings['Method$System.Collections.Generic.List<Character>.GetEnumerator()']
   if event('enumerator'):q(c,t);d(c+8,0);d(c+12,versions[t]);q(c+16,0);ret(c)
  elif r==0x9693D0:
   assert t==bindings['Method$System.Collections.Generic.List.Enumerator<Character>.MoveNext()']
   if not event('move_next'):return
   l=rq(c)
   if rd(c+12)!=versions[l]:state['error']='version';uc.emu_stop();return
   i=rd(c+8);values=lists[l]
   if i<len(values):q(c+16,values[i]);d(c+8,i+1);ret(1)
   else:q(c+16,0);ret(0)
  elif r==0x2B7010:
   assert t==target
   if event('isinst',role=labels[c]):
    if opt.get('mutate_on_classify'):versions[source]+=1
    ret(c if c and labels[c] in opt.get('matching',[0,2]) else 0)
  elif r==0x2EB0:
   assert m==bindings['Method$System.Collections.Generic.List<Character>.Add()'] and c==state['allocated']
   if event('add',value=labels[t]):
    lists[c].append(t);versions[c]+=1
    if opt.get('mutate_source'):versions[source]+=1
    ret()
  elif r==0x33ED50:
   assert t==bindings['Method$System.Collections.Generic.List.Enumerator<Character>.Dispose()']
   visited.add(r) # execute exact ret0, including preserved RAX
  elif r==0x2B7D90:state['error']='null';uc.emu_stop()
  elif r in decoded:visited.add(r)
  else:raise AssertionError(hex(r))
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 cases=[]
 def run(values,options=None):
  opt.clear();opt.update(options or {});lists.clear();versions.clear();state.clear();state.update(error=None,events=[],counts={},allocated=None)
  lists[source]=[chars[i] if i is not None else 0 for i in (values or [])];versions[source]=0
  q(method+0x38,0 if opt.get('cold_metadata') else rgctx);q(rgctx,target);uc.mem_write(target+0x135,b'\0' if opt.get('cold_type') else b'\1')
  for i,p in chars.items():q(p+0x50,0 if opt.get('null_data')==i else data[i]);q(data[i]+0x140,0 if opt.get('null_role')==i else roles[i])
  before=lists[source][:];sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,0);uc.reg_write(x.UC_X86_REG_RDX,0 if values is None else source);uc.reg_write(x.UC_X86_REG_R8,method)
  keep=[x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
  for rr in keep:uc.reg_write(rr,0xABC000+rr)
  uc.emu_start(base+0x6028C0,stop,count=10000)
  assert lists[source]==before
  if state['error'] is None:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8 and all(reg(rr)==0xABC000+rr for rr in keep) and reg(x.UC_X86_REG_RAX)==(0 if opt.get('null_allocation') else state['allocated'])
  output=lists[state['allocated']] if state['allocated'] else []
  result={'input':values,'options':dict(opt),'events':state['events'][:],'output_prefix':[labels[p] for p in output],'source_version':versions[source],'error':state['error']};cases.append(result);return result
 for values,matching in itertools.product(([],[0,1,2,3],[0,0,1,2],[None],[0,None,2]),([], [0,2],[0,1,2,3])):
  r=run(values,{'matching':matching});prefix=values[:values.index(None)] if None in values else values;assert r['output_prefix']==[i for i in prefix if i in matching] and r['error']==('null' if None in values else None)
 for options in ({'cold_metadata':True},{'cold_type':True},{'cold_metadata':True,'cold_type':True}):
  r=run([0,1,2],options);assert r['output_prefix']==[0,2] and r['error'] is None
  assert sum(e['event']=='metadata' for e in r['events'])==(7 if options.get('cold_metadata') else 0)
  assert sum(e['event']=='type_ready' for e in r['events'])==int(options.get('cold_type',False))
 assert run(None)['error']=='null'
 r=run([0,1,2],{'null_data':1});assert r['error']=='null' and r['output_prefix']==[0]
 r=run([0,1,2],{'null_role':0});assert r['error'] is None and r['output_prefix']==[2]
 for options,prefix in (({'mutate_source':True},[0]),({'mutate_on_classify':True},[0]),({'mutate_on_classify':True,'matching':[]},[])):
  r=run([0,1,2],options);assert r['error']=='version' and r['output_prefix']==prefix
 for fail in (['allocate',1],['constructor',1],['enumerator',1],['move_next',1],['move_next',2],['isinst',1],['isinst',2],['add',1],['add',2],['metadata',1],['metadata',7],['rgctx',1],['type_ready',1]):
  r=run([0,2],{'cold_metadata':True,'cold_type':True,'fail':fail});assert r['error']==fail[0]
  assert r['output_prefix']==([0] if fail in (['move_next',2],['isinst',2],['add',2]) else [])
 # Constructor's null-receiver handling remains explicit: this synthetic
 # permissive ctor shows the caller checks output only upon the first match.
 assert run([],{'null_allocation':True})['error'] is None
 assert run([1],{'null_allocation':True})['error'] is None
 assert run([0],{'null_allocation':True})['error']=='null'
 return {'schema_version':1,'build_id':BUILD,'exact_declarations':exact,'cases_passed':len(cases),'distinct_native_instructions':len(visited),'metadata_bindings':sorted(bindings),'cases':cases,'scope':'Native generic role filter caller, exact rgctx ABI, metadata/type readiness, reference-preserving ordered output and null/version/failure prefixes. Managed allocation/constructor/enumerator/Add and runtime isinst are explicit fixture services. No isinst internals or managed exception unwinding claimed.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(f"Passed {r['cases_passed']} generic role-filter cases")
