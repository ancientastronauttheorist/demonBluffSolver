"""Uncovered Gameplay iterator factories, delayed deck continuation and Reset paths."""
import argparse,hashlib,itertools,json,struct
from pathlib import Path
from audit_character_assets import BUILD

def audit(game_root,dumper_root):
 import capstone,pefile,unicorn
 from unicorn import x86_const as x
 assert unicorn.__version__=='2.1.4'
 repo=Path(__file__).parents[1];lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'));ext=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
 def pin(p,h):
  b=p.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==h.upper();return b
 raw=pin(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256']);meta=json.loads(pin(Path(dumper_root)/'script.json',ext['outputs']['script_json']['sha256']).decode('utf-8-sig'))
 decl=[];factories={'DelayedDeckIntro':0x37BDE0,'InitCoroutine':0x37DEA0,'SetupDelay':0x380C70};suffix={'DelayedDeckIntro':50,'InitCoroutine':41,'SetupDelay':47};resets={'DelayedDeckIntro':0x38FA90,'InitCoroutine':0x38FE50,'SetupDelay':0x391290}
 for n,a in factories.items():
  r=next(r for r in meta['ScriptMethod'] if r['Name']=='Gameplay$$'+n);assert r['Address']==a and r['Signature']==f'System_Collections_IEnumerator_o* Gameplay__{n} (Gameplay_o* __this, const MethodInfo* method);';decl.append(r)
  r=next(r for r in meta['ScriptMethod'] if r['Name']==f'Gameplay.<{n}>d__{suffix[n]}$$System.Collections.IEnumerator.Reset');assert r['Address']==resets[n] and r['Signature']==f'void Gameplay__{n}_d__{suffix[n]}__System_Collections_IEnumerator_Reset (Gameplay__{n}_d__{suffix[n]}_o* __this, const MethodInfo* method);';decl.append(r)
 r=next(r for r in meta['ScriptMethod'] if r['Name']=='Gameplay.<DelayedDeckIntro>d__50$$MoveNext');assert r['Address']==0x38F9C0 and r['Signature']=='bool Gameplay__DelayedDeckIntro_d__50__MoveNext (Gameplay__DelayedDeckIntro_d__50_o* __this, const MethodInfo* method);';decl.append(r)
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
 ends={0x38F9C0:0x38FA88,0x37BDE0:0x37BE2B,0x37DEA0:0x37DEEB,0x380C70:0x380CD6,0x38FA90:0x38FACE,0x38FE50:0x38FE8E,0x391290:0x3912CE,0x33ED50:0x33ED53};decoded={}
 for a,b in ends.items():
  buf=pe.get_data(a,b-a);assert len(buf)==b-a;ins=list(cs.disasm(buf,a));assert ins[-1].address+ins[-1].size==b and ins[-1].mnemonic!='int3';decoded.update({i.address:i for i in ins})
 checks=[(0x38F9FD,'mov','dword ptr [rdi + 0x10], 0xffffffff'),(0x38FA3C,'mov','dword ptr [rdi + 0x10], 1'),(0x38FA50,'mov','dword ptr [rdi + 0x10], 0xffffffff'),(0x38FA5C,'cmp','eax, 1'),(0x380CB9,'mov','qword ptr [rcx], rdi'),(0x33ED50,'ret','0')]
 for a,m,o in checks:assert (decoded[a].mnemonic,decoded[a].op_str)==(m,o)
 assert struct.unpack('<f',pe.get_data(0x1F34B18,4))[0]==1.0
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
 arena,stack,stop=0x200000000,0x300000000,0x400000000;uc.mem_map(arena,0x100000);uc.mem_map(stack,0x10000);uc.mem_map(stop,0x1000)
 def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
 def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
 def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
 def ri(a):return struct.unpack('<i',uc.mem_read(a,4))[0]
 def reg(r):return uc.reg_read(r)
 def ret(v=0):
  sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v&0xffffffffffffffff);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
 names=['Gameplay_TypeInfo','UnityEngine.WaitForSeconds_TypeInfo','System.NotSupportedException_TypeInfo']+[f'Gameplay.<{n}>d__{suffix[n]}_TypeInfo' for n in factories];types={n:arena+0x1000+i*0x400 for i,n in enumerate(names)};found=set()
 for r in meta['ScriptMetadata']:
  if r['Name'] in types:q(base+r['Address'],types[r['Name']]);found.add(r['Name'])
 assert found==set(types),set(types)-found
 tokens={}
 for n in factories:
  name=f'Method$Gameplay.<{n}>d__{suffix[n]}.System.Collections.IEnumerator.Reset()';r=next(r for r in meta['ScriptMetadataMethod'] if r['Name']==name);tokens[n]=arena+0x4000+len(tokens)*0x100;q(base+r['Address'],tokens[n])
 for i in decoded.values():
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
 iterator,receiver,stale=arena+0x6000,arena+0x7000,arena+0x8000;opt={};state={};visited=set()
 def halt(n):state['error']=n;uc.emu_stop()
 def event(n,**kw):
  state['events'].append({'event':n,'iterator_state':ri(iterator+0x10),'current':rq(iterator+0x18),**kw})
  if opt.get('fail')==n:halt(n);return False
  return True
 def hook(_,a,size,__):
  r=a-base;c,t,m=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
  if a==stop:uc.emu_stop();return
  if r==0x2B7B40:
   assert rq(c) in set(types.values())|set(tokens.values());ret(rq(c))
  elif r==0x2B7D40:
   if event('allocate',type=next(n for n,p in types.items() if p==c)):
    state['alloc']+=1;p=arena+0x10000+state['alloc']*0x100;uc.mem_write(p,bytes(0x80));q(p,c);state['allocated'].append(p);ret(p)
  elif r==0x1C961F0:
   assert m==0 and (reg(x.UC_X86_REG_XMM1)&0xffffffff)==0x3f800000
   if event('wait_ctor',seconds=1.0):d(c+0x10,0x3f800000);ret()
  elif r==0x2B6FF0:
   state['barriers'].append(c);event('barrier',address=c);ret()
  elif r==0x3C6F20:
   assert c==0
   if event('blind_deck'):ret(opt.get('blind',0))
  elif r==0x281D90:
   assert c==types['Gameplay_TypeInfo']
   if event('class_init'):d(c+0xE0,1);ret()
  elif r==0x37B620:
   assert c==8 and t==0
   if event('change_state',value=c):ret()
  elif r==0x111B720:
   assert rq(c)==types['System.NotSupportedException_TypeInfo'] and t==0
   if event('exception_ctor'):ret()
  elif r==0x2B7D50:
   assert c==state['allocated'][-1] and t==tokens[opt['reset']];event('throw');halt('not_supported')
  elif r in decoded:visited.add(r)
  else:raise AssertionError(hex(r))
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 cases=[]
 def run(entry,options=None,preserve=False):
  opt.clear();opt.update(options or {});state.clear();state.update(events=[],error=None,alloc=0,allocated=[],barriers=[])
  if not preserve:d(iterator+0x10,opt.get('state',0));q(iterator+0x18,stale)
  d(types['Gameplay_TypeInfo']+0xE0,0 if opt.get('cold') else 1)
  sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,(0 if opt.get('null_receiver') else receiver) if entry in factories.values() else iterator);uc.reg_write(x.UC_X86_REG_RDX,0)
  keep=[x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
  for rr in keep:uc.reg_write(rr,0xABC000+rr)
  uc.emu_start(base+entry,stop,count=3000)
  if state['error'] is None:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8 and all(reg(rr)==0xABC000+rr for rr in keep)
  result={'entry':hex(entry),'options':dict(opt),'events':state['events'][:],'error':state['error'],'state':ri(iterator+0x10),'current':rq(iterator+0x18),'result_bool':bool(reg(x.UC_X86_REG_RAX)&0xff)};cases.append(result);return result
 for n,entry in factories.items():
  for null in (False,True):
   r=run(entry,{'null_receiver':null});p=reg(x.UC_X86_REG_RAX);assert r['error'] is None and ri(p+0x10)==0 and rq(p+0x18)==0
   assert rq(p)==types[f'Gameplay.<{n}>d__{suffix[n]}_TypeInfo'] and rq(p+0x20)==(receiver if n=='SetupDelay' and not null else 0)
   assert state['barriers']==([p+0x20] if n=='SetupDelay' else [])
  r=run(entry,{'fail':'allocate'});assert r['error']=='allocate' and not state['allocated']
 for blind,cold in itertools.product((-2147483648,-1,0,1,2,2147483647),(False,True)):
  first=run(0x38F9C0,{'state':0,'blind':blind,'cold':cold});assert first['result_bool'] and first['state']==1 and [v['event'] for v in first['events']]==['allocate','wait_ctor','barrier'];current=first['current'];assert current!=stale
  second=run(0x38F9C0,{'blind':blind,'cold':cold},True);assert not second['result_bool'] and second['state']==-1 and second['current']==current
  assert [v['event'] for v in second['events']]==['blind_deck']+((['class_init'] if cold else [])+['change_state'] if blind!=1 else [])
  third=run(0x38F9C0,{},True);assert not third['events'] and not third['result_bool'] and third['current']==current
 for value in (-2147483648,-2,-1,2,2147483647):
  r=run(0x38F9C0,{'state':value});assert r['state']==value and r['current']==stale and not r['events'] and not r['result_bool']
 for fail in ('allocate','wait_ctor'):
  r=run(0x38F9C0,{'state':0,'fail':fail});assert r['error']==fail and r['state']==-1 and r['current']==stale
 for fail in ('blind_deck','class_init','change_state'):
  r=run(0x38F9C0,{'state':1,'cold':True,'fail':fail});assert r['error']==fail and r['state']==-1 and r['current']==stale
 for n,entry in resets.items():
  for value in (-1,0,1):
   r=run(entry,{'reset':n,'state':value});assert r['error']=='not_supported' and r['state']==value and r['current']==stale and [v['event'] for v in r['events']]==['allocate','exception_ctor','throw']
 return {'schema_version':1,'build_id':BUILD,'exact_declarations':decl,'cases_passed':len(cases),'distinct_native_instructions':len(visited),'native_relationships':len(checks),'wait_seconds':1.0,'cases':cases,'scope':'Seven uncovered factory/continuation/Reset declarations execute natively. Allocator, WaitForSeconds ctor, BlindDeck getter, class init, ChangeGameplayState, exception construction/throw and GC metadata are gateways. Existing SetupDelay/InitCoroutine MoveNext evidence is not duplicated. No scheduler advancement, live timing or cancellation semantics inferred.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(f"Passed {r['cases_passed']} iterator cases")
