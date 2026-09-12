"""Pinned Gameplay relic insertion and generic special-rule lookup callers."""
import argparse,hashlib,itertools,json,struct
from pathlib import Path
from audit_character_assets import BUILD

def audit(game_root,dumper_root):
 import capstone,pefile,unicorn
 from unicorn import x86_const as x
 repo=Path(__file__).parents[1];lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'));ext=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'));assert unicorn.__version__=='2.1.4'
 def pin(p,h):
  b=p.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==h.upper();return b
 raw=pin(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256']);meta=json.loads(pin(Path(dumper_root)/'script.json',ext['outputs']['script_json']['sha256']).decode('utf-8-sig'));dump=pin(Path(dumper_root)/'dump.cs',ext['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
 specs=[('Gameplay$$AddRelic',0x37B2A0,'void Gameplay__AddRelic (Gameplay_o* __this, RelicData_o* relic, const MethodInfo* method);'),('Gameplay$$GetSpecialRuleIfAble<object>',0x638DE0,'Il2CppObject* Gameplay__GetSpecialRuleIfAble_object_ (Gameplay_o* __this, const MethodInfo_638DE0* method);'),('Gameplay$$GetRuleOfType<__Il2CppFullySharedGenericType>',0x638C80,'SpecialRule_o* Gameplay__GetRuleOfType___Il2CppFullySharedGenericType_ (Gameplay_o* __this, const MethodInfo_638C80* method);')];exact=[]
 for n,a,s in specs:
  r=next(r for r in meta['ScriptMethod'] if r['Name']==n);assert r['Address']==a and r['Signature']==s;exact.append(r)
 declaration=dump.split('public class Gameplay ',1)[1].split('// Namespace:',1)[0]
 for field in ('public static List<RelicData> CurrentRelics; // 0x0','public List<SpecialRule> specialRules; // 0x88'):assert field in declaration
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
 ends={0x37B2A0:0x37B36B,0x638DE0:0x638F7B,0x638C80:0x638DDD,0x33ED50:0x33ED53};decoded={}
 for a,b in ends.items():
  data=pe.get_data(a,b-a);assert len(data)==b-a;ins=list(cs.disasm(data,a));assert ins[-1].address+ins[-1].size==b and ins[-1].mnemonic!='int3';decoded.update({i.address:i for i in ins})
 checks=[(0x37B306,'inc','dword ptr [rcx + 0x1c]'),(0x37B33C,'mov','dword ptr [rcx + 0x18], eax'),(0x37B353,'mov','qword ptr [rcx], rbx'),(0x638DF5,'mov','rsi, rdx'),(0x638C8F,'mov','rsi, rdx'),(0x638D81,'mov','rax, rdi'),(0x638F18,'mov','rax, r14')]
 for a,m,o in checks:assert (decoded[a].mnemonic,decoded[a].op_str)==(m,o)
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image());arena,stack,stop=0x200000000,0x300000000,0x400000000
 uc.mem_map(arena,0x100000);uc.mem_map(stack,0x10000);uc.mem_map(stop,0x1000)
 def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
 def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
 def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
 def rd(a):return struct.unpack('<I',uc.mem_read(a,4))[0]
 def reg(r):return uc.reg_read(r)
 def ret(v=0):
  sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
 refs=set()
 for i in decoded.values():
  for o in i.operands:
   if o.type==capstone.CS_OP_MEM and o.mem.base==capstone.x86.X86_REG_RIP:refs.add(i.address+i.size+o.mem.disp)
 bindings={}
 for r in meta['ScriptMetadata']+meta['ScriptMetadataMethod']:
  if r['Address'] in refs:p=arena+0x1000+len(bindings)*0x400;bindings[r['Name']]=p;q(base+r['Address'],p)
 assert 'Gameplay_TypeInfo' in bindings and 'Method$System.Collections.Generic.List<RelicData>.Add()' in bindings
 for i in decoded.values():
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
 game,static,items,collection,method,rgctx,kind,growth= [arena+a for a in (0x8000,0x9000,0xA000,0xB000,0xC000,0xD000,0xE000,0xF000)]
 q(bindings['Gameplay_TypeInfo']+0xB8,static);add_token=bindings['Method$System.Collections.Generic.List<RelicData>.Add()'];q(add_token+0x20,arena+0x10000);q(arena+0x10000+0xC0,arena+0x11000);q(arena+0x11000+0x70,growth)
 objects={i:arena+0x20000+i*0x100 for i in range(4)};labels={0:None}|{v:k for k,v in objects.items()};opt={};state={};visited=set();values=[]
 def event(n,**kw):
  state['counts'][n]=state['counts'].get(n,0)+1;state['events'].append({'event':n,**kw})
  if opt.get('fail')==[n,state['counts'][n]]:state['error']=n;uc.emu_stop();return False
  return True
 def hook(_,a,size,__):
  r=a-base;c,t,m=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
  if a==stop:uc.emu_stop();return
  if r==0x2B7B40:assert rq(c) in bindings.values();ret(rq(c))
  elif r==0x281D90:
   assert c==bindings['Gameplay_TypeInfo']
   if event('class_init'):d(c+0xE0,1);ret()
  elif r==0x2B6FF0:
   assert rq(c)==t;event('barrier',value=labels[t]);ret()
  elif r==0xB54090:
   assert c==collection and m==growth
   if event('grow',value=labels[t]):
    count=rd(c+0x18);q(items+0x20+count*8,t);d(items+0x18,count+4);d(c+0x18,count+1);ret()
  elif r==0x29C910:
   assert c==method
   if event('rgctx_init'):q(method+0x38,rgctx);ret()
  elif r==0x29C890:
   assert c==kind
   if event('type_init'):uc.mem_write(kind+0x135,b'\1');ret(kind)
  elif r==0xB16640:
   assert t==collection and m==bindings['Method$System.Collections.Generic.List<SpecialRule>.GetEnumerator()']
   if event('enumerator'):q(c,t);d(c+8,0);d(c+12,rd(t+0x1C));q(c+16,0);ret(c)
  elif r==0x9693D0:
   assert t==bindings['Method$System.Collections.Generic.List.Enumerator<SpecialRule>.MoveNext()']
   if event('move_next'):
    if rd(c+12)!=rd(collection+0x1C):state['error']='version';uc.emu_stop();return
    i=rd(c+8)
    if i<len(values):q(c+16,values[i]);d(c+8,i+1);ret(1)
    else:q(c+16,0);ret(0)
  elif r==0x2B7010:
   assert t==kind
   if event('is_instance',value=labels[c]):
    if opt.get('mutate_version'):d(collection+0x1C,rd(collection+0x1C)+1)
    answer=c if labels[c] in opt.get('matches',[1]) and c!=0 else 0
    if opt.get('second_cast_false') and state['counts']['is_instance']==2:answer=0
    ret(answer)
  elif r in (0x2B7D90,0x2B7D80,0x2B7040):state['error']={0x2B7D90:'null',0x2B7D80:'bounds',0x2B7040:'cast'}[r];uc.emu_stop()
  elif r in decoded:visited.add(r)
  else:raise AssertionError(hex(r))
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 cases=[]
 def run(name,options):
  opt.clear();opt.update(options);state.clear();state.update(events=[],counts={},error=None);values[:]=[0 if i is None else objects[i] for i in opt.get('rules',[])]
  q(game+0x88,0 if opt.get('null_list') else collection);q(static,0 if opt.get('null_list') else collection);q(collection+0x10,0 if opt.get('null_items') else items);d(collection+0x18,opt.get('count',1));d(collection+0x1C,opt.get('version',3));d(items+0x18,opt.get('capacity',4));q(items+0x20,objects[0])
  d(bindings['Gameplay_TypeInfo']+0xE0,0 if opt.get('cold') else 1);q(method+0x38,0 if opt.get('cold_rgctx') else rgctx);q(rgctx,kind);uc.mem_write(kind+0x135,b'\0' if opt.get('cold_type') else b'\1')
  sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,0 if name=='add' else game);uc.reg_write(x.UC_X86_REG_RDX,(0 if opt.get('null_relic') else objects[0]) if name=='add' else method)
  keep=[x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
  for rr in keep:uc.reg_write(rr,0xABC000+rr)
  uc.emu_start(base+{'add':0x37B2A0,'able':0x638DE0,'type':0x638C80}[name],stop,count=5000)
  if state['error'] is None:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8 and all(reg(rr)==0xABC000+rr for rr in keep)
  result={'method':name,'options':options,'events':state['events'][:],'error':state['error'],'count':rd(collection+0x18),'version':rd(collection+0x1C),'returned':labels.get(reg(x.UC_X86_REG_RAX)) if name!='add' and not state['error'] else None};cases.append(result);return result
 for cold,null_relic,capacity,version in itertools.product((False,True),(False,True),(1,4),(3,0xffffffff)):
  r=run('add',{'cold':cold,'null_relic':null_relic,'capacity':capacity,'version':version});assert r['error'] is None and r['count']==2 and r['version']==(version+1)&0xffffffff;assert rq(items+0x28)==(0 if null_relic else objects[0])
 for options,version in [({'null_list':True},3),({'null_items':True},4),({'cold':True,'fail':['class_init',1]},3),({'capacity':1,'fail':['grow',1]},4)]:
  r=run('add',options);assert r['error'] is not None and r['count']==1 and r['version']==version
 for name,rules,cold in itertools.product(('able','type'),([], [None],[0,1,2],[1,1,2],[None,2]),(False,True)):
  r=run(name,{'rules':rules,'cold_rgctx':cold,'cold_type':cold});assert r['error'] is None and r['returned']==(1 if 1 in rules else None)
  calls=sum(e['event']=='is_instance' for e in r['events']);assert calls==(rules.index(1)+1+int(name=='able') if 1 in rules else len(rules))
 for name in ('able','type'):
  assert run(name,{'null_list':True})['error']=='null'
  for fail in (['rgctx_init',1],['type_init',1],['is_instance',1],['move_next',2]):
   r=run(name,{'rules':[0,1],'cold_rgctx':True,'cold_type':True,'fail':fail});assert r['error']==fail[0]
  assert run(name,{'rules':[0,1],'mutate_version':True})['error']=='version'
  r=run(name,{'rules':[1,0],'mutate_version':True});assert r['error'] is None and r['returned']==1
 assert run('able',{'rules':[1],'second_cast_false':True})['error']=='cast'
 return {'schema_version':1,'build_id':BUILD,'exact_declarations':exact,'cases_passed':len(cases),'native_relationships':len(checks),'distinct_native_instructions':len(visited),'cases':cases,'scope':'Actual relic append fast path and both generic reference-return lookup bodies execute. Growth, metadata/type initialization, generic instance classification and versioned enumerator operations are explicit gateways. No relic uniqueness policy, ability/enabled predicate or hidden fully-shared value return ABI is inferred.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(f"Passed {r['cases_passed']} relic/rule cases")
