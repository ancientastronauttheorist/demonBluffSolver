"""Native ActedVersion text, animation-ID and saved-scale helpers."""
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
 entries={'Awake':(0x35D860,0x35D8EB),'GetActed':(0x35D8F0,0x35D91A),'Start':(0x35DBF0,0x35DC38),'UpdateActed':(0x35DC40,0x35DC67),'.ctor':(0x33E820,0x33E827)}
 exact=[]
 for name,(a,b) in entries.items():
  rows=[r for r in meta['ScriptMethod'] if r['Name']=='ActedVersion$$'+name and r['Address']==a];assert len(rows)==1;exact+=rows
  ret='System_String_o*' if name=='GetActed' else 'void';symbol='ActedVersion___ctor' if name=='.ctor' else 'ActedVersion__'+name;extra=', System_String_o* description' if name=='UpdateActed' else ''
  assert rows[0]['Signature']==f'{ret} {symbol} (ActedVersion_o* __this{extra}, const MethodInfo* method);'
 block=dump.split('public class ActedVersion ',1)[1].split('// Namespace:',1)[0]
 for field in ('public TextMeshProUGUI blankText; // 0x28','public TextMeshProUGUI text; // 0x30','private string animationId; // 0x38','private Vector3 savedScale; // 0x40'):assert field in block
 assert 'private static readonly Vector3 zeroVector; // 0x0' in dump
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True;decoded={}
 for a,b in entries.values():
  nxt=min(r['Address'] for r in meta['ScriptMethod'] if r['Address']>a);ins=list(cs.disasm(pe.get_data(a,nxt-a),a))
  while ins[-1].mnemonic=='int3':ins.pop()
  assert ins[-1].address+ins[-1].size==b;assert all(i.address+i.size==j.address for i,j in zip(ins,ins[1:]));decoded.update({i.address:i for i in ins})
 for a,expected in {0x35D8F4:('mov','rcx, qword ptr [rcx + 0x28]'),0x35DC44:('mov','rcx, qword ptr [rcx + 0x28]'),0x35DC23:('movsd','xmm0, qword ptr [rdx]'),0x35DC2A:('movsd','qword ptr [rbx + 0x40], xmm0'),0x35DC2F:('mov','dword ptr [rbx + 0x48], eax')}.items():assert a in decoded and (decoded[a].mnemonic,decoded[a].op_str)==expected
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
 for r in meta['ScriptString']:
  if r['Address'] in refs:
   p=arena+0x1000+len(bindings)*0x300;bindings['literal:'+r['Value']]=p;q(base+r['Address'],p)
 for n in ('UnityEngine.Vector3_TypeInfo','int_TypeInfo','literal:{0}_actedAnim'):assert n in bindings,n
 owner,blank,visible,texttype,game,boxed,static=[arena+0x10000+n*0x1000 for n in range(7)]
 q(bindings['UnityEngine.Vector3_TypeInfo']+0xB8,static)
 strings={arena+0x20000:'old',arena+0x20100:'description',arena+0x20200:'blank value',arena+0x20300:'formatted'}
 opt={};state={};visited=set()
 def snap():return {'animation':None if rq(owner+0x38)==0 else strings[rq(owner+0x38)],'saved_scale':list(struct.unpack('<III',uc.mem_read(owner+0x40,12))),'blank':state['blank'],'visible':state['visible']}
 def event(n,**kw):
  state['counts'][n]=state['counts'].get(n,0)+1;state['events'].append({'event':n,**kw,'fields':snap()})
  if opt.get('fail')==[n,state['counts'][n]]:state['error']=n;uc.emu_stop();return False
  return True
 def hook(_,a,size,__):
  r=a-base;c,t,m=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
  if a==stop:uc.emu_stop();return
  if r==0x2B7B40:
   assert c-base in refs
   if event('metadata'):ret(rq(c))
  elif r==0x1C79FD0:
   assert c==owner and t==0
   if event('game_object'):ret(0 if opt.get('null_game') else game)
  elif r==0x1C81060:
   assert c==game and t==0
   if event('instance_id'):ret(opt.get('instance_id',-9)&0xffffffff)
  elif r==0x282580:
   assert c==bindings['int_TypeInfo']
   if event('box',bits=rd(t)):d(boxed+0x10,rd(t));ret(boxed)
  elif r==0xF74DF0:
   assert c==bindings['literal:{0}_actedAnim'] and t==boxed and m==0
   if event('format',bits=rd(boxed+0x10)):ret(0 if opt.get('null_format') else arena+0x20300)
  elif r==0x2B6FF0:
   assert c==owner+0x38 and rq(c)==t
   if event('barrier'):ret()
  elif r==0x1C79770:
   assert c==owner and t==0
   if event('base_constructor'):ret()
  elif a==stop+0x100:
   assert c==blank and t==0xABC1
   if event('get_text'):ret(0 if opt.get('null_result') else arena+0x20200)
  elif a==stop+0x110:
   assert c==blank and m==0xABC2
   if event('set_text',value=None if t==0 else strings[t]):state['blank']=None if t==0 else strings[t];ret()
  elif r==0x2B7D90:state['error']='null';uc.emu_stop()
  elif r in decoded:visited.add(r)
  else:raise AssertionError(hex(r))
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 cases=[]
 def run(name,options=None):
  opt.clear();opt.update(options or {});state.clear();state.update(events=[],counts={},error=None,blank='blank value',visible='visible value')
  uc.mem_write(owner,bytes(0x100));q(owner+0x28,0 if opt.get('null_blank') else blank);q(owner+0x30,visible);q(owner+0x38,arena+0x20000);uc.mem_write(owner+0x40,struct.pack('<III',1,2,3));q(blank,texttype)
  for o,p in [(0x548,stop+0x100),(0x550,0xABC1),(0x558,stop+0x110),(0x560,0xABC2)]:q(texttype+o,p)
  uc.mem_write(static,struct.pack('<III',*opt.get('zero_bits',[0,0,0])))
  for i in decoded.values():
   if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\0' if opt.get('cold_metadata') else b'\1')
  sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,owner);uc.reg_write(x.UC_X86_REG_RDX,0 if opt.get('null_description') else arena+0x20100)
  keep=[x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
  for rr in keep:uc.reg_write(rr,0xABC000+rr)
  before=bytes(uc.mem_read(owner,0x100));uc.emu_start(base+entries[name][0],stop+0x200,count=10000);after=bytes(uc.mem_read(owner,0x100));assert before[:0x38]==after[:0x38] and before[0x4C:]==after[0x4C:]
  if state['error'] is None:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8 and all(reg(rr)==0xABC000+rr for rr in keep)
  result={'method':name,'options':dict(opt),'events':state['events'][:],'fields':snap(),'error':state['error'],'return_text':(None if reg(x.UC_X86_REG_RAX)==0 else strings[reg(x.UC_X86_REG_RAX)]) if name=='GetActed' and state['error'] is None else None};cases.append(result);return result
 for name in entries:
  baseline=run(name);assert baseline['error'] is None
  assert [e['event'] for e in baseline['events']]=={'Awake':['game_object','instance_id','box','format','barrier'],'GetActed':['get_text'],'UpdateActed':['set_text'],'Start':[],'.ctor':['base_constructor']}[name]
  if name=='Awake':assert baseline['fields']['animation']=='formatted' and baseline['events'][2]['bits']==0xfffffff7
  if name=='Start':assert baseline['fields']['saved_scale']==[0,0,0]
  if name=='GetActed':assert baseline['return_text']=='blank value'
  if name=='UpdateActed':assert baseline['fields']['blank']=='description' and baseline['fields']['visible']=='visible value'
  counts={}
  for i,e in enumerate(baseline['events']):
   n=e['event'];counts[n]=counts.get(n,0)+1;r=run(name,{'fail':[n,counts[n]]});assert r['error']==n and r['fields']==e['fields'] and r['events']==baseline['events'][:i+1]
 for name in ('GetActed','UpdateActed'):assert run(name,{'null_blank':True})['error']=='null'
 assert run('GetActed',{'null_result':True})['return_text'] is None
 assert run('UpdateActed',{'null_description':True})['fields']['blank'] is None
 assert run('Awake',{'null_game':True})['error']=='null'
 assert run('Awake',{'null_format':True})['fields']['animation'] is None
 for value in (-2147483648,0,2147483647):
  r=run('Awake',{'instance_id':value});assert r['events'][2]['bits']==value&0xffffffff
 for bits in ([0,0,0],[0x80000000,0x7FC12345,1],[0x3F800000,0x40000000,0x40400000]):
  r=run('Start',{'zero_bits':bits});assert r['fields']['saved_scale']==bits and not r['events']
 for name in ('Awake','Start'):
  r=run(name,{'cold_metadata':True});count=sum(e['event']=='metadata' for e in r['events']);assert count==(2 if name=='Awake' else 1)
  for n in range(1,count+1):
   r=run(name,{'cold_metadata':True,'fail':['metadata',n]});assert r['error']=='metadata' and r['fields']['animation']=='old' and r['fields']['saved_scale']==[1,2,3]
 return {'schema_version':1,'build_id':BUILD,'exact_declarations':exact,'cases_passed':len(cases),'distinct_native_instructions':len(visited),'metadata_bindings':sorted(bindings),'cases':cases,'scope':'Five exact ActedVersion helper declarations including shared ctor alias. Native blankText forwarding, instance-ID format store/barrier and raw Vector3.zeroVector copy. Text/engine/boxing/format/metadata/base constructor remain explicit; Show and Animate deferred.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(f"Passed {r['cases_passed']} ActedVersion cases")
