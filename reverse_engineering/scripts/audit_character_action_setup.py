"""Actual Character.Act/RoleAct/CheckLying Init and Start, supplied role callbacks."""
import argparse,hashlib,itertools,json,struct
from pathlib import Path
from audit_character_assets import BUILD

def audit(game_root,dumper_root):
 import capstone,pefile,unicorn
 from unicorn import x86_const as x
 assert unicorn.__version__=='2.1.4'
 root=Path(__file__).parents[1];lock=json.loads((root/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'));ext=json.loads((root/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
 def pin(p,h):b=p.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==h.upper();return b
 raw=pin(game_root/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256']);meta=json.loads(pin(dumper_root/'script.json',ext['outputs']['script_json']['sha256']).decode('utf-8-sig'));dump=pin(dumper_root/'dump.cs',ext['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
 declarations={'public class Character : MonoBehaviour, ICard //':['public Role role; // 0x168','public Role bluffRole; // 0x170','public Action<Character, ETriggerPhase> onTrigger; // 0x110','private bool characterStartActed; // 0x11C','public EAlignment alignment; // 0xF8','public CharacterStatuses statuses; // 0xF0'],'public abstract class Role //':['public Action<ActedInfo> onActed; // 0x28']}
 for decl,fields in declarations.items():
  assert dump.count(decl)==1;block=dump.split(decl,1)[1].split('// Methods',1)[0]
  for field in fields:assert field in block,field
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True;decoded={};exact=[]
 for address,name in [(0x3645c0,'Character$$Act'),(0x368790,'Character$$RoleAct'),(0x397750,'CharacterHelper$$CheckLying')]:
  rows=[m for m in meta['ScriptMethod'] if m['Address']==address and m['Name']==name];assert len(rows)==1;exact+=rows
  end=min(m['Address'] for m in meta['ScriptMethod'] if m['Address']>address);ins=list(cs.disasm(pe.get_data(address,end-address),address))
  while ins[-1].mnemonic=='int3':ins.pop()
  assert all(a.address+a.size==b.address for a,b in zip(ins,ins[1:]));decoded.update({i.address:i for i in ins})
 noop=list(cs.disasm(pe.get_data(0x33ed50,3),0x33ed50));assert len(noop)==1 and (noop[0].mnemonic,noop[0].op_str)==('ret','0');decoded[0x33ed50]=noop[0]
 checks={0x364712:('call','qword ptr [rax + 0x18]'),0x364727:('mov','byte ptr [rbx + 0x11c], 1'),0x364733:('call','0x397750'),0x36476d:('mov','rdx, qword ptr [rbx + 0x170]'),0x36479b:('mov','rdx, qword ptr [rbx + 0x170]'),0x36884e:('mov','qword ptr [rcx], rbp'),0x36886e:('call','qword ptr [rax + 0x258]'),0x368896:('call','qword ptr [rax + 0x208]'),0x3977dd:('call','0x363c40'),0x3977fb:('call','0x363c40')}
 for a,v in checks.items():assert (decoded[a].mnemonic,decoded[a].op_str)==v
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
 arena,stack,stop=0x200000000,0x300000000,0x400000000
 uc.mem_map(arena,0x40000);uc.mem_map(stack,0x10000);uc.mem_map(stop,0x1000)
 def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
 def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
 def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
 def rd(a):return struct.unpack('<I',uc.mem_read(a,4))[0]
 def reg(r):return uc.reg_read(r)
 def ret(value=0):sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,value);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
 refs=set();flags=set()
 for i in decoded.values():
  for o in i.operands:
   if o.type==capstone.CS_OP_MEM and o.mem.base==capstone.x86.X86_REG_RIP:refs.add(i.address+i.size+o.mem.disp)
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:flags.add(i.address+i.size+i.operands[0].mem.disp)
 bindings={};slots={}
 for row in meta['ScriptMetadata']+meta['ScriptMetadataMethod']+meta['ScriptString']:
  if row['Address'] in refs:
   p=arena+0x1000+len(bindings)*0x400;name=row.get('Name',row.get('Value'));bindings[name]=p;slots[base+row['Address']]=p;q(base+row['Address'],p)
 for name in ['Gameplay_TypeInfo','System.Action<ActedInfo>_TypeInfo','Character.<>c__DisplayClass125_0_TypeInfo','Method$Character.<>c__DisplayClass125_0.<RoleAct>b__0()']:assert name in bindings,name
 actor,status,real,copied,replacement,subscriber,static,raw_bluff=[arena+i for i in range(0x10000,0x18000,0x1000)]
 labels={0:None,actor:'actor',status:'status',real:'real',copied:'copied',replacement:'replacement',subscriber:'subscriber',raw_bluff:'raw_bluff'}
 state={};opt={};visited=set();allocations={}
 def snap():return {'started':bool(uc.mem_read(actor+0x11c,1)[0]),'alignment':rd(actor+0xf8),'real':labels[rq(actor+0x168)],'copied':labels[rq(actor+0x170)],'statuses':state['statuses'][:],'on_acted':{labels[p]:allocations.get(rq(p+0x28)) for p in [real,copied,replacement]},'allocations':list(allocations.values()),'calls':state['calls'][:]}
 def emit(kind,**kw):
  state['counts'][kind]=state['counts'].get(kind,0)+1;state['events'].append({'kind':kind,**kw,'snapshot':snap()})
  if opt.get('fail')==[kind,state['counts'][kind]]:state['error']=kind;uc.emu_stop();return False
  return True
 def mutate(key):
  value=opt.get(key)
  if value=='replace_copy':q(actor+0x170,replacement)
  elif value=='clear_copy':q(actor+0x170,0)
  elif value=='add_copy':q(actor+0x170,copied)
  elif value=='corrupt':state['statuses'].append(10)
  elif value=='clean':state['statuses'][:]=[];d(actor+0xf8,10);q(actor+0x58,0)
  elif value=='reset_latch':uc.mem_write(actor+0x11c,b'\0')
  elif value=='set_latch':uc.mem_write(actor+0x11c,b'\1')
 def hook(_,a,size,__):
  r=a-base;c,t,m,n=[reg(rr) for rr in [x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8,x.UC_X86_REG_R9]]
  if a==stop+0x100:
   assert c==subscriber and t==actor and m==opt['trigger'] and n==subscriber+0x80
   if emit('subscriber'):mutate('subscriber_mutation');ret()
  elif a in [stop+0x200,stop+0x300]:
   route='act' if a==stop+0x200 else 'bluff_act';assert c in [real,copied,replacement] and t==opt['trigger'] and m==actor and n==(c+0x500 if route=='act' else c+0x580)
   if emit('role',role=labels[c],route=route):
    state['calls'].append({'role':labels[c],'route':route});
    if len(state['calls'])==1:mutate('first_role_mutation')
    ret()
  elif r==0x2b7b40:
   assert c in slots
   if emit('metadata'):ret(slots[c])
  elif r==0x281d90:
   assert c in bindings.values()
   if emit('class_init'):d(c+0xe0,1);ret()
  elif r==0x282580:
   assert c==bindings['ETriggerPhase_TypeInfo'] and rd(t)==opt['trigger']
   if emit('box'):ret(arena+0x19000)
  elif r==0xf74df0:
   assert t==arena+0x19000 and m==0
   if emit('format'):ret(arena+0x19100)
  elif r==0x1c4b450:
   assert c==arena+0x19100 and t==0
   if emit('log'):ret()
  elif r==0x1c82480:
   assert t==0 and m==0
   if emit('unity_live',raw=labels[c]):ret(int(bool(c) and opt.get('raw_live',False)))
  elif r==0x363c40:
   assert c==status and t in [30,10] and m==0
   if emit('has_status',status=t):ret(int(t in state['statuses']))
  elif r==0x2b7d40:
   kind='closure' if c==bindings['Character.<>c__DisplayClass125_0_TypeInfo'] else 'delegate';assert c==bindings['Character.<>c__DisplayClass125_0_TypeInfo'] or c==bindings['System.Action<ActedInfo>_TypeInfo']
   if emit('allocate',object=kind):
    p=arena+0x20000+len(allocations)*0x100;allocations[p]=kind+str(len(allocations));uc.mem_write(p,bytes(0x100));ret(p)
  elif r==0x2b6ff0:
   assert rq(c)==t
   if emit('barrier',target='actor_capture' if c-0x10 in allocations else labels[c-0x28]+'_on_acted'):ret()
  elif r==0x4d5b60:
   assert c in allocations and t in allocations and m==bindings['Method$Character.<>c__DisplayClass125_0.<RoleAct>b__0()'] and n==0
   assert rq(t+0x10)==actor and rd(t+0x18)==opt['trigger']
   if emit('delegate_ctor'):ret()
  elif r==0x2b7d90:state['error']='null';uc.emu_stop()
  else:assert r in decoded and decoded[r].size==size,hex(r);visited.add(r)
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 keep=[x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
 def run(options):
  opt.clear();opt.update(trigger=3,started=False,alignment=10,copied='copied',statuses=[],raw_live=False);opt.update(options);state.clear();state.update(counts={},events=[],calls=[],statuses=opt['statuses'][:],error=None);allocations.clear()
  uc.mem_write(actor,bytes(0x200));q(actor+0xf0,status);q(actor+0x168,0 if opt.get('null_real') else real);q(actor+0x170,{'copied':copied,'real':real,None:0}[opt['copied']]);q(actor+0x58,raw_bluff if opt.get('raw_live') else 0);d(actor+0xf8,opt['alignment']);uc.mem_write(actor+0x11c,bytes([opt['started']]));q(actor+0x110,subscriber if opt.get('subscriber') else 0)
  q(subscriber+0x18,stop+0x100);q(subscriber+0x28,subscriber+0x80);q(subscriber+0x40,subscriber)
  for p in [real,copied,replacement]:
   q(p,p+0x600);q(p+0x28,0);q(p+0x600+0x208,stop+0x200);q(p+0x600+0x210,p+0x500);q(p+0x600+0x258,stop+0x300);q(p+0x600+0x260,p+0x580)
  for p in bindings.values():d(p+0xe0,0 if opt.get('cold') else 1)
  q(bindings['Gameplay_TypeInfo']+0xb8,static);d(static+0x28,50)
  for flag in flags:uc.mem_write(base+flag,bytes([0 if opt.get('cold') else 1]))
  sp=stack+0x8008;q(sp,stop)
  for rr in keep:uc.reg_write(rr,0xabc000+rr)
  uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,actor);uc.reg_write(x.UC_X86_REG_RDX,opt['trigger']);uc.reg_write(x.UC_X86_REG_R8,0)
  uc.emu_start(base+0x3645c0,stop,count=10000);returned=reg(x.UC_X86_REG_RIP)==stop
  assert returned or state['error']
  if returned:assert reg(x.UC_X86_REG_RSP)==sp+8 and all(reg(rr)==0xabc000+rr for rr in keep)
  return {'input':dict(opt),'events':state['events'][:],'final':snap(),'error':state['error'],'returned':returned}
 cases=[]
 for trigger,started,alignment,copy,statuses,raw_live in itertools.product([3,5],[False,True],[10,20,30],[None,'copied','real'],[[],[10],[30],[10,30]],[False,True]):
  r=run(dict(trigger=trigger,started=started,alignment=alignment,copied=copy,statuses=statuses,raw_live=raw_live));assert r['returned']
  lying=10 in statuses or (30 not in statuses and (alignment==20 or raw_live))
  expected=[] if trigger==5 and started else [{'role':'real','route':'act' if not lying or alignment==20 and copy is not None else 'bluff_act'}]+([] if copy is None else [{'role':copy,'route':'bluff_act' if lying else 'act'}])
  assert r['final']['calls']==expected;cases.append(r)
 for trigger in [3,5]:
  baseline=run(dict(trigger=trigger,cold=True,subscriber=True));cases.append(baseline);counts={}
  for i,e in enumerate(baseline['events']):
   k=e['kind'];counts[k]=counts.get(k,0)+1;r=run(dict(trigger=trigger,cold=True,subscriber=True,fail=[k,counts[k]]));assert r['events']==baseline['events'][:i+1] and r['final']==e['snapshot'] and r['error']==k;cases.append(r)
 for key in ['subscriber_mutation','first_role_mutation']:
  for value in ['replace_copy','clear_copy','add_copy','corrupt','clean','reset_latch','set_latch']:
   r=run(dict(trigger=5,subscriber=True,**{key:value}));assert r['returned'];cases.append(r)
 r=run(dict(trigger=3,first_role_mutation='corrupt'));assert [c['route'] for c in r['final']['calls']]==['act','act'];cases.append(r)
 r=run(dict(trigger=5,alignment=20,first_role_mutation='clean'));assert [c['route'] for c in r['final']['calls']]==['act','bluff_act'];cases.append(r)
 r=run(dict(trigger=5,started=True,subscriber=True,subscriber_mutation='reset_latch'));assert len(r['final']['calls'])==2;cases.append(r)
 r=run(dict(trigger=3,null_real=True));assert r['error']=='null' and len(r['final']['allocations'])==2;cases.append(r)
 return {'build_id':BUILD,'metadata_verified':exact,'fields_verified':declarations,'cases_passed':len(cases),'native_assertions':len(checks)+1,'native_instructions_executed':len(visited),'cases':cases,'scope':'Actual Act/RoleAct/CheckLying for Init3 and Start5. Explicit role virtual gateways, status membership, Unity liveness, allocation/delegate construction, metadata/class init, barriers and logging. Supplied role/subscriber mutations are controlled effects, not concrete role implementations. Warm/cold failure stops preserve observed prefix; no unwind, clues, OnActed invocation, scheduler or engine semantics.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',required=True,type=Path)
 a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(r['cases_passed'],r['native_instructions_executed'])
