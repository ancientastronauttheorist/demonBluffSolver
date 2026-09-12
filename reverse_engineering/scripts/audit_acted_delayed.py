"""Native Acted delayed wrapper, iterator construction, resumption and Reset."""
import argparse,hashlib,itertools,json,struct
from pathlib import Path
from audit_character_assets import BUILD

def audit(game_root,dumper_root):
 import capstone,pefile,unicorn
 from unicorn import x86_const as x
 assert unicorn.__version__=='2.1.4';repo=Path(__file__).parents[1]
 lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'));ext=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
 def pin(p,h):
  b=p.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==h.upper();return b
 raw=pin(game_root/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256']);meta=json.loads(pin(dumper_root/'script.json',ext['outputs']['script_json']['sha256']).decode('utf-8-sig'));dump=pin(dumper_root/'dump.cs',ext['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
 for text in ['public ActedVersion acted; // 0x20','public RectTransform[] layoutsToRebuild; // 0x28','public float delay; // 0x20','public Acted <>4__this; // 0x28','public string description; // 0x30']:assert text in dump
 entries={'ActDelay':(0x35ddc0,0x35de65),'Factory':(0x35dc70,0x35dd04),'MoveNext':(0x374fc0,0x3750e4),'Reset':(0x3750f0,0x37512e)}
 sigs=['void Acted__Act (Acted_o* __this, System_String_o* description, float delay, const MethodInfo* method);','System_Collections_IEnumerator_o* Acted__ActDelayed (Acted_o* __this, System_String_o* description, float delay, const MethodInfo* method);','bool Acted__ActDelayed_d__9__MoveNext (Acted__ActDelayed_d__9_o* __this, const MethodInfo* method);','void Acted__ActDelayed_d__9__System_Collections_IEnumerator_Reset (Acted__ActDelayed_d__9_o* __this, const MethodInfo* method);']
 exact=[]
 for (a,_),sig in zip(entries.values(),sigs):
  rows=[r for r in meta['ScriptMethod'] if r['Address']==a];assert len(rows)==1 and rows[0]['Signature']==sig;exact.append(rows[0])
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True;decoded={}
 for a,b in [*entries.values(),(0x33ed50,0x33ed53)]:
  ins=list(cs.disasm(pe.get_data(a,b-a),a));assert ins[0].address==a and sum(i.size for i in ins)==b-a;decoded.update({i.address:i for i in ins})
 checks={0x35dddf:('call','0x1c7f4a0'),0x35de23:('mov','dword ptr [rbx + 0x10], 0'),0x35de41:('movss','dword ptr [rbx + 0x20], xmm6'),0x35de60:('jmp','0x1c7f160'),0x37500a:('mov','dword ptr [rdi + 0x10], 0xffffffff'),0x375035:('mov','qword ptr [rcx], rbx'),0x375044:('mov','dword ptr [rdi + 0x10], 1'),0x375060:('mov','dword ptr [rdi + 0x10], 0xffffffff'),0x37507c:('call','0x35d920'),0x375081:('mov','rdi, qword ptr [rbx + 0x28]'),0x3750bc:('call','0x1ec1010')}
 for a,v in checks.items():assert (decoded[a].mnemonic,decoded[a].op_str)==v
 immediate=[]
 for a,name in [(0x1c7f4a0,'UnityEngine.MonoBehaviour$$StopAllCoroutines'),(0x1c7f160,'UnityEngine.MonoBehaviour$$StartCoroutine'),(0x1c961f0,'UnityEngine.WaitForSeconds$$.ctor'),(0x35d920,'ActedVersion$$Show'),(0x1ec1010,'UnityEngine.UI.LayoutRebuilder$$ForceRebuildLayoutImmediate')]:
  rows=[r for r in meta['ScriptMethod'] if r['Address']==a and r['Name']==name];assert len(rows)==1;immediate.append(rows[0])
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
 arena,stack,stop=0x200000000,0x300000000,0x400000000
 uc.mem_map(arena,0x20000);uc.mem_map(stack,0x10000);uc.mem_map(stop,0x1000)
 def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
 def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
 def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
 def rd(a):return struct.unpack('<I',uc.mem_read(a,4))[0]
 def reg(r):return uc.reg_read(r)
 def ret(v=0):
  sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
 refs=set();flags=set()
 for i in decoded.values():
  for o in i.operands:
   if o.type==capstone.CS_OP_MEM and o.mem.base==capstone.x86.X86_REG_RIP:refs.add(i.address+i.size+o.mem.disp)
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:flags.add(i.address+i.size+i.operands[0].mem.disp)
 bindings={};slotvalues={}
 for r in meta['ScriptMetadata']+meta['ScriptMetadataMethod']:
  if r['Address'] in refs:
   p=arena+0x1000+len(bindings)*0x200;bindings[r['Name']]=p;slotvalues[base+r['Address']]=p;q(base+r['Address'],p)
 for n in ['Acted.<ActDelayed>d__9_TypeInfo','UnityEngine.WaitForSeconds_TypeInfo','UnityEngine.UI.LayoutRebuilder_TypeInfo','System.NotSupportedException_TypeInfo','Method$Acted.<ActDelayed>d__9.System.Collections.IEnumerator.Reset()']:assert n in bindings,n
 actor,version,iterator,wait,exception,description,old_current,array,other_array=[arena+i for i in range(0x8000,0x11000,0x1000)]
 layouts=[arena+0x18000,arena+0x18100];labels={0:None,actor:'actor',version:'version',iterator:'iterator',wait:'wait',exception:'exception',description:'description',old_current:'old_current',array:'array',other_array:'other_array',layouts[0]:'layout0',layouts[1]:'layout1'}
 state={};opt={};visited=set()
 def snapshot():return {'state':rd(iterator+0x10),'current':labels[rq(iterator+0x18)],'delay_bits':rd(iterator+0x20),'actor':labels[rq(iterator+0x28)],'description':labels[rq(iterator+0x30)],'wait_bits':rd(wait+0x10),'layouts':labels[rq(actor+0x28)]}
 def emit(kind,**kw):
  state['counts'][kind]=state['counts'].get(kind,0)+1;state['events'].append({'kind':kind,**kw,'snapshot':snapshot()})
  if opt.get('fail')==[kind,state['counts'][kind]]:state['error']=kind;uc.emu_stop();return False
  return True
 def hook(_,a,size,__):
  r=a-base;c,t=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX)
  if a==stop:uc.emu_stop();return
  if r==0x2b7b40:
   assert c in slotvalues
   if emit('metadata'):ret(slotvalues[c])
  elif r==0x2b7d40:
   kind={bindings['Acted.<ActDelayed>d__9_TypeInfo']:'iterator',bindings['UnityEngine.WaitForSeconds_TypeInfo']:'wait',bindings['System.NotSupportedException_TypeInfo']:'exception'}[c]
   if emit('allocate',object=kind):
    target={'iterator':iterator,'wait':wait,'exception':exception}[kind];uc.mem_write(target,bytes(0x80));q(target,c);ret(target)
  elif r==0x1c7f4a0:
   assert c==(0 if opt.get('null_actor') else actor) and t==0
   if emit('stop',actor=labels[c]):ret()
  elif r==0x2b6ff0:
   assert rq(c)==t and c in [iterator+0x18,iterator+0x28,iterator+0x30]
   if emit('barrier',offset=c-iterator):ret()
  elif r==0x1c7f160:
   assert c==(0 if opt.get('null_actor') else actor) and t==iterator and reg(x.UC_X86_REG_R8)==0
   if emit('start',actor=labels[c]):ret(old_current)
  elif r==0x1c961f0:
   assert c==wait and reg(x.UC_X86_REG_R8)==0;bits=reg(x.UC_X86_REG_XMM1)&0xffffffff
   if emit('wait_ctor',delay_bits=bits):d(wait+0x10,bits);ret()
  elif r==0x35d920:
   assert c==version and t==(0 if opt.get('null_description') else description)
   if emit('show',description=labels[t]):
    state['effects'].append('show')
    if opt.get('replace_on_show'):q(actor+0x28,other_array)
    ret()
  elif r==0x281d90:
   assert c==bindings['UnityEngine.UI.LayoutRebuilder_TypeInfo']
   if emit('class_init'):d(c+0xe0,1);ret()
  elif r==0x1ec1010:
   assert c in [0,*layouts]
   if emit('rebuild',target=labels[c]):
    state['effects'].append(labels[c])
    if opt.get('replace_on_rebuild'):q(actor+0x28,other_array)
    ret()
  elif r==0x111b720:
   assert c==exception
   if emit('exception_ctor'):ret()
  elif r==0x2b7d50:
   assert c==exception and t==bindings['Method$Acted.<ActDelayed>d__9.System.Collections.IEnumerator.Reset()']
   emit('throw');state['error']=state['error'] or 'not_supported';uc.emu_stop()
  elif r in (0x2b7d90,0x2b7d80):state['error']='null' if r==0x2b7d90 else 'bounds';uc.emu_stop()
  else:assert r in decoded and decoded[r].size==size,hex(r);visited.add(r)
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 def run(method,options=None,sequence=False):
  opt.clear();opt.update(options or {});state.clear();state.update(counts={},events=[],effects=[],error=None)
  for p in [actor,version,iterator,wait,exception]:uc.mem_write(p,bytes(0x80))
  q(actor+0x20,0 if opt.get('null_version') else version);q(actor+0x28,0 if opt.get('null_layouts') else array)
  vals=opt.get('layouts',[0,1]);q(array+0x18,len(vals));q(other_array+0x18,1);q(other_array+0x20,layouts[1])
  for i,v in enumerate(vals):q(array+0x20+i*8,0 if v is None else layouts[v])
  d(iterator+0x10,opt.get('state',0));q(iterator+0x18,old_current);d(iterator+0x20,opt.get('delay_bits',0x3f800000));q(iterator+0x28,0 if opt.get('null_actor') else actor);q(iterator+0x30,0 if opt.get('null_description') else description)
  for flag in flags:uc.mem_write(base+flag,bytes([0 if opt.get('cold') else 1]))
  d(bindings['UnityEngine.UI.LayoutRebuilder_TypeInfo']+0xe0,0 if opt.get('cold') else 1)
  initial=snapshot();returns=[];snapshots=[]
  for operation in ([method,'MoveNext','MoveNext','MoveNext'] if sequence else [method]):
   sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,iterator if operation in ['MoveNext','Reset'] else 0 if opt.get('null_actor') else actor);uc.reg_write(x.UC_X86_REG_RDX,0 if opt.get('null_description') else description);uc.reg_write(x.UC_X86_REG_XMM2,opt.get('delay_bits',0x3f800000))
   keep=[x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
   for rr in keep:uc.reg_write(rr,0xabc000+rr)
   uc.reg_write(x.UC_X86_REG_XMM6,0x123456789abcdef123456789abcdef)
   uc.emu_start(base+entries[operation][0],stop,count=10000)
   if state['error'] is None:
    assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8 and all(reg(rr)==0xabc000+rr for rr in keep)
    assert reg(x.UC_X86_REG_XMM6)==0x123456789abcdef123456789abcdef
    returns.append(reg(x.UC_X86_REG_RAX)&255 if operation=='MoveNext' else labels.get(reg(x.UC_X86_REG_RAX)) if operation=='Factory' else None)
   snapshots.append(snapshot())
   if state['error']:break
  return {'method':method,'input':dict(opt),'sequence':sequence,'initial':initial,'final':snapshot(),'steps':snapshots,'returns':returns,'events':state['events'][:],'effects':state['effects'][:],'error':state['error']}
 cases=[]
 for method,bits,null_actor,null_description in itertools.product(['ActDelay','Factory'],[0,0x80000000,1,0xbf800000,0x3f800000,0x7f800000,0xff800000,0x7fc12345,0x7f812345],[False,True],[False,True]):
  r=run(method,{'delay_bits':bits,'null_actor':null_actor,'null_description':null_description});assert r['error'] is None
  assert r['final']['state']==0 and r['final']['current'] is None and r['final']['delay_bits']==bits
  assert [e['kind'] for e in r['events']]==(['stop'] if method=='ActDelay' else [])+['allocate','barrier','barrier']+(['start'] if method=='ActDelay' else [])
  cases.append(r)
 for bits in [0,0x80000000,1,0xbf800000,0x7f800000,0x7fc12345,0x7f812345]:
  r=run('Factory',{'delay_bits':bits},True);assert r['error'] is None and r['returns']==['iterator',1,0,0]
  assert r['steps'][1]['wait_bits']==bits and r['steps'][1]['state']==1 and r['final']['current']=='wait' and r['final']['state']==0xffffffff
  assert r['effects']==['show','layout0','layout1'];cases.append(r)
 for method,options in [('ActDelay',{'cold':True}),('Factory',{'cold':True}),('MoveNext',{'cold':True}),('MoveNext',{'cold':True,'state':1})]:
  baseline=run(method,options);assert baseline['error'] is None;cases.append(baseline);counts={}
  for index,event in enumerate(baseline['events']):
   kind=event['kind'];counts[kind]=counts.get(kind,0)+1;r=run(method,dict(options,fail=[kind,counts[kind]]))
   assert r['error']==kind and r['events']==baseline['events'][:index+1] and r['final']==event['snapshot']
   expected_effects=['show' if e['kind']=='show' else e['target'] for e in baseline['events'][:index] if e['kind'] in ['show','rebuild']]
   assert r['effects']==expected_effects;cases.append(r)
 for options in [{'null_actor':True},{'null_description':True}]:
  r=run('Factory',options,True)
  assert r['returns'][:2]==['iterator',1] and r['steps'][1]['state']==1
  assert r['error']==('null' if options.get('null_actor') else None)
  if options.get('null_actor'):assert not r['effects'] and r['final']['current']=='wait' and r['final']['state']==0xffffffff
  cases.append(r)
 for status in [-2**31,-1,2,2**31-1]:
  r=run('MoveNext',{'state':status});assert r['returns']==[0] and r['initial']==r['final'] and not r['events'];cases.append(r)
 for options in [{'null_actor':True},{'null_version':True},{'null_layouts':True},{'layouts':[]},{'layouts':[0,None,0]},{'replace_on_show':True},{'replace_on_rebuild':True},{'null_description':True}]:
  r=run('MoveNext',dict(options,state=1));assert r['final']['state']==0xffffffff
  assert r['error']==('null' if any(options.get(k) for k in ['null_actor','null_version','null_layouts']) else None)
  if options.get('null_layouts'):assert r['effects']==['show']
  if options.get('null_actor') or options.get('null_version'):assert not r['effects']
  if options.get('replace_on_show'):assert r['effects']==['show','layout1']
  if options.get('replace_on_rebuild'):assert r['effects']==['show','layout0','layout1']
  if options.get('layouts')==[0,None,0]:assert r['effects']==['show','layout0',None,'layout0']
  cases.append(r)
 for fail in [None,['metadata',1],['allocate',1],['exception_ctor',1],['metadata',2],['throw',1]]:
  r=run('Reset',{'fail':fail});assert r['initial']==r['final'] and r['error']==(fail[0] if fail else 'not_supported');cases.append(r)
 table=[];indices={}
 for case in cases:
  for event in case['events']:
   snap=event.pop('snapshot');key=json.dumps(snap,sort_keys=True)
   if key not in indices:indices[key]=len(table);table.append(snap)
   event['snapshot_index']=indices[key]
 return {'build_id':BUILD,'metadata_verified':exact,'immediate_methods':immediate,'cases_passed':len(cases),'native_assertions':len(checks),'native_instructions_executed':len(visited),'snapshot_table':table,'cases':cases,'scope':'Actual delayed wrapper/factory/MoveNext/Reset; raw float bits and partial writes. Stop/StartCoroutine, WaitForSeconds constructor, Show, LayoutRebuilder, allocation/metadata/class-init/GC and exception construction/throwing are explicit services. No scheduler timing or engine/UI effects inferred.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',required=True,type=Path)
 a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(r['cases_passed'],r['native_instructions_executed'])
