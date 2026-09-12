"""Pinned mode-selection UI callers; UI and mode services are explicit gateways."""
import argparse,hashlib,itertools,json,re,struct
from pathlib import Path
from audit_character_assets import BUILD

def audit(game_root,dumper_root):
 import capstone,pefile,unicorn
 from unicorn import x86_const as x
 assert unicorn.__version__=='2.1.4'
 repo=Path(__file__).parents[1]
 lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'));ext=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
 def pin(p,h):
  b=p.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==h.upper();return b
 raw=pin(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256'])
 meta=json.loads(pin(Path(dumper_root)/'script.json',ext['outputs']['script_json']['sha256']).decode('utf-8-sig'))
 dump=pin(Path(dumper_root)/'dump.cs',ext['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
 methods={'ChangeGameModeButton$$OnClick':0x397440,'ChangeGameModeButton$$.ctor':0x33E820,'GameModeCard$$OnEnable':0x3A11D0,'GameModeCard$$OnDisable':0x3A10B0,'GameModeCard$$UpdateView':0x3A1300,'GameModeCard$$Lock':0x3A1060,'GameModeCard$$.ctor':0x33E820}
 declarations=[]
 for n,a in methods.items():
  r=next(r for r in meta['ScriptMethod'] if r['Name']==n);assert r['Address']==a
  assert r['Signature']==f"void {n.replace('$$','__').replace('.ctor','_ctor')} ({n.split('$$')[0]}_o* __this, const MethodInfo* method);";declarations.append(r)
 game_decl=dump.split('public abstract class GameMode ',1)[1].split('// Namespace:',1)[0]
 for slot,name in [(6,'LoadGame'),(7,'OnLoadGame'),(15,'AbandonRun'),(21,'IsLocked'),(22,'GetScores')]:
  assert re.search(rf'Slot: {slot}\n\s*public (?:abstract|virtual) [^\n]+ {name}\(',game_decl)
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
 ends={0x397440:0x39748D,0x3A11D0:0x3A12F3,0x3A10B0:0x3A11C9,0x3A1300:0x3A14E6,0x3A1060:0x3A10AC,0x33E820:0x33E827};decoded={}
 for a,b in ends.items():
  buf=pe.get_data(a,b-a);assert len(buf)==b-a;ins=list(cs.disasm(buf,a));assert ins[-1].address+ins[-1].size==b and ins[-1].mnemonic!='int3';decoded.update({i.address:i for i in ins})
 checks=[(0x397488,'jmp','0x3dbe10'),(0x3A12D7,'jmp','0x3a1300'),(0x3A142D,'call','qword ptr [rax + 0x228]'),(0x3A1375,'call','qword ptr [rax + 0x1a8]'),(0x3A13B8,'call','qword ptr [r9 + 0x558]'),(0x33E822,'jmp','0x1c79770')]
 for a,m,o in checks:assert (decoded[a].mnemonic,decoded[a].op_str)==(m,o)
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
 arena,stack,stop=0x200000000,0x300000000,0x400000000;uc.mem_map(arena,0x100000);uc.mem_map(stack,0x10000);uc.mem_map(stop,0x1000)
 def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
 def d(a,v):uc.mem_write(a,struct.pack('<I',v))
 def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
 def reg(r):return uc.reg_read(r)
 def ret(v=0):
  sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
 types={n:arena+0x1000+i*0x800 for i,n in enumerate(('GameData_TypeInfo','UIEvents_TypeInfo','System.Action_TypeInfo','StandardMode_TypeInfo'))};found=set()
 for r in meta['ScriptMetadata']:
  if r['Name'] in types:q(base+r['Address'],types[r['Name']]);found.add(r['Name'])
 assert found==set(types)
 token=arena+0x4000;r=next(r for r in meta['ScriptMetadataMethod'] if r['Name']=='Method$GameModeCard.UpdateView()');q(base+r['Address'],token)
 card,original,loaded,text_obj,locked,button,trigger,ui= [arena+o for o in range(0x5000,0xD000,0x1000)]
 labels={0:'null',card:'card',original:'original',loaded:'loaded',text_obj:'text',locked:'locked',button:'button',trigger:'trigger'}
 q(types['UIEvents_TypeInfo']+0xB8,ui)
 klass=arena+0xD000;text_class=arena+0xE000
 services={stop+0x100+i*0x10:n for i,n in enumerate(('load','on_load','scores','abandon','is_locked','text'))};svc={n:p for p,n in services.items()}
 for off,n in [(0x198,'load'),(0x1A8,'on_load'),(0x298,'scores'),(0x228,'abandon'),(0x288,'is_locked')]:q(klass+off,svc[n]);q(klass+off+8,token)
 q(original,klass);q(loaded,klass);q(text_obj,text_class);q(text_class+0x558,svc['text']);q(text_class+0x560,token)
 q(klass+0xC8,arena+0xF000);q(arena+0xF000,types['StandardMode_TypeInfo']);uc.mem_write(types['StandardMode_TypeInfo']+0x130,b'\1')
 for i in decoded.values():
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
 opt={};state={};lists={};visited=set()
 def alloc(typ):
  state['alloc']+=1;p=arena+0x20000+state['alloc']*0x100;q(p,typ);return p
 def halt(n):state['error']=n;uc.emu_stop()
 def event(n,**kw):
  state['trace'].append({'event':n,'mode':labels[rq(card+0x20)],**kw})
  if opt.get('fail')==n:halt(n);return False
  return True
 def hook(_,a,size,__):
  r=a-base;c,t,m=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
  if a==stop:uc.emu_stop();return
  if a in services:
   n=services[a];assert c==(text_obj if n=='text' else (original if n=='load' else opt['result']))
   assert (m if n=='text' else t)==token
   if n=='text':assert t==arena+0x18000
   if event(n,target=labels[c]):
    ret(opt['result'] if n=='load' else arena+0x18000 if n=='scores' else int(opt.get('locked',False)) if n=='is_locked' else 0)
  elif r==0x281D90:
   assert c==types['GameData_TypeInfo']
   if event('class_init'):d(c+0xE0,1);ret()
  elif r==0x3DBE10:
   assert c==opt['input'] and t==0
   if event('change',target=labels[c]):ret()
  elif r==0x2B7D40:ret(alloc(c))
  elif r==0x4D5170:assert t==card and m==token;lists[c]=[('card',token)];ret()
  elif r in (0x116BCC0,0x116E070):
   n='combine' if r==0x116BCC0 else 'remove'
   if not event(n):return
   values=lists.get(c,[])[:]
   if n=='combine':values+=lists[t]
   elif lists[t][0] in values:values.reverse();values.remove(lists[t][0]);values.reverse()
   p=alloc(types['System.Action_TypeInfo']) if values else 0
   if p:lists[p]=values
   if opt.get('bad_cast'):p=alloc(0)
   ret(p)
  elif r==0x2B6FF0:
   assert c in (ui,card+0x20);event('subscription_store' if c==ui else 'mode_store');ret()
  elif r in (0x1C7D810,0x1C79840):
   n='active' if r==0x1C7D810 else 'button_enabled' if c==button else 'trigger_enabled'
   assert c in (locked,button,trigger) and m==0
   if event(n,value=t&0xff):ret()
  elif r==0x1C79770:
   assert c==card and t==0
   if event('behaviour_ctor'):ret()
  elif r in (0x2B7D90,0x2B7040):halt('null' if r==0x2B7D90 else 'cast')
  elif r in decoded:visited.add(r)
  else:raise AssertionError(hex(r))
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 cases=[]
 def run(name,options=None):
  opt.clear();opt.update(input=original,result=loaded);opt.update(options or {});state.clear();state.update(alloc=0,trace=[],error=None);lists.clear()
  q(card+0x20,opt['input'])
  for off,p,n in ((0x28,text_obj,'text'),(0x30,locked,'locked'),(0x38,button,'button'),(0x40,trigger,'trigger')):q(card+off,0 if opt.get('missing')==n else p)
  uc.mem_write(klass+0x130,b'\1' if opt.get('standard') else b'\0');uc.mem_write(loaded+0x34,bytes([opt.get('completed',0)]));uc.mem_write(original+0x34,bytes([opt.get('completed',0)]))
  d(types['GameData_TypeInfo']+0xE0,0 if opt.get('cold') else 1)
  p=alloc(types['System.Action_TypeInfo']);lists[p]=([] if opt.get('no_other') else [('other',token)])+[('card',token)]*opt.get('existing',0);q(ui,p)
  sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,card);uc.reg_write(x.UC_X86_REG_RDX,0)
  keep=[x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
  for r in keep:uc.reg_write(r,0xABC000+r)
  uc.emu_start(base+methods[name],stop,count=5000)
  if state['error'] is None:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8 and all(reg(r)==0xABC000+r for r in keep)
  result={'method':name,'options':{k:labels.get(v,v) if k in ('input','result') else v for k,v in opt.items()},'trace':state['trace'][:],'error':state['error'],'final_mode':labels[rq(card+0x20)],'subscribers':len(lists.get(rq(ui),[]))};cases.append(result);return result
 for standard,completed,islocked,identity in itertools.product((False,True),(0,1,7),(False,True),('loaded','original')):
  r=run('GameModeCard$$UpdateView',{'standard':standard,'completed':completed,'locked':islocked,'result':loaded if identity=='loaded' else original})
  expected=['load','mode_store','on_load','scores','text']+(['abandon'] if standard and completed else [])+['is_locked','active','button_enabled','trigger_enabled']
  assert [v['event'] for v in r['trace']]==expected and r['error'] is None and r['final_mode']==identity
  assert [v['value'] for v in r['trace'] if 'value'in v]==[int(islocked),int(not islocked),int(not islocked)]
 for input_value,result in ((0,loaded),(original,0)):
  r=run('GameModeCard$$UpdateView',{'input':input_value,'result':result})
  assert r['error']==('null' if result==0 else None)
  assert [v['event'] for v in r['trace']]==(['load','mode_store'] if result==0 else ['active','button_enabled','trigger_enabled'])
 for missing in ('text','locked','button','trigger'):
  r=run('GameModeCard$$UpdateView',{'missing':missing});assert r['error']=='null'
  expected={'text':['load','mode_store','on_load','scores'],'locked':['load','mode_store','on_load','scores','text','is_locked'],'button':['load','mode_store','on_load','scores','text','is_locked','active'],'trigger':['load','mode_store','on_load','scores','text','is_locked','active','button_enabled']}
  assert [v['event'] for v in r['trace']]==expected[missing]
 success=['load','mode_store','on_load','scores','text','abandon','is_locked','active','button_enabled','trigger_enabled']
 for fail in ('load','on_load','scores','text','abandon','is_locked','active','button_enabled','trigger_enabled'):
  r=run('GameModeCard$$UpdateView',{'standard':True,'completed':1,'fail':fail});assert r['error']==fail and [v['event'] for v in r['trace']]==success[:success.index(fail)+1]
  assert r['final_mode']==('original' if fail=='load' else 'loaded')
 for name in ('GameModeCard$$OnEnable','GameModeCard$$OnDisable'):
  for existing in (0,1,2):
   r=run(name,{'existing':existing});assert r['error'] is None and r['subscribers']==1+(existing+1 if name.endswith('OnEnable') else max(0,existing-1))
   assert [v['event'] for v in r['trace']][:2]==['combine' if name.endswith('OnEnable') else 'remove','subscription_store']
   if name.endswith('OnDisable'):assert len(r['trace'])==2
  for fail in ('combine' if name.endswith('OnEnable') else 'remove',):
   r=run(name,{'fail':fail});assert r['error']==fail and r['subscribers']==1
  r=run(name,{'bad_cast':True});assert r['error']=='cast' and r['subscribers']==1
 for existing in (0,1):
  r=run('GameModeCard$$OnDisable',{'no_other':True,'existing':existing});assert r['error'] is None and r['subscribers']==0 and rq(ui)==0
 r=run('GameModeCard$$OnEnable',{'fail':'load'});assert r['error']=='load' and r['subscribers']==2
 for missing in (None,'locked','button','trigger'):
  r=run('GameModeCard$$Lock',{'missing':missing});count={None:3,'locked':0,'button':1,'trigger':2}[missing];assert len(r['trace'])==count and r['error']==(None if missing is None else 'null')
 for inp,cold,fail in itertools.product((0,original),(False,True),(None,'class_init','change')):
  r=run('ChangeGameModeButton$$OnClick',{'input':inp,'cold':cold,'fail':fail});expect=['class_init'] if cold and fail=='class_init' else (['class_init'] if cold else [])+['change'];assert [v['event'] for v in r['trace']]==expect
  assert r['error']==(fail if fail in expect else None)
 for cls in ('GameModeCard','ChangeGameModeButton'):
  r=run(cls+'$$.ctor');assert [v['event'] for v in r['trace']]==['behaviour_ctor']
 return {'schema_version':1,'build_id':BUILD,'exact_declarations':declarations,'cases_passed':len(cases),'native_relationships':len(checks),'distinct_native_instructions':len(visited),'cases':cases,'scope':'All seven declarations execute native callers. Mode virtual methods, UI setters, class init, delegate construction/list operations and base Behaviour ctor are explicit gateways. Gateways preserve card references except the native LoadGame-result store. No live UI, JSON or full notification history claimed.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(f"Passed {r['cases_passed']} mode-selection cases")
