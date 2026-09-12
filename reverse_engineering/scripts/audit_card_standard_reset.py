"""Native card refresh composed with StandardMode load, display, abandon and UI reentry."""
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
 for manifest_name in ('game_mode_lifecycle','standard_mode_progression'):
  for f in json.loads((repo/f'targets/{manifest_name}.json').read_text(encoding='utf-8'))['functions']:
   assert any(r['Name']==f['metadata_name'] and r['Address']==int(f['rva'],16) and r['Signature']==f['signature'] for r in meta['ScriptMethod'])
 for name,a in [('GameMode$$OnLoadGame',0x33ED50),('StandardMode$$IsLocked',0x3BCC90)]:
  r=next(r for r in meta['ScriptMethod'] if r['Name']==name);assert r['Address']==a
 game_decl=dump.split('public abstract class GameMode ',1)[1].split('// Namespace:',1)[0]
 for slot,name in [(6,'LoadGame'),(7,'OnLoadGame'),(15,'AbandonRun'),(21,'IsLocked'),(22,'GetScores')]:
  assert re.search(rf'Slot: {slot}\n\s*public (?:abstract|virtual) [^\n]+ {name}\(',game_decl)
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
 ends={0x3EE3A0:0x3EE3A7,0x387B00:0x387BA0,0x33ED50:0x33ED53,0x3EDE90:0x3EDF8E,0x3ED800:0x3ED880,0x3BCC90:0x3BCC93,0x387AC0:0x387AF5,0x387F60:0x387FA6,0x397440:0x39748D,0x3A11D0:0x3A12F3,0x3A10B0:0x3A11C9,0x3A1300:0x3A14E6,0x3A1060:0x3A10AC,0x33E820:0x33E827};decoded={}
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
 types={n:arena+0x1000+i*0x600 for i,n in enumerate(('GameData_TypeInfo','UIEvents_TypeInfo','System.Action_TypeInfo','StandardMode_TypeInfo','int_TypeInfo'))};found=set()
 for r in meta['ScriptMetadata']:
  if r['Name'] in types:q(base+r['Address'],types[r['Name']]);found.add(r['Name'])
 assert found==set(types)
 token=arena+0x4000;r=next(r for r in meta['ScriptMetadataMethod'] if r['Name']=='Method$GameModeCard.UpdateView()');q(base+r['Address'],token)
 card,original,loaded,text_obj,locked,button,trigger,ui= [arena+o for o in range(0x5000,0xD000,0x1000)]
 labels={0:'null',card:'card',original:'original',loaded:'loaded',text_obj:'text',locked:'locked',button:'button',trigger:'trigger'}
 q(types['UIEvents_TypeInfo']+0xB8,ui)
 klass=arena+0xD000;text_class=arena+0xE000
 services={stop+0x100+i*0x10:n for i,n in enumerate(('load','on_load','scores','abandon','is_locked','text','deinit'))};svc={n:p for p,n in services.items()}
 for off,n in [(0x198,'load'),(0x1A8,'on_load'),(0x298,'scores'),(0x228,'abandon'),(0x288,'is_locked')]:q(klass+off,svc[n]);q(klass+off+8,token)
 q(original,klass);q(loaded,klass);q(text_obj,text_class);q(text_class+0x558,svc['text']);q(text_class+0x560,token)
 q(klass+0xC8,arena+0xF000);q(arena+0xF000,types['StandardMode_TypeInfo']);uc.mem_write(types['StandardMode_TypeInfo']+0x130,b'\1')
 for i in decoded.values():
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
 # Concrete virtual dispatch targets; base OnLoadGame is the exact ret 0 stub.
 for off,rva in ((0x198,0x3EE3A0),(0x1A8,0x33ED50),(0x298,0x3EDE90),(0x228,0x3ED800),(0x288,0x3BCC90),(0x268,0x387AC0)):
  q(klass+off,base+rva);q(klass+off+8,token)
 q(klass+0x1B8,svc['deinit']);q(klass+0x1C0,token)
 strings={}
 for row in meta['ScriptString']:
  if any(i.mnemonic=='mov' and len(i.operands)>1 and i.operands[1].type==capstone.CS_OP_MEM and i.operands[1].mem.base==capstone.x86.X86_REG_RIP and i.address+i.size+i.operands[1].mem.disp==row['Address'] for i in decoded.values()):
   ptr=arena+0x40000+len(strings)*0x100;strings[ptr]=row['Value'];q(base+row['Address'],ptr)
 json_token=arena+0x17000
 row=next(r for r in meta['ScriptMetadataMethod'] if r['Name']=='Method$UnityEngine.JsonUtility.FromJson<StandardMode>()');q(base+row['Address'],json_token)
 # One native Action invocation can directly target UpdateView with the card target.
 notification=arena+0x16000;q(notification+0x18,base+0x3A1300);q(notification+0x40,card);q(notification+0x28,token)
 opt={};state={};visited=set()
 fields={'score':0x10,'currentLevel':0x14,'savedVillages':0x18,'currentDiedTimes':0x1C,'bestDiedTimes':0x20,'completed':0x24,'roundScore':0x28,'currentScore':0x2C,'bestScore':0x30,'currentCompleted':0x34,'failScoreDecrease':0x38}
 defaults=dict(zip(fields,(100,6,8,3,7,1,60,400,800,1,20)))
 def snapshot():return {n:uc.mem_read(loaded+o,1)[0] if n in ('completed','currentCompleted') else struct.unpack('<i',uc.mem_read(loaded+o,4))[0] for n,o in fields.items()}
 def halt(n):state['error']=n;uc.emu_stop()
 def event(n,**kw):
  state['trace'].append({'event':n,'mode':labels[rq(card+0x20)],'state':snapshot(),**kw})
  if opt.get('fail')==n:halt(n);return False
  return True
 def hook(_,a,size,__):
  r=a-base;c,t,m=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
  if a==stop:uc.emu_stop();return
  if r==0x3A1300:
   state['refreshes']+=1;event('refresh',ordinal=state['refreshes'])
   if state['refreshes']>opt.get('max_refreshes',3):halt('reentry_bound');return
  if a==svc['text']:
   assert c==text_obj and m==token
   if event('text',value=state['formatted'][t]):ret()
  elif a==svc['deinit']:
   assert c==loaded and t==token
   if event('deinit'):ret()
  elif r==0x2B6FF0:assert c==card+0x20;event('mode_store');ret()
  elif r==0x1C85F20:
   assert strings[c]=='SavedStandard';state['reads']+=1
   if event('get_string'):ret(arena+0x18000)
  elif r==0xF76390:ret(0)
  elif r==0x645DA0:
   assert c==arena+0x18000 and t==json_token
   if event('from_json'):
    # Controlled stale serializer reload can restore completed on each recursion.
    if opt.get('stale_completed'):uc.mem_write(loaded+0x34,b'\1')
    ret(0 if opt.get('null_json') else loaded)
  elif r in (0x1C85EE0,0x1C85EA0):
   assert strings[c]=='SavedStandardMode'
   if event('get_int'):ret(6)
  elif r==0x282580:
   assert c==types['int_TypeInfo'];value=struct.unpack('<i',uc.mem_read(t,4))[0];p=arena+0x20000+len(state['boxes'])*0x100;state['boxes'][p]=value;ret(p)
  elif r==0xF74DF0:
   assert c in strings and t in state['boxes'];p=arena+0x28000+len(state['formatted'])*0x100;state['formatted'][p]=(strings[c],state['boxes'][t]);ret(p)
  elif r==0xF71940:
   values=[state['formatted'][p] for p in (c,t,m)];p=arena+0x28000+len(state['formatted'])*0x100;state['formatted'][p]=values
   if event('scores',values=values):ret(p)
  elif r==0x1CD6420:
   assert c==loaded and t==0
   if event('to_json'):ret(arena+0x19000)
  elif r==0x1C86170:
   assert strings[c]=='SavedStandard' and t==arena+0x19000
   if event('set_string'):ret()
  elif r in (0x1C7D810,0x1C79840):
   n='active' if r==0x1C7D810 else 'button_enabled' if c==button else 'trigger_enabled'
   if event(n,value=t&0xff):ret()
  elif r in (0x2B7D90,0x2B7040):halt('null' if r==0x2B7D90 else 'cast')
  elif r in decoded:
   visited.add(r)
   if r==0x3ED800:event('abandon')
  else:raise AssertionError(hex(r))
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 cases=[]
 def run(options):
  opt.clear();opt.update(options);state.clear();state.update(trace=[],error=None,refreshes=0,reads=0,boxes={},formatted={})
  q(card+0x20,original)
  for off,p in ((0x28,text_obj),(0x30,locked),(0x38,button),(0x40,trigger)):q(card+off,p)
  uc.mem_write(klass+0x130,b'\1');q(ui,notification if opt.get('notify') else 0)
  for n,o in fields.items():
   value=opt.get('completed',1) if n=='currentCompleted' else defaults[n]
   if n in ('completed','currentCompleted'):uc.mem_write(loaded+o,bytes([value]))
   else:d(loaded+o,value)
  before=snapshot();sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,card);uc.reg_write(x.UC_X86_REG_RDX,token)
  keep=[x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
  for rr in keep:uc.reg_write(rr,0xABC000+rr)
  uc.emu_start(base+0x3A1300,stop,count=20000)
  if state['error'] is None:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8 and all(reg(rr)==0xABC000+rr for rr in keep)
  result={'options':options,'initial':before,'final':snapshot(),'refreshes':state['refreshes'],'trace':state['trace'][:],'error':state['error']};cases.append(result);return result
 for complete,notify in itertools.product((0,1,7),(False,True)):
  result=run({'completed':complete,'notify':notify});assert result['error'] is None
  expected=defaults|({'score':0,'currentLevel':0,'currentDiedTimes':0,'roundScore':0,'currentScore':0,'currentCompleted':0} if complete else {'currentCompleted':0})
  assert result['final']==expected
  assert result['refreshes']==(2 if complete and notify else 1)
  kinds=[v['event'] for v in result['trace']]
  assert kinds.count('abandon')==int(bool(complete)) and kinds.count('deinit')==int(bool(complete))
  if complete:
   assert kinds.index('text')<kinds.index('abandon')<kinds.index('deinit')<kinds.index('to_json')<kinds.index('set_string')
   if notify:assert kinds.index('set_string')<kinds.index('refresh',1)
  assert kinds.count('active')==result['refreshes'] and all(v['value']==0 for v in result['trace'] if v['event']=='active')
 for fail in ('get_string','from_json','get_int','scores','text','deinit','to_json','set_string','active','button_enabled','trigger_enabled'):
  result=run({'fail':fail,'notify':True});assert result['error']==fail
  if fail in ('get_string','from_json','get_int','scores','text','deinit'):assert result['final']==defaults
  else:assert result['final']['currentCompleted']==0 and result['final']['currentScore']==0
 result=run({'null_json':True});assert result['error']=='null' and [v['event'] for v in result['trace']][-1]=='mode_store'
 result=run({'notify':True,'stale_completed':True,'max_refreshes':3});assert result['error']=='reentry_bound' and result['refreshes']==4 and sum(v['event']=='abandon' for v in result['trace'])==3
 return {'schema_version':1,'build_id':BUILD,'cases_passed':len(cases),'distinct_native_instructions':len(visited),'cases':cases,'scope':'Actual Card.UpdateView, Standard.LoadGame/saved getter, inherited OnLoadGame, GetScores/saved-max getter, AbandonRun/save setter, IsLocked and direct OnUIUpdate delegate call reentry execute natively. JSON/PlayerPrefs, format/boxing, UI setters and virtual DeInit are explicit gateways. Controlled current/stale JSON state is not actual persistence fidelity; recursion is bounded diagnostically.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(f"Passed {r['cases_passed']} native card-reset cases")
