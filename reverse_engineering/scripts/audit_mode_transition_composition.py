"""Native ChangeGameMode composed with both concrete lifecycle implementations."""
import argparse, hashlib, itertools, json, struct
from pathlib import Path
from audit_character_assets import BUILD

def audit(game_root,dumper_root):
 import capstone, pefile, unicorn
 from unicorn import x86_const as x
 assert unicorn.__version__=='2.1.4'
 repo=Path(__file__).parents[1]
 lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
 extraction=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
 def pin(p,h):
  b=p.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==h.upper();return b
 raw=pin(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256'])
 metadata=json.loads(pin(Path(dumper_root)/'script.json',extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
 verified=[]
 for manifest in ('game_data_lifecycle','game_mode_lifecycle','roguelike_standard_lifecycle'):
  for f in json.loads((repo/f'targets/{manifest}.json').read_text(encoding='utf-8'))['functions']:
   assert any(r['Name']==f['metadata_name'] and r['Address']==int(f['rva'],16) and r['Signature']==f['signature'] for r in metadata['ScriptMethod']);verified.append(f['metadata_name'])
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase
 cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
 entries=[0x3DBE10,0x3EE020,0x3ED930,0x3EE3A0,0x3E9E90,0x3E98C0,0x3EA1B0,0x387B00,0x387A20]
 decoded={}
 for a in entries:
  end=min(r['Address'] for r in metadata['ScriptMethod'] if r['Address']>a)
  ins=list(cs.disasm(pe.get_data(a,end-a),a))
  while ins and ins[-1].mnemonic=='int3':ins.pop()
  decoded.update({i.address:i for i in ins})
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
 arena,stack,stop=0x200000000,0x300000000,0x400000000
 uc.mem_map(arena,0x100000);uc.mem_map(stack,0x10000);uc.mem_map(stop,0x1000)
 def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
 def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
 def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
 def rd(a):return struct.unpack('<I',uc.mem_read(a,4))[0]
 def reg(r):return uc.reg_read(r)
 def ret(v=0):
  sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
 names=['GameData','GameEvents','GameplayEvents','StandardMode','RoguelikeStandard','System.Action','System.Action<Character>']
 types={n+'_TypeInfo':arena+0x1000+k*0x500 for k,n in enumerate(names)}
 found=set()
 for r in metadata['ScriptMetadata']:
  if r['Name'] in types:q(base+r['Address'],types[r['Name']]);found.add(r['Name'])
 assert found==set(types)
 tokens={}
 for r in metadata['ScriptMetadataMethod']:
  if r['Name'] in [f'Method${m}.{n}()' for m in ('StandardMode','RoguelikeStandard') for n in ('OnFailed','OnCharacterKilled')]+[f'Method$UnityEngine.JsonUtility.FromJson<{m}>()' for m in ('StandardMode','RoguelikeStandard')]:
   p=arena+0x4000+len(tokens)*0x100;tokens[r['Name']]=p;q(base+r['Address'],p)
 assert len(tokens)==6
 for r in metadata['ScriptString']:
  if r['Value'] in ('SavedStandard','SavedRoguelikeStandard'):q(base+r['Address'],arena+0x5000)
 game,events,gameplay=arena+0x6000,arena+0x7000,arena+0x8000
 for n,p in types.items():d(p+0xE0,1)
 for n,p in [('GameData',game),('GameEvents',events),('GameplayEvents',gameplay)]:q(types[n+'_TypeInfo']+0xB8,p)
 old,input_obj,loaded=arena+0x9000,arena+0x9100,arena+0x9200
 labels={0:'null',old:'old',input_obj:'input',loaded:'loaded'}
 bodies={'StandardMode':(0x3EE3A0,0x3ED930,0x3EE020),'RoguelikeStandard':(0x3EA1B0,0x3E98C0,0x3E9E90)}
 stage={m:arena+0xA000+i*0x100 for i,m in enumerate(bodies)}
 for m,(load,deinit,init) in bodies.items():
  c=types[m+'_TypeInfo']
  for off,rva in [(0x198,load),(0x1B8,deinit),(0x188,init)]:q(c+off,base+rva)
  q(c+0x240,stage[m])
 for i in decoded.values():
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
 slots={0x48:'kill',0x20:'won',0xB0:'died'};lists={};state={};opt={};visited=set()
 def alloc(typ):
  state['alloc']+=1;p=arena+0x10000+state['alloc']*0x100;q(p,typ);return p
 def identity(obj,m,event):return (labels[obj],stage[m] if event=='won' else tokens[f'Method${m}.On'+('CharacterKilled' if event=='kill' else 'Failed')+'()'])
 def snapshot():return {'global':labels[rq(game+0x10)],'village':rd(game+0x18),'events':{n:[list(v) for v in lists.get(rq(gameplay+o),[])] for o,n in slots.items()}}
 def emit(kind,**kw):state['trace'].append({'kind':kind,**kw,**snapshot()})
 def halt(reason):state['error']=reason;uc.emu_stop()
 def hook(_,a,size,__):
  r=a-base;c,t,m=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
  if a==stop:uc.emu_stop();return
  if a in (stop+0x100,stop+0x110):
   name='mode_changed' if a==stop+0x100 else 'game_init';emit(name)
   if opt.get('fail')==name:halt(name)
   else:ret()
  elif r==0x2B7D40:ret(alloc(c))
  elif r in (0x4D5170,0x4D5B60):lists[c]=[(labels[t],m)];ret()
  elif r in (0x116BCC0,0x116E070):
   state['ops']+=1
   if opt.get('fail')==state['ops']:halt('delegate');return
   values=lists.get(c,[])[:];rhs=lists[t]
   if r==0x116BCC0:values+=rhs
   else:
    for i in range(len(values)-len(rhs),-1,-1):
     if values[i:i+len(rhs)]==rhs:del values[i:i+len(rhs)];break
   p=alloc(rq(t)) if values else 0
   if p:lists[p]=values
   ret(p)
  elif r==0x2B7010:ret(c if rq(c)==t else 0)
  elif r==0x2B6FF0:
   emit('store',slot=slots.get(c-gameplay,'mode'));ret()
  elif r in (0x2B7040,0x2B7D90):halt('null_or_cast')
  elif r==0x1C85F20:state['reads']+=1;ret(arena+0x5100)
  elif r==0xF76390:ret(0)
  elif r==0x645DA0:
   emit('json')
   if opt.get('fail')=='json':halt('json')
   else:ret(opt['result'])
  elif r in decoded:
   visited.add(r)
   for mode,(load,deinit,init) in bodies.items():
    if r in (load,deinit,init):emit(('load','deinit','init')[(load,deinit,init).index(r)],mode=mode,target=labels[c])
  else:raise AssertionError(f'unhandled {r:x}')
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 results=[]
 for old_type,new_type,relation,fail in itertools.product(bodies,bodies,('distinct','same','old_null','loaded_null'),(None,'json',1,2,3,4,5,6,'mode_changed','game_init')):
  state.clear();state.update(alloc=0,ops=0,reads=0,trace=[],error=None);lists.clear();opt.clear();opt.update(result=old if relation=='same' else 0 if relation=='loaded_null' else loaded,fail=fail)
  # A same-object result retains its original runtime class even if loaded by the other mode's JSON gateway.
  q(old,types[old_type+'_TypeInfo']);q(input_obj,types[new_type+'_TypeInfo']);q(loaded,types[new_type+'_TypeInfo'])
  q(game+0x10,0 if relation=='old_null' else old);d(game+0x18,99)
  for obj in (old,input_obj,loaded):d(obj+0x14,7);d(obj+0x28,123)
  for off,event in slots.items():
   p=alloc(types[('System.Action<Character>' if event=='kill' else 'System.Action')+'_TypeInfo']);lists[p]=[identity(old,old_type,event)] if relation!='old_null' else [];q(gameplay+off,p)
  for j,off in enumerate((0x10,0x18)):
   p=arena+0xD000+j*0x100;q(p+0x18,stop+0x100+j*0x10);q(p+0x40,0);q(events+off,p)
  sp=stack+0x8000;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,input_obj);uc.reg_write(x.UC_X86_REG_RDX,0)
  uc.emu_start(base+0x3DBE10,stop,count=10000)
  tr=state['trace'];kinds=[v['kind'] for v in tr];lifecycle=[v for v in tr if v['kind'] in ('load','deinit','init')]
  assert lifecycle[0]['target']=='input'
  if fail=='json':assert len(lifecycle)==1 and state['error']=='json'
  elif relation=='old_null':assert len(lifecycle)==1 and rd(game+0x18)==99 and rd(loaded+0x28)==123
  else:
   assert lifecycle[1]['kind']=='deinit' and lifecycle[1]['target']=='old'
   if state['ops']>=3 and fail not in (1,2,3) and relation!='loaded_null':
    assert lifecycle[2]['kind']=='init' and lifecycle[2]['target']==labels[opt['result']] and lifecycle[2]['global']=='old'
  expected_order=['kill','won','died']
  actual_type=old_type if relation=='same' else new_type
  expected_order+=['won','died','kill'] if actual_type=='StandardMode' else ['kill','won','died']
  completed_ops=state['ops']-int(state['error']=='delegate')
  stores=[v['slot'] for v in tr if v['kind']=='store' and v['slot']!='mode']
  assert stores==expected_order[:completed_ops]
  if relation=='loaded_null' and fail not in ('json',1,2,3):assert state['error']=='null_or_cast' and completed_ops==3
  if isinstance(fail,int) and relation!='old_null' and (relation!='loaded_null' or fail<=3):
   assert state['error']=='delegate' and completed_ops==fail-1
  published=any(v['kind']=='store' and v['slot']=='mode' for v in tr)
  for v in tr:
   if v['kind'] in ('init','deinit') or (v['kind']=='store' and v['slot']!='mode'):assert v['global']=='old'
  if 'mode_changed' in kinds:
   assert next(i for i,v in enumerate(tr) if v['kind']=='store' and v['slot']=='mode')<kinds.index('mode_changed')
  if fail=='mode_changed' and published:assert state['error']=='mode_changed' and 'game_init' not in kinds
  if fail=='game_init' and published:assert state['error']=='game_init'

  assert rq(game+0x10)==(opt['result'] if published else (0 if relation=='old_null' else old))
  if not state['error']:
   assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8
   assert kinds[-2:]==['mode_changed','game_init'] and published
   if relation not in ('old_null','loaded_null'):
    actual_type=old_type if relation=='same' else new_type
    assert rd(game+0x18)==(7 if actual_type=='StandardMode' else 99)
    counts={n:len(lists.get(rq(gameplay+o),[])) for o,n in slots.items()}
    assert counts=={'kill':3 if old_type=='RoguelikeStandard' else 1,'won':1,'died':1}
  if any(v['kind']=='mode_changed' for v in tr):assert published
  results.append({'old_type':old_type,'requested_type':new_type,'identity':relation,'failure':fail,'error':state['error'],'trace':tr,'final':snapshot()})
 before=snapshot();state.update(trace=[],error=None,ops=0,reads=0)
 sp=stack+0x8000;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,0);uc.reg_write(x.UC_X86_REG_RDX,0)
 uc.emu_start(base+0x3DBE10,stop,count=10000)
 assert state['error']=='null_or_cast' and not state['trace'] and snapshot()==before
 results.append({'identity':'null_input','error':state['error'],'unchanged':True})
 return {'schema_version':1,'build_id':BUILD,'cases_passed':len(results),'distinct_native_instructions':len(visited),'metadata_verified':sorted(set(verified)),'cases':results,'scope':'Actual ChangeGameMode and concrete LoadGame, saved getters, Init, DeInit execute natively. Nonempty preferences and JSON results, delegate construction/list operations, GC barriers and notification invocation are explicit gateways. Runtime classes initialized. No live event history, callback execution, class initializer or exception unwinding completeness claimed.'}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(f"Passed {r['cases_passed']} composed lifecycle cases")
