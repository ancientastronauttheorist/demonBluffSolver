"""Native Init -> ResetSavedCharacters -> saved/current copies -> ScoreOld; contrasting RestartGame."""
import argparse,hashlib,json,struct
from pathlib import Path
from audit_character_assets import BUILD
METHODS={'Init':0x37def0,'RestartGame':0x37ffc0,'ResetSavedCharacters':0x37fde0}
CALLBACKS=[(0x78,'HandOut'),(0x88,'SameHandOut'),(0x38,'RestartGame'),(0x50,'OnCharacterReveal'),(0x48,'IncreaseOrderCountOnHiddenKill'),(0x48,'ManageKilledCharacter'),(0x18,'UpdateScore'),(0,'SameHandOut')]
def audit(game_root,dumper_root):
 import capstone,pefile,unicorn
 from unicorn import x86_const as x
 assert unicorn.__version__=='2.1.4'
 repo=Path(__file__).parents[1];lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'));ex=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
 def pin(path,digest):
  b=path.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==digest.upper();return b
 raw=pin(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256']);m=json.loads(pin(Path(dumper_root)/'script.json',ex['outputs']['script_json']['sha256']).decode('utf-8-sig'));dump=pin(Path(dumper_root)/'dump.cs',ex['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
 decoded={};ends={};verified=[]
 for name,start in METHODS.items():
  row=next(r for r in m['ScriptMethod'] if r['Name']=='Gameplay$$'+name);assert row['Address']==start
  args='const MethodInfo* method' if name=='.cctor' else 'Gameplay_o* __this, const MethodInfo* method'
  assert row['Signature']==f'void Gameplay__{name.replace(".","_")} ({args});';verified.append(row)
  end=min(r['Address'] for r in m['ScriptMethod'] if r['Address']>start);ins=list(cs.disasm(pe.get_data(start,end-start),start))
  while ins[-1].mnemonic=='int3':ins.pop()
  assert all(a.address+a.size==b.address for a,b in zip(ins,ins[1:]));ends[name]=hex(ins[-1].address+ins[-1].size);decoded.update({i.address:i for i in ins})
 def instruction(a,mn,op):assert a in decoded and (decoded[a].mnemonic,decoded[a].op_str)==(mn,op)
 for a in [0x37e0de,0x3801ab]:instruction(a,'mov','dword ptr [rax + 0x3c], 0x32')
 for a in [0x37e0e5,0x3801b2]:instruction(a,'mov','dword ptr [rax + 0x40], 0xa')
 for a in [0x37e0ec,0x3801b9]:instruction(a,'mov','dword ptr [rax + 0x14], 0x64')
 instruction(0x37e106,'mov','qword ptr [rdx + 8], rbx');instruction(0x3801ee,'mov','qword ptr [rax + 8], rbx')
 score_type=next(r['Address'] for r in m['ScriptMetadata'] if r['Name']=='ScoreOld_TypeInfo')
 for a in [0x37e0ca,0x380197]:
  i=decoded[a];assert i.address+i.size+i.operands[1].mem.disp==score_type
 assert 'public static Score Score; // 0x8' in dump and 'public static Gameplay Instance; // 0x10' in dump
 # Bounded inventory of all distinct Gameplay-declared entries referencing this
 # known allocation type. This is not an all-indirect-writers absence proof.
 references=[];gameplay_entries={r['Address']:r for r in m['ScriptMethod'] if r['Name'].startswith('Gameplay$$')}
 for start,row in gameplay_entries.items():
  end=min(r['Address'] for r in m['ScriptMethod'] if r['Address']>start)
  for i in cs.disasm(pe.get_data(start,end-start),start):
   if any(o.type==capstone.CS_OP_MEM and o.mem.base==capstone.x86.X86_REG_RIP and i.address+i.size+o.mem.disp==score_type for o in i.operands):references.append({'method':row['Name'],'instruction':hex(i.address)})
 assert {r['method'] for r in references}=={'Gameplay$$Init','Gameplay$$RestartGame'}
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
 arena,stack,stop=0x200000000,0x300000000,0x400000000;uc.mem_map(arena,0x100000);uc.mem_map(stack,0x10000);uc.mem_map(stop,0x1000)
 def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
 def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
 def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
 def rd(a):return struct.unpack('<I',uc.mem_read(a,4))[0]
 def reg(r):return uc.reg_read(r)
 def ret(v=0):
  sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v&0xffffffffffffffff);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
 needed=['UnityEngine.Object','ProjectContext','Gameplay','GameData','GameEvents','GameplayEvents','PlayerController','ScoreOld','System.Action','System.Action<Character>','System.Collections.Generic.List<RelicData>','System.Collections.Generic.List<Character>','System.Collections.Generic.List<CharacterData>','System.Collections.Generic.List<SpecialRule>','Gameplay.<InitCoroutine>d__41']
 types={name:arena+0x1000+k*0x400 for k,name in enumerate(needed)};found=set()
 for row in m['ScriptMetadata']:
  name=row['Name'].removesuffix('_TypeInfo')
  if name in types:q(base+row['Address'],types[name]);found.add(name)
 assert found==set(types)
 token_names={};callback_metadata=[]
 for row in m['ScriptMetadataMethod']:
  if row['Name'] in [f'Method$Gameplay.{name}()' for name in {name for _,name in CALLBACKS}|{'Init'}]:
   assert row['MethodAddress']==next(r['Address'] for r in m['ScriptMethod'] if r['Name']=='Gameplay$$'+row['Name'].split('.')[1].split('(')[0])
   p=arena+0x50000+len(token_names)*0x100;token_names[p]=row['Name'];q(base+row['Address'],p);callback_metadata.append(row)
 assert len(token_names)==8
 for i in decoded.values():
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
 gp,gd,ge,gpe,pc,obj,oldscore,mode,mklass,player,health,hklass=[arena+v for v in range(0x10000,0x1c000,0x1000)]
 base_labels={arena+0x1c000:'previous_instance',0:'null',obj:'receiver',oldscore:'previous_score',mode:'mode',player:'player',health:'health'}
 slots={p+o:f'{name}+{o:X}' for name,p,offsets in [('static',gp,[0,8,0x10,0x18,0x20,0x30]),('instance',obj,list(range(0x20,0x68,8))+[0x88]),('GameEvents',ge,[0x18]),('GameplayEvents',gpe,sorted({o for o,_ in CALLBACKS}))] for o in offsets}
 project_static,project,game=arena+0x1d000,arena+0x1e000,arena+0x1f000
 state={};opt={};labels={};lists={};contents={};sources={};delegate_lists={};visited=set()
 def label(p):return labels.get(p,f'unknown:{p:X}')
 def snapshot():
  refs={name:label(rq(p)) for p,name in slots.items() if not name.startswith(('GameEvents','GameplayEvents'))}
  counters={name:rd(p) for name,p in [('currentLevel',obj+0x78),('currentDay',obj+0x7c),('startingLevel',obj+0x80),('GameplayState',gp+0x28),('PrevState',gp+0x2c),('CurrentReveal',gp+0x38)]}
  events={name:[list(h) for h in delegate_lists.get(rq(p),[])] for p,name in slots.items() if name.startswith(('GameEvents','GameplayEvents'))}
  list_headers={label(p):[rd(p+0x18),rd(p+0x1c)] for p in lists}
  score=rq(gp+8);score_fields=None if score==0 else {f'{o:X}':rd(score+o) for o in [0x14,0x3c,0x40]}
  return {'references':refs,'counters':counters,'events':events,'list_headers':list_headers,'score_defaults':score_fields,'project_present':rq(project_static)!=0,'list_contents':{label(p):contents.get(p,[])[:rd(p+0x18)] for p in lists},'copy_sources':{label(p):label(v) for p,v in sources.items()},'allocated_scores':{label(p):{f'{o:X}':rd(p+o) for o in [0x14,0x3c,0x40]} for p,n in labels.items() if n.endswith(':ScoreOld')}}
 def emit(name,**kw):
  state['events'].append({'kind':name,**kw,'snapshot':snapshot()});state['counts'][name]=state['counts'].get(name,0)+1
  if opt.get('fail')==[name,state['counts'][name]]:state['error']=name;uc.emu_stop();return False
  return True
 def allocate(typ):
  state['alloc']+=1;p=arena+0x60000+state['alloc']*0x1000;q(p,typ);labels[p]=f'allocation{state["alloc"]}:{next(n for n,t in types.items() if t==typ)}';return p
 def hook(_,a,size,__):
  r=a-base;c=reg(x.UC_X86_REG_RCX);dx=reg(x.UC_X86_REG_RDX);r8=reg(x.UC_X86_REG_R8)
  if a==stop:uc.emu_stop();return
  if a in (stop+0x100,stop+0x110,stop+0x120,stop+0x130):
   name={stop+0x100:'starting_level',stop+0x110:'reset_level',stop+0x120:'mode_kind',stop+0x130:'health_reset'}[a]
   if emit(name):ret(opt.get('mode_kind',0) if name=='mode_kind' else opt.get('levels',[2,5])[min(state['counts'][name]-1,1)] if name!='health_reset' else 0)
  elif r==0x2b7d40:
   typ=next(n for n,p in types.items() if p==c)
   if emit('allocate',type=typ):ret(allocate(c))
  elif r==0x1c822c0:
   assert c in (0,project) and dx==0
   if emit('unity_null_check',project_present=c!=0):
    if opt.get('replace_after_equality'):q(project_static,0)
    ret(0xabc000|int(opt.get('equality_null',False)))
  elif r==0x3dc6f0:
   assert c==game and dx in [10,20,30,100] and r8==0
   if emit('typed_pool',character_type=dx):
    if opt.get('replace_after_first_pool'):q(project_static,0)
    ret(0 if opt.get('null_pool')==dx else arena+0x38000+[10,20,30,100].index(dx)*0x400)
  elif r in (0xb02160,0xb610a0):
   if emit('list_ctor' if r==0xb02160 else 'list_copy',target=label(c),source=label(dx) if r==0xb610a0 else None):
    if r==0xb610a0 and dx==0:state['error']='null_list';uc.emu_stop();return
    lists[c]=True;contents[c]=contents[dx][:rd(dx+0x18)] if r==0xb610a0 else [];
    if r==0xb610a0:sources[c]=dx
    d(c+0x18,rd(dx+0x18) if r==0xb610a0 else 0);d(c+0x1c,0);ret()
  elif r==0x2b6ff0:
   assert c in slots,(hex(c),label(dx));assert rq(c)==dx
   if emit('store',slot=slots[c],value=label(dx)):ret()
  elif r==0x281d90:
   typ=next(n for n,p in types.items() if p==c)
   if emit('class_init',type=typ):d(c+0xe0,1);ret()
  elif r in (0x4d5170,0x4d5b60):
   assert r8 in token_names and dx==obj
   if emit('delegate_ctor',callback=token_names[r8]):delegate_lists[c]=[(label(dx),token_names[r8])];ret()
  elif r in (0x116bcc0,0x116e070):
   operation='combine' if r==0x116bcc0 else 'remove'
   if emit(operation):
    values=delegate_lists.get(c,[])[:];rhs=delegate_lists[dx]
    if operation=='combine':values+=rhs
    else:
     for index in range(len(values)-len(rhs),-1,-1):
      if values[index:index+len(rhs)]==rhs:del values[index:index+len(rhs)];break
    if values:
     typ=rq(dx)
     if opt.get('wrong_delegate_at')==state['counts'][operation]:typ=types['System.Action<Character>'] if typ==types['System.Action'] else types['System.Action']
     p=allocate(typ);delegate_lists[p]=values;ret(p)
    else:ret(0)
  elif r==0x2b7010:
   if emit('cast'):ret(c if rq(c)==dx else 0)
  elif r==0x2b7040:state['error']='cast';uc.emu_stop()
  elif r in (0x37fc90,0x37b620,0x37e240,0x1c7f160,0x1c79770,0x33ed50,0x112b9d0):
   name={0x37fde0:'reset_saved',0x37fc90:'reset_player',0x37b620:'change_state',0x37e240:'load_characters',0x1c7f160:'start_coroutine',0x1c79770:'monobehaviour_ctor',0x33ed50:'object_ctor',0x112b9d0:'array_clear'}[r]
   if emit(name,target=label(c)):
    if name=='array_clear':uc.mem_write(c+0x20+dx*8,bytes(r8*8))
    ret()
  elif r==0x2b7d90:state['error']='null';uc.emu_stop()
  elif r in decoded:visited.add(r)
  else:raise AssertionError(('unexpected gateway',hex(a)))
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 def run(name,options):
  opt.clear();opt.update(options);state.clear();state.update(events=[],counts={},error=None,alloc=0);labels.clear();labels.update(base_labels);lists.clear();contents.clear();sources.clear();delegate_lists.clear()
  uc.mem_write(arena+0x10000,bytes(0xF0000))
  for n,p in types.items():d(p+0xe0,0 if n in opt.get('cold',[]) else 1)
  q(types['ProjectContext']+0xb8,project_static);q(project_static,0 if opt.get('null')=='project' else project);q(project+0x20,0 if opt.get('null')=='project_game' else game)
  labels[project]='project';labels[game]='captured_game_data'
  for k,typ in enumerate([10,20,30,100]):
   p=arena+0x38000+k*0x400;labels[p]=f'pool{typ}';lists[p]=True;contents[p]=[typ*100+j for j in range(k+1)];d(p+0x18,k+1)
  for n,p in [('Gameplay',gp),('GameData',gd),('GameEvents',ge),('GameplayEvents',gpe),('PlayerController',pc)]:q(types[n]+0xb8,p)
  for n,p in [('currentLevel',obj+0x78),('currentDay',obj+0x7c),('startingLevel',obj+0x80),('GameplayState',gp+0x28),('PrevState',gp+0x2c),('CurrentReveal',gp+0x38)]:d(p,0xaabbccdd)
  for k,o in enumerate(list(range(0x20,0x68,8))+[0x88]):
   p=arena+0x20000+k*0x1000;labels[p]=f'previous_instance_{o:X}';q(obj+o,p);lists[p]=True;contents[p]=[o*100+j for j in range(opt.get('list_size',2))];d(p+0x18,opt.get('list_size',2));d(p+0x1c,0xffffffff);q(p+0x10,p+0x100);uc.mem_write(p+0x120,b'\x55'*16)
  for k,o in enumerate([0,0x18,0x20]):
   p=arena+0x30000+k*0x1000;labels[p]=f'previous_static_{o:X}';q(gp+o,p);lists[p]=True;contents[p]=[o*100+50+j for j in range(opt.get('list_size',2))];d(p+0x18,opt.get('list_size',2));d(p+0x1c,0xffffffff);q(p+0x10,p+0x100);uc.mem_write(p+0x120,b'\x66'*16)
  q(gp+8,oldscore);q(gp+0x10,arena+0x1c000);q(gp+0x30,oldscore)
  for o in [0x14,0x3c,0x40]:d(oldscore+o,777)
  q(gd+0x10,0 if opt.get('null')=='mode' else mode);q(mode,mklass)
  for o,target in [(0x1c8,stop+0x100),(0x1d8,stop+0x110),(0x178,stop+0x120)]:q(mklass+o,target)
  q(pc,0 if opt.get('null')=='player' else player);q(player+0x28,0 if opt.get('null')=='health_group' else health);q(health+0x10,0 if opt.get('null')=='health' else hklass);q(hklass,hklass+0x100);q(hklass+0x100+0x198,stop+0x130)
  if opt.get('null')=='deck':q(obj+0x20,0)
  if opt.get('null')=='relics':q(gp,0)
  if opt.get('null')=='saved':q(obj+0x48,0)
  # Initial duplicate lists make last-match removal and ordered additions visible.
  for p,key in slots.items():
   if key.startswith(('GameEvents','GameplayEvents')):
    matching=['Init'] if key.startswith('GameEvents') else [cb for o,cb in CALLBACKS if p==gpe+o]
    entries=[('receiver',f'Method$Gameplay.{cb}()') for cb in matching]
    entries=entries+ [('other','Method$Gameplay.Init()')] + entries if opt.get('duplicates',True) else []
    if entries:
     dp=arena+0x40000+(p-ge if p<gpe else p-gpe+0x1000);q(dp,types['System.Action<Character>'] if p in (gpe+0x48,gpe+0x50) else types['System.Action']);labels[dp]=key+':old_handlers';delegate_lists[dp]=entries;q(p,dp)
  before=snapshot();sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,obj)
  uc.emu_start(base+METHODS[name],stop+0x200,count=30000)
  return {'method':name,'input':dict(options),'initial':before,'final':snapshot(),'error':state['error'],'events':state['events'][:]}
 cases=[]
 def success(name,options):
  result=run(name,options);assert result['error'] is None,result
  before=result['initial'];after=result['final'];events=result['events']
  assert after['score_defaults']=={'14':100,'3C':50,'40':10}
  assert after['references']['static+8'].endswith(':ScoreOld')
  assert after['references']['static+10']==before['references']['static+10']
  assert after['counters']['currentDay']==0
  assert after['counters']['currentLevel']==options.get('levels',[2,5])[0]&0xffffffff
  assert after['counters']['startingLevel']==(options.get('levels',[2,5])[1]&0xffffffff if name=='Init' else before['counters']['startingLevel'])
  refreshed=name=='Init' and not options.get('equality_null',False)
  for index,(saved,current,typ) in enumerate(zip([0x48,0x50,0x58,0x60],[0x28,0x30,0x38,0x40],[10,20,30,100])):
   saved_id=after['references'][f'instance+{saved:X}'];current_id=after['references'][f'instance+{current:X}']
   assert current_id!=saved_id and after['copy_sources'][current_id]==saved_id
   if refreshed:
    assert saved_id!=before['references'][f'instance+{saved:X}']
    assert after['copy_sources'][saved_id]==f'pool{typ}'
    assert after['list_contents'][saved_id]==after['list_contents'][f'pool{typ}']
   else:assert saved_id==before['references'][f'instance+{saved:X}']
   assert after['list_contents'][current_id]==after['list_contents'][saved_id]
  assert after['events']==before['events']
  stores=[e['slot'] for e in events if e['kind']=='store']
  expected_stores=([f'instance+{o:X}' for o in [0x48,0x50,0x58,0x60]] if refreshed else [])+[f'instance+{o:X}' for o in [0x28,0x30,0x38,0x40]]+['static+8']
  assert stores==expected_stores,(name,stores)
  if name=='Init':
   assert after['list_headers']['previous_instance_20']==[0,0] and after['list_headers']['previous_static_0']==[0,0]
   assert not after['list_contents']['previous_instance_20'] and not after['list_contents']['previous_static_0']
  assert bool([e for e in events if e['kind']=='load_characters'])==(options.get('mode_kind',0)==0)
  cases.append(result);return result
 for name in ['Init','RestartGame']:
  for mode_value in [0,1,10,-1]:
   for levels in [[0,0],[-2**31,2**31-1],[2,5]]:success(name,{'mode_kind':mode_value,'levels':levels,'cold':['Gameplay','GameData','UnityEngine.Object']})
  baseline=success(name,{'cold':['Gameplay','GameData','UnityEngine.Object']})
  counts={}
  for index,event in enumerate(baseline['events']):
   gateway=event['kind'];counts[gateway]=counts.get(gateway,0)+1
   options=dict(baseline['input'],fail=[gateway,counts[gateway]])
   result=run(name,options);assert result['error']==gateway
   assert result['events']==baseline['events'][:index+1]
   assert result['final']==event['snapshot'];cases.append(result)
 for null in ['project','project_game']:
  result=run('Init',{'null':null});assert result['error']=='null' and result['final']['references']['static+8']=='previous_score';cases.append(result)
 for null in ['deck','relics','mode']:
  result=run('Init',{'null':null});assert result['error']=='null';cases.append(result)
  assert all(result['final']['references'][f'instance+{o:X}'].startswith('allocation') for o in [0x48,0x50,0x58,0x60])
  assert result['final']['references']['static+8'].endswith(':ScoreOld')==(null=='mode')
 for typ in [10,20,30,100]:
  result=run('Init',{'null_pool':typ});assert result['error']=='null_list'
  assert len([e for e in result['events'] if e['kind']=='store'])==[10,20,30,100].index(typ)
  assert result['final']['references']['static+8']=='previous_score';cases.append(result)
 result=run('Init',{'replace_after_equality':True});assert result['error']=='null' and result['final']['references']['static+8']=='previous_score';cases.append(result)
 result=success('Init',{'replace_after_first_pool':True});assert not result['final']['project_present']
 success('Init',{'equality_null':True})
 success('Init',{'equality_null':True,'null':'project'})
 for null in ['saved','player','health_group','health','mode']:
  result=run('RestartGame',{'null':null});assert result['error'] in ('null','null_list');assert result['final']['references']['static+8']=='previous_score';cases.append(result)
 # Intern snapshots while retaining exact failure prefixes and identities.
 table=[];indices={}
 for case in cases:
  for event in case['events']:
   snap=event.pop('snapshot');key=json.dumps(snap,sort_keys=True)
   if key not in indices:indices[key]=len(table);table.append(snap)
   event['snapshot_index']=indices[key]
 return {'schema_version':1,'build_id':BUILD,'cases_passed':len(cases),'distinct_native_instructions':len(visited),'metadata_verified':verified,'complete_entry_ends':ends,'score_type_reference_inventory':{'decoded_gameplay_entries':len(gameplay_entries),'references':references},'snapshot_table':table,'cases':cases,'boundary':'Actual Gameplay.Init and its ResetSavedCharacters callee; contrasting RestartGame copies existing saved lists without calling ResetSavedCharacters. Unity equality, typed-pool providers, allocation/copy/barrier, runtime initialization, ResetPlayerInfo, health, state-change, mode selectors, LoadCharacters and scheduling remain explicit gateways. Controlled ProjectContext replacement probes reread/capture only.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();result=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k in ['cases_passed','distinct_native_instructions','score_type_reference_inventory']}))
