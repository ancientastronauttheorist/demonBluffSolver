"""Actual script concatenation and CharacterData filters into round duplicates."""
import argparse, hashlib, itertools, json, struct
from pathlib import Path
from audit_character_assets import BUILD

ENTRIES = {0x36d720: 'PickRoundDuplicates', 0x37dc00: 'GetScriptCharacters',
           0x36a550: 'FilterBluffableCharacters', 0x36b9c0: 'FilterRealCharacterType',
           0x369eb0: 'FilterAlignmentCharacters'}
ASSETS = {0: [10,20,1], 1: [10,10,1], 2: [20,20,1], 3: [30,10,1],
          4: [10,10,0], 5: [20,10,1], 6: [100,20,255], 7: [10,10,1]}

def audit(game_root, dumper_root):
 import capstone, pefile, unicorn
 from unicorn import x86_const as x
 assert unicorn.__version__ == '2.1.4'
 repo=Path(__file__).parents[1]
 lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
 ext=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
 def pin(p,h):
  b=p.read_bytes(); assert hashlib.sha256(b).hexdigest().upper()==h.upper(); return b
 raw=pin(game_root/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256'])
 meta=json.loads(pin(dumper_root/'script.json',ext['outputs']['script_json']['sha256']).decode('utf-8-sig'))
 dump=pin(dumper_root/'dump.cs',ext['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
 for field in ['public ECharacterType type; // 0x130','public EAlignment startingAlignment; // 0x134','public bool bluffable; // 0x13C']:
  assert field in dump,field
 pe=pefile.PE(data=raw,fast_load=True); base=pe.OPTIONAL_HEADER.ImageBase
 cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64); cs.detail=True
 decoded={}; exact=[]; ends={}
 for start,name in ENTRIES.items():
  rows=[r for r in meta['ScriptMethod'] if r['Address']==start and r['Name'].endswith('$$'+name)]
  assert len(rows)==1
  if start==0x36d720:signature='void Characters__PickRoundDuplicates (Characters_o* __this, const MethodInfo* method);'
  elif start==0x37dc00:signature='System_Collections_Generic_List_CharacterData__o* Gameplay__GetScriptCharacters (Gameplay_o* __this, const MethodInfo* method);'
  else:
   extra=', int32_t type' if start==0x36b9c0 else ', int32_t alignment' if start==0x369eb0 else ''
   signature=f'System_Collections_Generic_List_CharacterData__o* Characters__{name} (Characters_o* __this, System_Collections_Generic_List_CharacterData__o* inpuCharacters{extra}, const MethodInfo* method);'
  assert rows[0]['Signature']==signature;exact.append(rows[0])
  end=min(r['Address'] for r in meta['ScriptMethod'] if r['Address']>start)
  ins=list(cs.disasm(pe.get_data(start,end-start),start))
  while ins[-1].mnemonic=='int3':ins.pop()
  assert ins[0].address==start and all(a.address+a.size==b.address for a,b in zip(ins,ins[1:]))
  ends[name]=hex(ins[-1].address+ins[-1].size); decoded.update({i.address:i for i in ins})
 assert ends=={'PickRoundDuplicates':'0x36da3c','GetScriptCharacters':'0x37dcc3','FilterBluffableCharacters':'0x36a6c4','FilterRealCharacterType':'0x36bb3d','FilterAlignmentCharacters':'0x36a02d'}
 checks={0x36a64f:('cmp','byte ptr [rdx + 0x13c], 0'),0x36bac1:('cmp','dword ptr [rdx + 0x130], esi'),
         0x369fb1:('cmp','dword ptr [rdx + 0x134], esi')}
 for a,expected in checks.items():assert (decoded[a].mnemonic,decoded[a].op_str)==expected
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64)
 uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
 arena,stack,stop=0x200000000,0x300000000,0x400000000
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
  if r['Address'] in refs:
   p=arena+0x1000+len(bindings)*0x200;bindings[r['Name']]=p;q(base+r['Address'],p)
 for name in ['Gameplay_TypeInfo','System.Collections.Generic.List<CharacterData>_TypeInfo',
              'Method$System.Collections.Generic.List<CharacterData>.Add()',
              'Method$System.Collections.Generic.List<CharacterData>.get_Item()',
              'Method$System.Collections.Generic.List<CharacterData>.Remove()']:
  assert name in bindings,name
 for i in decoded.values():
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:
   uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
 owner,game,static,pool=arena+0x10000,arena+0x11000,arena+0x12000,arena+0x20000
 q(bindings['Gameplay_TypeInfo']+0xb8,static)
 data={i:arena+0x40000+i*0x200 for i in ASSETS}; labels={0:None}|{p:i for i,p in data.items()}
 lists={};state={};opt={};visited=set()
 def contents(p):return [rq(rq(p+0x10)+0x20+i*8) for i in range(rd(p+0x18))]
 def fill(p,values,version=0):
  q(p+0x10,p+0x1000);d(p+0x18,len(values));d(p+0x1c,version);d(p+0x1018,128)
  for i,v in enumerate(values):q(p+0x1020+i*8,v)
 def append(p,v):
  n=rd(p+0x18);q(rq(p+0x10)+0x20+n*8,v);d(p+0x18,n+1);d(p+0x1c,rd(p+0x1c)+1)
 def snap():return {name:{'items':[labels[v] for v in contents(p)],'version':rd(p+0x1c)} for p,name in lists.items()}
 def emit(kind,**kw):
  state['counts'][kind]=state['counts'].get(kind,0)+1
  state['events'].append({'kind':kind,**kw,'snapshot':snap()})
  if opt.get('fail')==[kind,state['counts'][kind]]:state['error']=kind;uc.emu_stop();return False
  return True
 def hook(_,a,size,__):
  r=a-base;c,t,m=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
  if a==stop:uc.emu_stop();return
  if r in ENTRIES:
   state['stage']=('villagers' if m==10 else 'outcasts') if r==0x36b9c0 else {0x36d720:'duplicates',0x37dc00:'script',0x36a550:'bluffable',0x369eb0:'discarded_good'}[r]
   state['entries'].append({'method':ENTRIES[r],'type':m if r in (0x36b9c0,0x369eb0) else None})
  if r==0x281d90:
   assert c==bindings['Gameplay_TypeInfo']
   if emit('class_init'):d(c+0xe0,1);ret()
  elif r==0x2b7d40:
   assert c==bindings['System.Collections.Generic.List<CharacterData>_TypeInfo']
   if emit('allocate',list=state['stage']):
    state['alloc']+=1;p=arena+0x50000+state['alloc']*0x3000;uc.mem_write(p,bytes(0x2000));lists[p]=state['stage'];q(p,c);fill(p,[]);ret(p)
  elif r==0xb02160:
   assert c in lists
   if emit('list_ctor',list=lists[c]):ret()
  elif r==0xb53f50:
   assert c in lists
   if emit('append_range',source=lists.get(t),destination=lists[c]):
    if t==0:state['error']='null_collection';uc.emu_stop()
    else:
     values=contents(t);old=contents(c);fill(c,old+values,rd(c+0x1c)+1);ret()
  elif r==0xb16640:
   assert t in lists
   if emit('enumerator',source=lists[t]):uc.mem_write(c,bytes(24));q(c,t);d(c+8,0);ret(c)
  elif r==0x9693d0:
   source=rq(c);index=rd(c+8)
   if emit('move_next',source=lists[source]):
    values=contents(source);d(c+8,index+1);q(c+0x10,values[index] if index<len(values) else 0);ret(int(index<len(values)))
  elif r==0x33ed50:
   if emit('dispose'):ret()
  elif r==0x112b9d0:
   assert c==pool+0x1000 and t==0
   if emit('clear',count=m):uc.mem_write(c+0x20,bytes(m*8));ret()
  elif r==0x2eb0:
   assert c in lists
   if emit('outcast_add' if c==pool else 'filter_add',list=lists[c],value=labels[t]):append(c,t);ret()
  elif r==0x1c86600:
   assert c==0 and m==0
   index=opt.get('choices',[])[len(state['draws'])] if len(state['draws'])<len(opt.get('choices',[])) else 0
   state['draws'].append({'width':t,'index':index})
   if emit('rng',width=t,index=index):ret(index)
  elif r==0xb22150:
   assert m==bindings['Method$System.Collections.Generic.List<CharacterData>.get_Item()']
   if t>=rd(c+0x18):state['error']='bounds';uc.emu_stop()
   else:ret(contents(c)[t])
  elif r==0x2b6ff0:
   assert rq(c)==t
   if emit('villager_store',value=labels[t]):ret()
  elif r==0xb59e70:
   assert m==bindings['Method$System.Collections.Generic.List<CharacterData>.Remove()']
   if emit('remove',list=lists[c],value=labels[t]):
    values=contents(c);values.remove(t);fill(c,values,rd(c+0x1c)+1);ret(1)
  elif r in (0x2b7d90,0x2b7d80):state['error']='null' if r==0x2b7d90 else 'bounds';uc.emu_stop()
  elif r==0x37c3f0:raise AssertionError('stable services cannot reach fallback')
  else:
   assert r in decoded and decoded[r].size==size,hex(r)
   visited.add(r)
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 def run(rosters,options=None):
  opt.clear();opt.update(options or {});state.clear();state.update(counts={},events=[],entries=[],draws=[],alloc=0,error=None)
  lists.clear();lists[pool]='duplicates';fill(pool,[data[7]],3)
  for i,values in enumerate(rosters):
   p=arena+0x24000+i*0x3000;lists[p]=f'roster{i}';fill(p,[0 if v is None else data[v] for v in values or []]);q(game+0x28+i*8,0 if values is None else p)
  for ident,(typ,alignment,bluffable) in ASSETS.items():d(data[ident]+0x130,typ);d(data[ident]+0x134,alignment);uc.mem_write(data[ident]+0x13c,bytes([bluffable]))
  q(static+0x10,0 if opt.get('null_game') else game);q(owner+0x48,0 if opt.get('null_pool') else pool);d(bindings['Gameplay_TypeInfo']+0xe0,0 if opt.get('cold') else 1)
  initial=snap();sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,owner);uc.reg_write(x.UC_X86_REG_RDX,0)
  keep=[x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
  for rr in keep:uc.reg_write(rr,0xabc000+rr)
  uc.emu_start(base+0x36d720,stop,count=30000)
  if state['error'] is None:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8 and all(reg(rr)==0xabc000+rr for rr in keep)
  return {'input':{'rosters':rosters,'options':dict(opt)},'initial':initial,'final':snap(),'entries':state['entries'][:],'events':state['events'][:],'draws':state['draws'][:],'error':state['error']}
 cases=[];rosters=[[0,0,1,4],[2,2,5],[3],[6]]
 for choices in itertools.product(range(3),range(2),range(1),range(3)):
  result=run(rosters,{'choices':list(choices)});assert result['error'] is None
  villagers=[0,0,1];outcasts=[2,2,5];output=[]
  for index in choices[:3]:v=villagers[index];output.append(v);villagers.remove(v)
  v=outcasts[choices[3]];output.append(v);outcasts.remove(v)
  assert result['final']['duplicates']['items']==output
  assert result['final']['villagers']['items']==villagers and result['final']['outcasts']['items']==outcasts
  assert result['final']['bluffable']['items']==[0,0,1,2,2,5,3,6]
  assert result['final']['discarded_good']['items']==[1,5,3]
  assert [d['width'] for d in result['draws']]==[3,2,1,3]
  result['probability']={'numerator':1,'denominator':18};cases.append(result)
 for inputs,expected,error in [([[1,1,1],[1,1,1],[3],[6]],[1,1,1,1],None),
                               ([[4],[2],[3],[6]],[],'bounds')]:
  result=run(inputs);assert result['error']==error and result['final']['duplicates']['items']==expected
  assert len(result['draws'])==(4 if error is None else 1);cases.append(result)
 baseline=run(rosters,{'cold':True});assert baseline['error'] is None;cases.append(baseline)
 counts={}
 for index,event in enumerate(baseline['events']):
  kind=event['kind'];counts[kind]=counts.get(kind,0)+1
  result=run(rosters,{'cold':True,'fail':[kind,counts[kind]]})
  assert result['error']==kind and result['events']==baseline['events'][:index+1]
  assert result['final']==event['snapshot'];cases.append(result)
 for rosters2,options,error in [([[],[],[],[]],{},'bounds'),([[4],[2],[],[]],{},'bounds'),
   ([[0],[],[],[]],{},None),([[0],[],[],[]],{'null_game':True},'null'),
   ([[0],[],[],[]],{'null_pool':True},'null')]:
  result=run(rosters2,options);assert result['error']==error;cases.append(result)
 for i in range(4):
  inputs=[v[:] for v in rosters];inputs[i]=None;result=run(inputs)
  assert result['error']=='null_collection' and result['final']['duplicates']['items']==[7];cases.append(result)
 for i in range(4):
  inputs=[v[:] for v in rosters];inputs[i].append(None);result=run(inputs)
  assert result['error']=='null' and result['final']['duplicates']['items']==[] and not result['draws'];cases.append(result)
 # Intern repeated snapshots without retaining proprietary bytes.
 snapshots=[];indices={}
 for case in cases:
  for event in case['events']:
   snap0=event.pop('snapshot');key=json.dumps(snap0,sort_keys=True)
   if key not in indices:indices[key]=len(snapshots);snapshots.append(snap0)
   event['snapshot_index']=indices[key]
 return {'build_id':BUILD,'cases_passed':len(cases),'weighted_paths':18,'metadata_verified':exact,
  'entry_ends':ends,'native_assertions':len(checks),'native_instructions_executed':len(visited),
  'asset_fields':ASSETS,'snapshot_table':snapshots,'cases':cases,
  'scope':'Actual PickRoundDuplicates -> GetScriptCharacters and Bluffable/RealType/Alignment filters. Warm metadata; explicit allocation/list construction, stable enumerators, AddRange/Add/Remove, RNG/bounds, GC and class-init services. Discarded Good filter executes; no Unity PRNG, List implementation or unwind recovery claims.'}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',required=True,type=Path)
 a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(r['cases_passed'],r['native_instructions_executed'])
