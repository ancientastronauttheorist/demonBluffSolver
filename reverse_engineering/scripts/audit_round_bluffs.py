"""Native PickRoundBluffs ordering, weighted selection and actual capture predicate."""
import argparse,hashlib,itertools,json,struct
from pathlib import Path
from audit_character_assets import BUILD
from audit_round_candidate_composition import ASSETS

def audit(game_root,dumper_root):
 import capstone,pefile,unicorn
 from unicorn import x86_const as x
 assert unicorn.__version__=='2.1.4';repo=Path(__file__).parents[1]
 lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'));ext=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
 def pin(p,h):
  b=p.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==h.upper();return b
 raw=pin(game_root/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256'])
 meta=json.loads(pin(dumper_root/'script.json',ext['outputs']['script_json']['sha256']).decode('utf-8-sig'))
 specs=[('Characters$$PickRoundBluffs',0x36d3a0,'void Characters__PickRoundBluffs (Characters_o* __this, const MethodInfo* method);'),
 ('Characters.<>c__DisplayClass22_0$$<PickRoundBluffs>b__0',0x377170,'bool Characters___c__DisplayClass22_0___PickRoundBluffs_b__0 (Characters___c__DisplayClass22_0_o* __this, CharacterData_o* cd, const MethodInfo* method);')]
 exact=[]
 for name,a,signature in specs:
  rows=[r for r in meta['ScriptMethod'] if r['Name']==name];assert len(rows)==1 and rows[0]['Address']==a and rows[0]['Signature']==signature;exact.append(rows[0])
 verified_gateways=[]
 for address,name in [(0x37c3f0,'GetAscensionAllStartingCharacters'),(0x37dc00,'GetScriptCharacters'),(0x37c1a0,'GetAllAscensionCharacters')]:
  rows=[r for r in meta['ScriptMethod'] if r['Address']==address and r['Name']=='Gameplay$$'+name]
  assert len(rows)==1 and rows[0]['Signature']==f'System_Collections_Generic_List_CharacterData__o* Gameplay__{name} (Gameplay_o* __this, const MethodInfo* method);'
  verified_gateways.append(rows[0])
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True;decoded={}
 for a,b in [(0x36d3a0,0x36d71a),(0x377170,0x3771c3)]:
  ins=list(cs.disasm(pe.get_data(a,b-a),a));assert ins[0].address==a and sum(i.size for i in ins)==b-a;decoded.update({i.address:i for i in ins})
 checks={0x36d4e4:('inc','dword ptr [rcx + 0x1c]'),0x36d4e7:('mov','dword ptr [rcx + 0x18], esi'),
 0x36d504:('call','0x2b7d40'),0x36d537:('call','0xb59980'),0x36d644:('cmp','dword ptr [rax + 0x18], 1'),
 0x36d6b5:('call','0x36b9c0'),0x37719c:('mov','rcx, qword ptr [rbx + 0x10]'),0x3771b9:('jmp','0xb55950')}
 for a,v in checks.items():assert (decoded[a].mnemonic,decoded[a].op_str)==v
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
 refs=set()
 for i in decoded.values():
  for o in i.operands:
   if o.type==capstone.CS_OP_MEM and o.mem.base==capstone.x86.X86_REG_RIP:refs.add(i.address+i.size+o.mem.disp)
 bindings={}
 for r in meta['ScriptMetadata']+meta['ScriptMetadataMethod']:
  if r['Address'] in refs:p=arena+0x1000+len(bindings)*0x200;bindings[r['Name']]=p;q(base+r['Address'],p)
 for name in ['Gameplay_TypeInfo','Characters.<>c__DisplayClass22_0_TypeInfo','System.Predicate<CharacterData>_TypeInfo',
              'Method$Characters.<>c__DisplayClass22_0.<PickRoundBluffs>b__0()',
              'Method$System.Collections.Generic.List<CharacterData>.Contains()']:
  assert name in bindings,name
 for i in decoded.values():
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
 owner,game,static,closure,predicate=arena+0x10000,arena+0x11000,arena+0x12000,arena+0x14000,arena+0x15000
 pointers={name:arena+0x20000+i*0x3000 for i,name in enumerate(['unique','all','script','bluffable','villagers','outcasts','fallback','fallback_bluffable','fallback_good','fallback_villagers'])}
 names={p:n for n,p in pointers.items()};data={i:arena+0x50000+i*0x200 for i in ASSETS};labels={0:None}|{p:i for i,p in data.items()}
 q(bindings['Gameplay_TypeInfo']+0xb8,static);state={};opt={};visited=set()
 def contents(p):return [rq(rq(p+0x10)+0x20+i*8) for i in range(rd(p+0x18))]
 def fill(p,vals,version=0):
  q(p+0x10,p+0x1000);d(p+0x18,len(vals));d(p+0x1c,version);d(p+0x1018,128)
  for i,v in enumerate(vals):q(p+0x1020+i*8,v)
 def append(p,v):
  n=rd(p+0x18);q(rq(p+0x10)+0x20+n*8,v);d(p+0x18,n+1);d(p+0x1c,rd(p+0x1c)+1)
 def snapshot():return {'lists':{n:{'items':[labels[v] for v in contents(p)],'version':rd(p+0x1c)} for n,p in pointers.items()},'captured_script':names.get(rq(closure+0x10)),'project_present':rq(static+0x10)!=0}
 def emit(kind,**kw):
  state['counts'][kind]=state['counts'].get(kind,0)+1;state['events'].append({'kind':kind,**kw,'snapshot':snapshot()})
  if opt.get('fail')==[kind,state['counts'][kind]]:state['error']=kind;uc.emu_stop();return False
  return True
 def hook(_,a,size,__):
  r=a-base;c,t,m=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
  if a==stop:uc.emu_stop();return
  if r==0x2b7d40:
   kind='closure' if c==bindings['Characters.<>c__DisplayClass22_0_TypeInfo'] else 'predicate';assert c==bindings['Characters.<>c__DisplayClass22_0_TypeInfo'] if kind=='closure' else c==bindings['System.Predicate<CharacterData>_TypeInfo']
   if emit('allocate',object=kind):ret(closure if kind=='closure' else predicate)
  elif r==0x33ed50:
   assert c==closure
   if emit('object_ctor'):ret()
  elif r==0x281d90:
   assert c==bindings['Gameplay_TypeInfo']
   if emit('class_init'):d(c+0xe0,1);ret()
  elif r in (0x37c3f0,0x37dc00,0x37c1a0):
   assert c==game
   name={0x37c3f0:'all',0x37dc00:'script',0x37c1a0:'fallback'}[r]
   if emit('getter',source=name):
    if opt.get('replace_after_getter')==name:q(static+0x10,0)
    ret(0 if opt.get('null_getter')==name else pointers[name])
  elif r==0x2b6ff0:
   assert c==closure+0x10 and rq(c)==t
   if emit('capture',source=names.get(t)):ret()
  elif r==0x112b9d0:
   assert c==pointers['unique']+0x1000 and t==0
   if emit('clear',count=m):uc.mem_write(c+0x20,bytes(m*8));ret()
  elif r==0xc8b620:
   assert c==predicate and t==closure and m==bindings['Method$Characters.<>c__DisplayClass22_0.<PickRoundBluffs>b__0()']
   if emit('predicate_ctor'):ret()
  elif r==0xb59980:
   assert c==pointers['all'] and t==predicate
   if emit('remove_all'):
    captured=rq(closure+0x10)
    if captured==0 and contents(c):state['error']='null';uc.emu_stop()
    else:
     before=contents(c);after=[v for v in before if v not in (contents(captured) if captured else [])];fill(c,after,rd(c+0x1c)+int(len(before)!=len(after)));ret(len(before)-len(after))
  elif r in (0x36a550,0x36b9c0,0x369eb0):
   assert c==owner
   source=names.get(t);name=('fallback_bluffable' if source=='fallback' else 'bluffable') if r==0x36a550 else 'fallback_good' if r==0x369eb0 else 'fallback_villagers' if source=='fallback_good' else 'villagers' if m==10 else 'outcasts'
   if emit('filter',destination=name,source=source,requested=m if r!=0x36a550 else None):
    if t==0:state['error']='null';uc.emu_stop()
    else:
     vals=contents(t)
     if 0 in vals:state['error']='null';uc.emu_stop()
     else:
      selected=[v for v in vals if (ASSETS[labels[v]][2]!=0 if r==0x36a550 else ASSETS[labels[v]][1 if r==0x369eb0 else 0]==m)]
      fill(pointers[name],selected);ret(0 if opt.get('null_filter')==name else pointers[name])
  elif r==0x1c86600:
   assert c==0 and m==0
   index=opt.get('choices',[])[len(state['draws'])] if len(state['draws'])<len(opt.get('choices',[])) else 0;state['draws'].append({'width':t,'index':index})
   if emit('rng',width=t,index=index):ret(index)
  elif r==0xb22150:
   if t>=rd(c+0x18):state['error']='bounds';uc.emu_stop()
   else:ret(contents(c)[t])
  elif r==0x2eb0:
   assert c==pointers['unique']
   if emit('add',value=labels[t]):append(c,t);ret()
  elif r==0xb59e70:
   assert c in (pointers['villagers'],pointers['outcasts'])
   if emit('remove',source=names[c],value=labels[t]):
    vals=contents(c);vals.remove(t);fill(c,vals,rd(c+0x1c)+1);ret(1)
  elif r==0xb55950:
   assert c==pointers['script'] and m==bindings['Method$System.Collections.Generic.List<CharacterData>.Contains()']
   if emit('contains',value=labels[t]):ret(opt.get('contains_return',0xabc001 if t in contents(c) else 0xabc000))
  elif r==0x2b7d90:state['error']='null';uc.emu_stop()
  else:assert r in decoded and decoded[r].size==size,hex(r);visited.add(r)
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 def run(all_items,script_items,fallback,options=None,predicate_case=False):
  opt.clear();opt.update(options or {});state.clear();state.update(counts={},events=[],draws=[],error=None)
  for n,p in pointers.items():fill(p,[0 if v is None else data[v] for v in {'unique':[7],'all':all_items,'script':script_items,'fallback':fallback}.get(n,[])],3 if n=='unique' else 0)
  q(closure+0x10,0 if not predicate_case or opt.get('null_capture') else pointers['script']);q(owner+0x40,0 if opt.get('null_pool') else pointers['unique']);q(static+0x10,0 if opt.get('null_game') else game);d(bindings['Gameplay_TypeInfo']+0xe0,0 if opt.get('cold') else 1)
  initial=snapshot();sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,closure if predicate_case else owner);uc.reg_write(x.UC_X86_REG_RDX,0 if opt.get('candidate') is None else data[opt['candidate']])
  keep=[x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
  for rr in keep:uc.reg_write(rr,0xabc000+rr)
  uc.emu_start(base+(0x377170 if predicate_case else 0x36d3a0),stop,count=10000)
  if state['error'] is None:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8 and all(reg(rr)==0xabc000+rr for rr in keep)
  return {'method':'predicate' if predicate_case else 'PickRoundBluffs','input':{'all':all_items,'script':script_items,'fallback':fallback,'options':dict(opt)},'initial':initial,'final':snapshot(),'events':state['events'][:],'draws':state['draws'][:],'error':state['error'],'return_al':reg(x.UC_X86_REG_RAX)&255 if predicate_case and state['error'] is None else None}
 cases=[]
 for choices in itertools.product(range(3),range(2),range(1),range(3)):
  result=run([0,0,1,2,2,5],[],[7],{'choices':list(choices)});assert result['error'] is None
  vs=[0,0,1];outs=[2,2,5];expected=[]
  for index in choices[:3]:v=vs[index];expected.append(v);vs.remove(v)
  v=outs[choices[3]];expected.append(v);outs.remove(v)
  assert result['final']['lists']['unique']['items']==expected;assert [d['width'] for d in result['draws']]==[3,2,1,3]
  result['probability']={'numerator':1,'denominator':18};cases.append(result)
 for all_items in [[],[0],[2]]:
  for index in range(3):
   choices=([0] if all_items else [])+[index];result=run(all_items,[1],[1,1,7,3,4,6],{'choices':choices})
   assert result['error'] is None and result['final']['lists']['unique']['items']==all_items+[[1,1,7][index]]
   assert result['final']['lists']['fallback_villagers']['items']==[1,1,7]
   assert len([e for e in result['events'] if e['kind']=='remove'])==len(all_items)
   result['probability']={'numerator':1,'denominator':3};cases.append(result)
 for all_items,script_items,fallback,error,expected in [([0]*6,[],[],None,[0]*4),([0,0,1,1,2],[1],[],None,[0,0,2]),([],[],[],'bounds',[]),([0],[],[],'bounds',[0]),([None],[None],[7],None,[7]),([None],[],[7],'null',[])]:
  result=run(all_items,script_items,fallback);assert result['error']==error and result['final']['lists']['unique']['items']==expected
  if error=='bounds':assert result['draws'][-1]=={'width':0,'index':0}
  cases.append(result)
 for all_items in [[0,1,2],[]]:
  baseline=run(all_items,[],[7],{'cold':True});assert baseline['error'] is None;cases.append(baseline);counts={}
  for index,event in enumerate(baseline['events']):
   kind=event['kind'];counts[kind]=counts.get(kind,0)+1;result=run(all_items,[],[7],{'cold':True,'fail':[kind,counts[kind]]})
   assert result['error']==kind and result['events']==baseline['events'][:index+1] and result['final']==event['snapshot'];cases.append(result)
 for options in [{'null_game':True},{'null_pool':True},*({'null_getter':n} for n in ['all','script','fallback']),*({'null_filter':n} for n in ['bluffable','villagers','outcasts','fallback_bluffable','fallback_good','fallback_villagers']),{'replace_after_getter':'all'},{'replace_after_getter':'script'}]:
  result=run([0] if options.get('null_getter')=='script' else [],[],[7],options);assert result['error']=='null';cases.append(result)
 for all_items,options,expected in [([],{'null_getter':'script'},[7]),([0,1],{'replace_after_getter':'script'},[0,1]),([],{'replace_after_getter':'fallback'},[7])]:
  result=run(all_items,[],[7],options);assert result['error'] is None and result['final']['lists']['unique']['items']==expected;cases.append(result)
 for captured,candidate,value,fail in itertools.product([False,True],[None,0],[0xabc000,0xabc001],[False,True]):
  options={'null_capture':not captured,'candidate':candidate,'contains_return':value}
  if fail:options['fail']=['contains',1]
  result=run([],[None,0],[],options,True)
  assert result['error']==('null' if not captured else 'contains' if fail else None)
  if captured and not fail:assert result['return_al']==value&255
  assert result['initial']==result['final'];cases.append(result)
 table=[];indices={}
 for case in cases:
  for event in case['events']:
   snap=event.pop('snapshot');key=json.dumps(snap,sort_keys=True)
   if key not in indices:indices[key]=len(table);table.append(snap)
   event['snapshot_index']=indices[key]
 return {'build_id':BUILD,'metadata_verified':exact,'verified_gateways':verified_gateways,'cases_passed':len(cases),'native_assertions':len(checks),'native_instructions_executed':len(visited),'asset_fields':ASSETS,'snapshot_table':table,'cases':cases,
 'scope':'Actual PickRoundBluffs caller and separate actual capture predicate. Supplied getter/filter/list services; RemoveAll commits after supplied reference-membership results and does not execute the predicate body within the caller. Contains predicate executed separately with supplied Boolean return bits. Warm metadata, allocation/GC/class-init and uniform occurrence RNG are explicit. No comparer internals or unwind rollback claims.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',required=True,type=Path)
 a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(r['cases_passed'],r['native_instructions_executed'])
