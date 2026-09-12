"""Native duplicate-pool occurrence sampling and uncovered shuffle scaffolding."""
import argparse,hashlib,itertools,json,struct
from fractions import Fraction
from pathlib import Path
from audit_character_assets import BUILD

def audit(game_root,dumper_root):
 import capstone,pefile,unicorn
 from unicorn import x86_const as x
 assert unicorn.__version__=='2.1.4';repo=Path(__file__).parents[1];lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'));ext=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
 def pin(p,h):
  b=p.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==h.upper();return b
 raw=pin(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256']);meta=json.loads(pin(Path(dumper_root)/'script.json',ext['outputs']['script_json']['sha256']).decode('utf-8-sig'))
 specs=[('Characters$$PickRoundDuplicates',0x36D720,'void Characters__PickRoundDuplicates (Characters_o* __this, const MethodInfo* method);'),('Characters$$ShuffleDeck',0x36E490,'System_Collections_IEnumerator_o* Characters__ShuffleDeck (Characters_o* __this, const MethodInfo* method);'),('Characters.<ShuffleDeck>d__16$$System.Collections.IEnumerator.Reset',0x376C40,'void Characters__ShuffleDeck_d__16__System_Collections_IEnumerator_Reset (Characters__ShuffleDeck_d__16_o* __this, const MethodInfo* method);')];exact=[]
 for n,a,s in specs:
  r=next(r for r in meta['ScriptMethod'] if r['Name']==n);assert r['Address']==a and r['Signature']==s;exact.append(r)
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
 ends={0x36D720:0x36DA3C,0x36E490:0x36E4DB,0x376C40:0x376C7E,0x33ED50:0x33ED53};decoded={}
 for a,b in ends.items():
  buf=pe.get_data(a,b-a);assert len(buf)==b-a;ins=list(cs.disasm(buf,a));assert ins[-1].address+ins[-1].size==b and ins[-1].mnemonic!='int3';decoded.update({i.address:i for i in ins})
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
 for n in ('Gameplay_TypeInfo','Method$System.Collections.Generic.List<CharacterData>.Add()','Method$System.Collections.Generic.List<CharacterData>.Remove()','Method$System.Collections.Generic.List<CharacterData>.get_Item()','Characters.<ShuffleDeck>d__16_TypeInfo','System.NotSupportedException_TypeInfo'):assert n in bindings
 for i in decoded.values():
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
 owner,game,static,iterator=arena+0x8000,arena+0x9000,arena+0xA000,arena+0xB000;pool,villagers,outcasts,script,bluff= [arena+v for v in (0x20000,0x24000,0x28000,0x2C000,0x30000)]
 q(bindings['Gameplay_TypeInfo']+0xB8,static);add=bindings['Method$System.Collections.Generic.List<CharacterData>.Add()'];q(add+0x20,arena+0x10000);q(arena+0x10000+0xC0,arena+0x11000);q(arena+0x11000+0x70,arena+0x12000)
 data={i:arena+0x40000+i*0x100 for i in range(8)};labels={0:None}|{p:i for i,p in data.items()};state={};opt={};visited=set()
 def content(p):return [rq(rq(p+0x10)+0x20+8*i) for i in range(rd(p+0x18))]
 def fill(p,values,capacity=16):
  q(p+0x10,p+0x1000);d(p+0x18,len(values));d(p+0x1C,3);d(p+0x1000+0x18,capacity)
  for i,v in enumerate(values):q(p+0x1020+8*i,v)
 def event(n,**kw):
  state['counts'][n]=state['counts'].get(n,0)+1;state['events'].append({'event':n,**kw})
  if opt.get('fail')==[n,state['counts'][n]]:state['error']=n;uc.emu_stop();return False
  return True
 def append(p,value):
  index=rd(p+0x18);d(p+0x1C,rd(p+0x1C)+1);q(rq(p+0x10)+0x20+8*index,value);d(p+0x18,index+1)
 def hook(_,a,size,__):
  r=a-base;c,t,m=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
  if a==stop:uc.emu_stop();return
  if r==0x2B7B40:ret(rq(c))
  elif r==0x281D90:
   if event('class_init'):d(c+0xE0,1);ret()
  elif r==0x37DC00:
   assert c==game
   if event('script'):ret(script)
  elif r==0x112B9D0:
   assert c==pool+0x1000 and t==0
   if event('clear',count=m):uc.mem_write(c+0x20,bytes(m*8));ret()
  elif r==0x36A550:
   assert c==owner and t==script
   if event('bluffable'):ret(bluff)
  elif r==0x36B9C0:
   assert c==owner and t==bluff and m in (10,20)
   if event('real_type',value=m):ret(0 if opt.get('null_type')==m else villagers if m==10 else outcasts)
  elif r==0x369EB0:
   assert c==owner and t==bluff and m==10
   if event('discarded_good_filter'):ret(arena+0x35000)
  elif r==0x1C86600:
   assert c==0 and m==0;index=opt.get('choices',[])[len(state['draws'])] if len(state['draws'])<len(opt.get('choices',[])) else 0
   state['draws'].append({'width':t,'index':index})
   if event('rng',width=t,index=index):ret(index)
  elif r==0xB22150:
   assert m==bindings['Method$System.Collections.Generic.List<CharacterData>.get_Item()']
   if t>=rd(c+0x18):state['error']='bounds';uc.emu_stop()
   else:ret(content(c)[t])
  elif r==0x2B6FF0:
   assert rq(c)==t
   if event('villager_store',value=labels[t]):ret()
  elif r==0xB59E70:
   assert m==bindings['Method$System.Collections.Generic.List<CharacterData>.Remove()']
   if event('remove',pool='villager' if c==villagers else 'outcast',value=labels[t]):
    vals=content(c);index=vals.index(t);vals.pop(index);version=rd(c+0x1C);fill(c,vals);d(c+0x1C,version+1);ret(1)
  elif r==0x2EB0:
   assert c==pool and m==add
   if event('outcast_add',value=labels[t]):append(c,t);ret()
  elif r==0xB54090:
   assert c==pool and m==arena+0x12000
   if event('grow',value=labels[t]):
    index=rd(c+0x18);q(rq(c+0x10)+0x20+8*index,t);d(c+0x18,index+1);d(rq(c+0x10)+0x18,index+16);ret()
  elif r==0x2B7D40:
   if event('allocate'):
    p=arena+0x50000;uc.mem_write(p,bytes(0x100));q(p,c);ret(p)
  elif r==0x111B720:
   assert rq(c)==bindings['System.NotSupportedException_TypeInfo']
   if event('exception_ctor'):ret()
  elif r==0x2B7D50:
   assert t==bindings['Method$Characters.<ShuffleDeck>d__16.System.Collections.IEnumerator.Reset()'];state['error']='not_supported';uc.emu_stop()
  elif r in (0x2B7D90,0x2B7D80):state['error']='null' if r==0x2B7D90 else 'bounds';uc.emu_stop()
  elif r==0x37C3F0:raise AssertionError('stable successful services cannot reach fallback')
  elif r in decoded:visited.add(r)
  else:raise AssertionError(hex(r))
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 cases=[]
 def run(vs,outs,options=None,entry=0x36D720):
  opt.clear();opt.update(options or {});state.clear();state.update(events=[],counts={},draws=[],error=None)
  q(static+0x10,0 if opt.get('null_game') else game);q(owner+0x48,0 if opt.get('null_pool') else pool);d(bindings['Gameplay_TypeInfo']+0xE0,0 if opt.get('cold') else 1)
  fill(pool,[] if opt.get('capacity',16)==0 else [data[7]],opt.get('capacity',16));fill(villagers,[data[v] for v in vs]);fill(outcasts,[data[v] for v in outs]);d(iterator+0x10,opt.get('iterator_state',1));q(iterator+0x18,data[7])
  sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,iterator if entry==0x376C40 else 0 if opt.get('null_receiver') else owner);uc.reg_write(x.UC_X86_REG_RDX,0)
  keep=[x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
  for rr in keep:uc.reg_write(rr,0xABC000+rr)
  uc.emu_start(base+entry,stop,count=7000)
  if state['error'] is None:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8 and all(reg(rr)==0xABC000+rr for rr in keep)
  result={'entry':hex(entry),'villagers':vs,'outcasts':outs,'options':opt.copy(),'events':state['events'][:],'draws':state['draws'][:],'output':[labels[p] for p in content(pool)],'remaining_villagers':[labels[p] for p in content(villagers)],'remaining_outcasts':[labels[p] for p in content(outcasts)],'version':rd(pool+0x1C),'error':state['error']};cases.append(result);return result
 for vs,outs in itertools.product(([],[0],[0,0,1],[0,1,2,3,4,5]),([],[6],[6,6])):
  r=run(vs,outs);assert r['error']==('bounds' if not vs else None)
  assert r['output']==(vs[:4]+outs[:1] if vs else [])
  if not vs:assert r['draws']==[{'width':0,'index':0}]
 weighted=[]
 for choices in itertools.product(range(3),range(2),range(1),range(3)):
  r=run([0,0,1],[6,6,7],{'choices':list(choices)});assert r['error'] is None
  expected=[];remaining=[0,0,1];remaining_out=[6,6,7]
  for index in choices[:3]:
   chosen=remaining[index];expected.append(chosen);remaining.remove(chosen)
  chosen=remaining_out[choices[3]];expected.append(chosen);remaining_out.remove(chosen)
  assert r['output']==expected and r['remaining_villagers']==remaining and r['remaining_outcasts']==remaining_out
  sampling=[e['event'] for e in r['events'] if e['event'] in ('villager_store','outcast_add','remove')]
  assert sampling==['villager_store','remove']*3+['outcast_add','remove']
  probability=Fraction(1,1)
  for draw in r['draws']:probability/=draw['width']
  assert probability==Fraction(1,18);weighted.append({'choices':choices,'probability':str(probability),'output':r['output']})
 assert sum((Fraction(r['probability']) for r in weighted),Fraction())==1
 for fail in (['script',1],['clear',1],['bluffable',1],['real_type',1],['real_type',2],['discarded_good_filter',1],['rng',1],['villager_store',1],['remove',1],['outcast_add',1],['remove',2]):
  r=run([0],[6],{'fail':fail});assert r['error']==fail[0]
  expected=[7] if fail[0]=='script' else [0,6] if fail==['remove',2] else [0] if fail in (['villager_store',1],['remove',1],['outcast_add',1]) else []
  assert r['output']==expected
 for opts in ({'null_game':True},{'null_pool':True},{'null_type':10},{'null_type':20}):assert run([0],[6],opts)['error']=='null'
 r=run([0],[],{'capacity':0,'fail':['grow',1]});assert r['error']=='grow' and r['output']==[] and r['version']==5
 r=run([0],[],{'cold':True,'fail':['class_init',1]});assert r['output']==[7]
 for null in (False,True):
  r=run([],[],{'null_receiver':null},0x36E490);p=reg(x.UC_X86_REG_RAX);assert r['error'] is None and rd(p+0x10)==0 and rq(p+0x18)==0 and rq(p+0x20)==0
 for value in (-1,0,1):
  r=run([],[],{'iterator_state':value},0x376C40);assert r['error']=='not_supported' and rd(iterator+0x10)==value&0xffffffff and rq(iterator+0x18)==data[7]
 return {'schema_version':1,'build_id':BUILD,'exact_declarations':exact,'cases_passed':len(cases),'distinct_native_instructions':len(visited),'weighted_paths':weighted,'cases':cases,'scope':'Native duplicate sampling/cap/clear/inlineAdd and uncovered ShuffleDeck factory/Reset. Candidate filters, uniform RNG, get_Item bounds, first-equal Remove, growth and outcastAdd are explicit stable services. No complete shuffle MoveNext recounting, live RNG or collection-internal failure rollback claimed.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(f"Passed {r['cases_passed']} duplicate/factory cases")
