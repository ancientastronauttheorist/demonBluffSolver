"""Native Score/ScoreNew/ScoreOld and Gameplay.UpdateScore composition.

Enumerated content, engine transforms, presentation, mode publication and class
initialization are preserving gateways. All score arithmetic executes natively.
"""
import argparse, hashlib, itertools, json, math, re, struct
from pathlib import Path
from audit_character_assets import BUILD
from audit_gameplay_score_resources import i32, mul, f32
FIELDS={'completedStages':0x10,'pointsForCompleting':0x14,'completedDays':0x18,'multiplier':0x1c,'roundPoints':0x20,'overallPoints':0x24,'killedGoods':0x28,'tempUnrevealedCards':0x2c,'unrevealedCards':0x30,'killedEvils':0x34,'tempKilledEvils':0x38,'pointPerKill':0x3c,'pointsPerUnrevealed':0x40}
BODIES={'Score':{'GetBaseScore':0x388b60,'GetMultiplier':0x388e10,'AddPointsOnEvilKill':0x388a60,'UpdateFullScore':0x389110,'GetFullPoints':0x388400,'.ctor':0x388820},'ScoreNew':{'GetBaseScore':0x388150,'GetMultiplier':0x388410,'AddPointsOnEvilKill':0x388050,'UpdateFullScore':0x388710,'GetFullPoints':0x388400,'.ctor':0x388820},'ScoreOld':{'AddPointsOnEvilKill':0x388830,'UpdateFullScore':0x388920,'GetFullPoints':0x388900,'.ctor':0x388a40}}
def bits(v):return struct.unpack('<I',struct.pack('<f',v))[0]
def cvtt(v):
 value=f32(v)
 return -2**31 if not math.isfinite(value) or value < -2**31 or value >= 2**31 else int(value)
def full(kind,s):
 if kind=='ScoreOld':return i32(s['pointPerKill']*s['killedEvils']+s['pointsPerUnrevealed']*s['unrevealedCards']+s['completedDays']*s['pointsForCompleting'])
 return i32(s['completedDays']*s['pointsForCompleting']+s['overallPoints'])

def expected(kind,method,initial,opt):
 s=dict(initial);events=[];counts={};error=None;value=None;initialized=set()
 class Halt(Exception):pass
 def event(name,**kw):
  events.append({'kind':name,**kw,'state':dict(s)});counts[name]=counts.get(name,0)+1
  if opt.get('fail')==[name,counts[name]]:raise Halt(name)
 def require(condition):
  if not condition:raise Halt('null')
 def cls(name):
  if name in opt.get('cold',[]) and name not in initialized:event('class_init',type=name);initialized.add(name)
 def fold(multiplier):
  cls('Gameplay');require(opt.get('null')!='instance');event('characters')
  require(opt.get('null')!='characters')
  value=0x3f800000 if multiplier else 0
  for which in ['characters','relics']:
   if which=='relics':require(opt.get('null')!='relics')
   event('enumerator',source=which)
   for item in opt.get(which,[]):
    event('move_next',source=which);require(item is not None);require(item!='null_info')
    value=mul(value,item[1]) if multiplier else i32(value+item[0])
   event('move_next',source=which);event('dispose',source=which)
  if multiplier:
   event('unrevealed');value=mul(mul(bits(min(opt.get('unrevealed',2),3)),bits(1.25)),value)
  return value
 def visual(amount):
  require(opt.get('null')!='character');require(opt.get('null')!='view');event('transform');require(opt.get('null')!='transform');event('position');require(opt.get('null')!='vfx');event('floating_score',amount=amount,position_bits=[0x3f800000,0xc0000000,0x40400000])
 try:
  if method=='GetBaseScore':value=fold(False)
  elif method=='GetMultiplier':value=fold(True)
  elif method=='.ctor':
   s['pointsForCompleting']=100
   if kind=='ScoreOld':s['pointPerKill']=50;s['pointsPerUnrevealed']=10
   event('object_ctor')
  elif method=='GetFullPoints':value=full(kind,s)
  elif method=='AddPointsOnEvilKill':
   if kind=='ScoreOld':
    event('unrevealed');n=opt.get('unrevealed',2);s['tempUnrevealedCards']=i32(s['tempUnrevealedCards']+n);s['tempKilledEvils']=i32(s['tempKilledEvils']+1);amount=i32(n*s['pointsPerUnrevealed']+s['pointPerKill'])
   else:
    event('base_score');base=fold(False);event('multiplier');multiplier=fold(True);amount=cvtt(mul(multiplier,bits(base)));s['roundPoints']=i32(s['roundPoints']+amount)
   visual(amount)
  elif method in ('UpdateFullScore','Gameplay.UpdateScore'):
   cls('Gameplay')
   if method=='Gameplay.UpdateScore':require(opt.get('null')!='score')
   require(opt.get('null')!='instance');s['completedStages']=i32(opt.get('level',6)+1)
   if kind=='ScoreOld':
    s['killedEvils']=i32(s['killedEvils']+s['tempKilledEvils']);s['unrevealedCards']=i32(s['unrevealedCards']+s['tempUnrevealedCards']);s['tempKilledEvils']=0;s['tempUnrevealedCards']=0
   else:s['overallPoints']=i32(s['overallPoints']+s['roundPoints']);s['roundPoints']=0
   s['completedDays']=i32(opt.get('day',3)+1)
   cls('GameData');event('full_points');total=full(kind,s);require(opt.get('null')!='mode');event('mode_update',score=total,level=i32(opt.get('level',6)+1))
 except Halt as halt:error=str(halt)
 # Initialized Gameplay is reused between the two getter calls and caller/callee.
 return s,value,error,events

def audit(game_root,dumper_root):
 import capstone,pefile,unicorn
 from unicorn import x86_const as x
 assert unicorn.__version__=='2.1.4'
 repo=Path(__file__).parents[1];lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'));ex=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
 def pin(path,digest):
  raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==digest.upper();return raw
 raw=pin(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256']);metadata=json.loads(pin(Path(dumper_root)/'script.json',ex['outputs']['script_json']['sha256']).decode('utf-8-sig'));dump=pin(Path(dumper_root)/'dump.cs',ex['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
 declarations=[]
 for kind,methods in BODIES.items():
  block=re.search(r'public (?:abstract )?class '+kind+r'\b[^\n]*\n\{(.*?)\n\}',dump,re.S).group(1)
  for method,rva in methods.items():
   row=next(r for r in metadata['ScriptMethod'] if r['Name']==kind+'$$'+method);assert row['Address']==rva;declarations.append(row)
   return_type='float' if method=='GetMultiplier' else 'int32_t' if method in ('GetBaseScore','GetFullPoints') else 'void'
   params=f'{kind}_o* __this'+(', Character_o* ch' if method=='AddPointsOnEvilKill' else ', int32_t completedDays' if method=='UpdateFullScore' else '')+', const MethodInfo* method'
   assert row['Signature']==f'{return_type} {kind}__{method.replace(".","_")} ({params});'
   if method!='.ctor':
    slot={'GetBaseScore':4,'GetMultiplier':5,'AddPointsOnEvilKill':6,'UpdateFullScore':7,'GetFullPoints':8}[method]
    assert re.search(r'// RVA: 0x'+f'{rva:X}'+r'[^\n]* Slot: '+str(slot)+r'\n\s*public (?:virtual|override) ',block)
  for field,offset in FIELDS.items():
   if (offset<0x28 and kind=='Score') or (offset>=0x28 and kind=='ScoreOld'):assert f'public int {field}; // 0x{offset:X}' in block
 for field in ['public Transform icon; // 0x20','public RoguelikeDataInfo roguelikeInfo; // 0x40','public int point; // 0x10','public float pointsMult; // 0x14']:assert field in dump
 assert re.search(r'Slot: 17\n\s*public abstract void UpdateScore\(int score, int level\)',dump)
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
 decoded={};ends={}
 for start in {a for methods in BODIES.values() for a in methods.values()}|{0x381910}:
  end=min(r['Address'] for r in metadata['ScriptMethod'] if r['Address']>start);ins=list(cs.disasm(pe.get_data(start,end-start),start))
  while ins[-1].mnemonic=='int3':ins.pop()
  assert all(a.address+a.size==b.address for a,b in zip(ins,ins[1:]));ends[hex(start)]=hex(ins[-1].address+ins[-1].size);decoded.update({i.address:i for i in ins})
 def instruction(a,m,o):assert a in decoded and (decoded[a].mnemonic,decoded[a].op_str)==(m,o)
 for a in [0x388ad0,0x3880c0]:instruction(a,'cvttss2si','ebp, xmm0')
 for a in [0x389211,0x388811,0x388a28]:instruction(a,'jmp','qword ptr [r10 + 0x248]')
 for a in [0x3890b3,0x3886b3]:instruction(a,'cmovg','eax, ecx')
 for a in [0x3890bd,0x3886bd]:
  i=decoded[a];assert struct.unpack('<f',pe.get_data(i.address+i.size+i.operands[1].mem.disp,4))[0]==1.25
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
 arena,stack,stop=0x200000000,0x300000000,0x400000000;uc.mem_map(arena,0x20000);uc.mem_map(stack,0x10000);uc.mem_map(stop,0x1000)
 def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
 def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
 def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
 def rd(a):return struct.unpack('<I',uc.mem_read(a,4))[0]
 def reg(r):return uc.reg_read(r)
 def ret(v=0):
  sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v&0xffffffffffffffff);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
 types={n:arena+0x1000+k*0x400 for k,n in enumerate(['Gameplay','GameData','VfxController'])}
 for n,p in types.items():
  row=next(r for r in metadata['ScriptMetadata'] if r['Name']==n+'_TypeInfo');q(base+row['Address'],p)
 for i in decoded.values():
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
 gp,gd,vfx,instance,obj,klass,mode,mklass,char,view,transform,chars,relics= [arena+v for v in range(0x3000,0x10000,0x1000)]
 state={};opt={};visited=set()
 def fields():return {n:i32(rd(obj+o)) for n,o in FIELDS.items()}
 def emit(name,**kw):
  state['events'].append({'kind':name,**kw,'state':fields()});state['counts'][name]=state['counts'].get(name,0)+1
  if opt.get('fail')==[name,state['counts'][name]]:state['error']=name;uc.emu_stop();return False
  return True
 def hook(_,a,size,__):
  r=a-base;c=reg(x.UC_X86_REG_RCX)
  if a==stop:uc.emu_stop();return
  if a==stop+0x100:
   assert c==mode
   if emit('mode_update',score=i32(reg(x.UC_X86_REG_RDX)&0xffffffff),level=i32(reg(x.UC_X86_REG_R8)&0xffffffff)):ret()
  elif r==0x281d90:
   name=next(n for n,p in types.items() if p==c)
   if emit('class_init',type=name):d(c+0xe0,1);ret()
  elif r==0x37c320:
   assert c==instance
   if emit('characters'):ret(0 if opt.get('null')=='characters' else chars)
  elif r==0x398d10:
   if emit('unrevealed'):ret(opt.get('unrevealed',2))
  elif r==0xb16640:
   source='characters' if reg(x.UC_X86_REG_RDX)==chars else 'relics';assert reg(x.UC_X86_REG_RDX) in (chars,relics)
   if emit('enumerator',source=source):uc.mem_write(c,bytes(24));q(c,chars if source=='characters' else relics);state['indices'][source]=0;ret(c)
  elif r==0x9693d0:
   source='characters' if rq(c)==chars else 'relics';assert rq(c) in (chars,relics)
   if emit('move_next',source=source):
    index=state['indices'][source];values=opt.get(source,[]);state['indices'][source]+=1
    if index==len(values):ret(0)
    else:q(c+0x10,0 if values[index] is None else arena+(0x10000 if source=='characters' else 0x14000)+index*0x100);ret(1)
  elif r==0x33ed50:
   name='object_ctor' if state['method']=='.ctor' else 'dispose';kw={} if name=='object_ctor' else {'source':'characters' if rq(c)==chars else 'relics'}
   if emit(name,**kw):ret()
  elif r==0x1c7a010:
   assert c==view
   if emit('transform'):ret(0 if opt.get('null')=='transform' else transform)
  elif r==0x1c91b80:
   assert reg(x.UC_X86_REG_RDX)==transform
   if emit('position'):uc.mem_write(c,struct.pack('<III',0x3f800000,0xc0000000,0x40400000));ret(c)
  elif r==0x3cea40:
   assert c==vfx
   if emit('floating_score',amount=i32(reg(x.UC_X86_REG_RDX)&0xffffffff),position_bits=list(struct.unpack('<III',uc.mem_read(reg(x.UC_X86_REG_R8),12)))):ret()
  elif r==0x2b7d90:state['error']='null';uc.emu_stop()
  elif r in decoded:
   visited.add(r)
   if r==state['base'] and state['method']=='AddPointsOnEvilKill':emit('base_score')
   elif r==state['mult'] and state['method']=='AddPointsOnEvilKill':emit('multiplier')
   elif r==state['full'] and state['method'] in ('UpdateFullScore','Gameplay.UpdateScore'):emit('full_points')
  else:raise AssertionError(('unexpected gateway',hex(a)))
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 def run(kind,method,initial,options):
  opt.clear();opt.update(options);state.clear();state.update(events=[],counts={},indices={},error=None,method=method,base=BODIES.get(kind,{}).get('GetBaseScore',0x388b60),mult=BODIES.get(kind,{}).get('GetMultiplier',0x388e10),full=BODIES[kind]['GetFullPoints'])
  uc.mem_write(arena+0x3000,bytes(0x1d000))
  for n,p in types.items():d(p+0xe0,0 if n in opt.get('cold',[]) else 1)
  for n,p in [('Gameplay',gp),('GameData',gd),('VfxController',vfx)]:q(types[n]+0xb8,p)
  q(gp,0 if opt.get('null')=='relics' else relics);q(gp+8,0 if opt.get('null')=='score' else obj);q(gp+0x10,0 if opt.get('null')=='instance' else instance)
  q(gd+0x10,0 if opt.get('null')=='mode' else mode);q(vfx,0 if opt.get('null')=='vfx' else vfx)
  q(mode,mklass);q(mklass+0x248,stop+0x100)
  q(obj,klass)
  for off,rva in [(0x178,state['base']),(0x188,state['mult']),(0x1a8,BODIES[kind]['UpdateFullScore']),(0x1b8,state['full'])]:q(klass+off,base+rva)
  for n,o in FIELDS.items():d(obj+o,initial[n])
  d(instance+0x78,opt.get('level',6));d(instance+0x7c,opt.get('day',3));q(char+0x20,0 if opt.get('null')=='view' else view)
  for source,offset,fieldoff in [('characters',0x10000,0x40),('relics',0x14000,0x30)]:
   for k,value in enumerate(opt.get(source,[])):
    p=arena+offset+k*0x100;q(p+fieldoff,0 if value=='null_info' else p+0x60)
    if isinstance(value,list):d(p+0x70,value[0]);d(p+0x74,value[1])
  before=bytes(uc.mem_read(arena+0x3000,0x1d000))
  sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,instance if method=='Gameplay.UpdateScore' else obj);uc.reg_write(x.UC_X86_REG_RDX,0 if opt.get('null')=='character' else char if method=='AddPointsOnEvilKill' else opt.get('day',3)&0xffffffff);uc.reg_write(x.UC_X86_REG_MXCSR,0x1f80)
  uc.emu_start(base+(0x381910 if method=='Gameplay.UpdateScore' else BODIES[kind][method]),stop+0x200,count=30000)
  after=bytes(uc.mem_read(arena+0x3000,0x1d000));skip=obj-(arena+0x3000)
  assert before[:skip]==after[:skip] and before[skip+0x44:]==after[skip+0x44:]
  value=None if state['error'] or method not in ['GetBaseScore','GetMultiplier','GetFullPoints'] else reg(x.UC_X86_REG_XMM0)&0xffffffff if method=='GetMultiplier' else i32(reg(x.UC_X86_REG_RAX)&0xffffffff)
  want,wvalue,error,events=expected(kind,method,initial,opt)
  assert fields()==want,(kind,method,opt,fields(),want)
  expected_memory=bytearray(before)
  for name,offset in FIELDS.items():struct.pack_into('<I',expected_memory,skip+offset,want[name]&0xffffffff)
  assert bytes(expected_memory)==after,(kind,method,'unexpected state byte write')
  assert state['error']==error,(kind,method,opt,state['error'],error)
  assert state['events']==events,(kind,method,opt,state['events'],events)
  if not error:assert value==wvalue,(kind,method,opt,value,wvalue)
  return {'type':kind,'method':method,'initial':initial,'input':dict(options),'state':fields(),'value':value,'error':error,'events':events}
 cases=[]
 initial={n:i32(0x10203040+k*101) for k,n in enumerate(FIELDS)}
 def case(kind,method,options=None,values=None):cases.append(run(kind,method,dict(initial if values is None else values),options or {}))
 for kind in BODIES:
  case(kind,'.ctor');case(kind,'.ctor',{'fail':['object_ctor',1]})
  for day,level,round_value in itertools.product([-2**31,-1,0,2**31-1],[-2**31,6,2**31-1],[-2**31,0,2**31-1]):
   values=dict(initial,roundPoints=round_value,tempKilledEvils=round_value,tempUnrevealedCards=round_value)
   for method in ['GetFullPoints','UpdateFullScore','Gameplay.UpdateScore']:case(kind,method,{'day':day,'level':level},values)
  for method in ['UpdateFullScore','Gameplay.UpdateScore']:
   for null in ['instance','mode']+(['score'] if method=='Gameplay.UpdateScore' else []):case(kind,method,{'null':null})
   for fail in [['class_init',1],['class_init',2],['full_points',1],['mode_update',1]]:case(kind,method,{'cold':['Gameplay','GameData'],'fail':fail})
  case(kind,'Gameplay.UpdateScore',{'cold':['Gameplay','GameData']})
  if kind!='ScoreOld':
   for method in ['GetBaseScore','GetMultiplier','AddPointsOnEvilKill']:
    case(kind,method,{'cold':['Gameplay'],'characters':[[3,0x40000000]],'relics':[[4,0x3f000000]]})
    case(kind,method,{'cold':['Gameplay'],'fail':['class_init',1]})
   sequences=[[],[[2,0x3f800000]],[[2**31-1,0x40000000],[1,0x3f000000]],[[-2**31,0x7f7fffff],[-1,0x40000000]],[[16777217,0x3f800001]],[[1,0x7fc12345]],[[1,0x7f812345]],[[1,0x80000000]],[[1,1]],[[1,0xff800000]]]
   for method in ['GetBaseScore','GetMultiplier','AddPointsOnEvilKill']:
    for chars_v,rels_v,n in itertools.product(sequences,sequences,[-2**31,-1,0,2,3,4,2**31-1]):case(kind,method,{'characters':chars_v,'relics':rels_v,'unrevealed':n})
    for null in ['instance','characters','relics']:case(kind,method,{'null':null})
    for source in ['characters','relics']:
     for value in [None,'null_info']:case(kind,method,{source:[[3,0x40000000],value]})
    for fail in [['characters',1],['enumerator',1],['enumerator',2],['move_next',1],['move_next',3],['dispose',1],['dispose',2]]+([['unrevealed',1]] if method!='GetBaseScore' else []):case(kind,method,{'characters':[[3,0x40000000]],'relics':[[4,0x3f000000]],'fail':fail})
   for fail in [['base_score',1],['multiplier',1]]:case(kind,'AddPointsOnEvilKill',{'fail':fail})
  else:
   for n in [-2**31,-1,0,1,2**31-1]:case(kind,'AddPointsOnEvilKill',{'unrevealed':n})
   case(kind,'AddPointsOnEvilKill',{'fail':['unrevealed',1]})
  for null in ['character','view','transform','vfx']:case(kind,'AddPointsOnEvilKill',{'null':null})
  for fail in [['transform',1],['position',1],['floating_score',1]]:case(kind,'AddPointsOnEvilKill',{'fail':fail})
 state_table=[];state_indices={}
 for case_result in cases:
  for event in case_result['events']:
   snapshot=event.pop('state');key=tuple(snapshot[n] for n in FIELDS)
   if key not in state_indices:state_indices[key]=len(state_table);state_table.append(snapshot)
   event['state_index']=state_indices[key]
 return {'schema_version':1,'event_state_table':state_table,'build_id':BUILD,'cases_passed':len(cases),'distinct_native_instructions':len(visited),'metadata_verified':declarations,'complete_entry_ends':ends,'cases':cases,'boundary':'Actual Score arithmetic and Gameplay.UpdateScore virtual composition; warmed metadata and controlled preserving providers, enumeration, class initialization, transforms, floating-score VFX, and mode.UpdateScore. No native exceptions/collection bodies or live UI.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();result=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['cases','metadata_verified','complete_entry_ends','event_state_table']}))
