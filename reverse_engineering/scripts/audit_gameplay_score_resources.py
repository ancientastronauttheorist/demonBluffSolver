"""Pinned Gameplay score/resource callers; provider/delegate boundaries explicit."""
import argparse, hashlib, itertools, json, math, re, struct
from pathlib import Path
from audit_character_assets import BUILD
ENTRIES={'GetScore':(0x37d8c0,0x37d90f),'GetScoreMultiplayer':(0x37d750,0x37d8b6),'UpdateScore':(0x381910,0x381982),'GetMaxDay':(0x37c870,0x37c8df),'GetCharactersLevel':(0x37c5a0,0x37c70d)}
def i32(v):return (v+2**31)%2**32-2**31
def f32(v):return struct.unpack('<f',struct.pack('<I',v))[0]
def mul(a,b):
 def nan(v):return v&0x7fffffff>0x7f800000
 if nan(a):return a|0x400000
 if nan(b):return b|0x400000
 v=f32(a)*f32(b)
 if math.isnan(v):return 0xffc00000
 try:return struct.unpack('<I',struct.pack('<f',v))[0]
 except OverflowError:return 0xff800000 if v<0 else 0x7f800000

def audit(game_root,dumper_root):
 import capstone,pefile,unicorn
 from unicorn import x86_const as x
 assert unicorn.__version__=='2.1.4'
 repo=Path(__file__).parents[1]
 lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
 extraction=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
 def pin(path,digest):
  raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==digest.upper();return raw
 raw=pin(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256'])
 metadata=json.loads(pin(Path(dumper_root)/'script.json',extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
 dump=pin(Path(dumper_root)/'dump.cs',extraction['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
 fields={'Gameplay':['public static List<RelicData> CurrentRelics; // 0x0','public static Score Score; // 0x8','public int currentLevel; // 0x78','public int currentDay; // 0x7C'], 'ProjectContext':['public GameData gameData; // 0x20','public static ProjectContext Instance; // 0x0'], 'GameData':['public AscensionsData currentTemporaryAscension; // 0x78'], 'RelicData':['public RoguelikeDataInfo roguelikeData; // 0x30'], 'RoguelikeDataInfo':['public float pointsMult; // 0x14']}
 for name,needles in fields.items():
  block=re.search(r'public (?:abstract )?class '+name+r'\b[^\n]*\n\{(.*?)\n\}',dump,re.S).group(1)
  assert all(needle in block for needle in needles),(name,needles)
 assert '// RVA: 0x389110 Offset: 0x387D10 VA: 0x180389110 Slot: 7' in dump
 assert 'public abstract EGameMode GetGameMode();' in dump
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase
 cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
 decoded={};verified=[]
 for name,(start,end) in ENTRIES.items():
  rows=[r for r in metadata['ScriptMethod'] if r['Name']=='Gameplay$$'+name];assert len(rows)==1 and rows[0]['Address']==start;verified+=rows
  result_type={'GetScore':'Score_o*','GetScoreMultiplayer':'float','UpdateScore':'void','GetMaxDay':'int32_t','GetCharactersLevel':'int32_t'}[name]
  params='const MethodInfo* method' if name in ('GetScore','GetScoreMultiplayer') else 'Gameplay_o* __this, const MethodInfo* method'
  assert rows[0]['Signature']==f'{result_type} Gameplay__{name} ({params});'
  next_entry=min(r['Address'] for r in metadata['ScriptMethod'] if r['Address']>start)
  ins=list(cs.disasm(pe.get_data(start,next_entry-start),start))
  while ins[-1].mnemonic=='int3':ins.pop()
  assert ins[-1].address+ins[-1].size==end
  decoded.update({i.address:i for i in ins})
 def instruction(a,m,o):assert a in decoded and (decoded[a].mnemonic,decoded[a].op_str)==(m,o)
 instruction(0x381976,'jmp','qword ptr [rax + 0x1a8]')
 instruction(0x381967,'mov','edx, dword ptr [rbx + 0x7c]')
 instruction(0x37c618,'mov','rax, qword ptr [rdx + 0x178]')
 instruction(0x37c67a,'jge','0x37c68a')
 instruction(0x37c6c1,'dec','eax');instruction(0x37c8d3,'dec','eax')
 instruction(0x37d84c,'mulss','xmm6, dword ptr [rcx + 0x14]')
 i=decoded[0x37d7a7];assert struct.unpack('<I',pe.get_data(i.address+i.size+i.operands[1].mem.disp,4))[0]==0x3f800000
 for name,rva in [('AscensionsData$$GetCharactersCount',0x3b1cb0),('UnityEngine.Random$$Range',0x1c86600),('Score$$UpdateFullScore',0x389110)]:
  row=next(r for r in metadata['ScriptMethod'] if r['Name']==name and r['Address']==rva);verified.append(row)
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
 arena,stack,stop=0x200000000,0x300000000,0x400000000
 uc.mem_map(arena,0x20000);uc.mem_map(stack,0x10000);uc.mem_map(stop,0x1000)
 def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
 def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
 def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
 def reg(r):return uc.reg_read(r)
 def ret(v=0):
  sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v&0xffffffffffffffff);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
 types={n:arena+0x1000+k*0x400 for k,n in enumerate(['Gameplay','GameData','ProjectContext'])}
 for n,p in types.items():
  row=next(r for r in metadata['ScriptMetadata'] if r['Name']==n+'_TypeInfo');q(base+row['Address'],p)
 for i in decoded.values():
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
 gp,gd,pc,obj,mode,klass,score,sklass,project,data,asc,count,rels= [arena+v for v in range(0x3000,0x10000,0x1000)]
 state={};opt={};visited=set()
 def emit(name,**kw):
  state['events'].append({'kind':name,**kw});state['counts'][name]=state['counts'].get(name,0)+1
  if opt.get('fail')==[name,state['counts'][name]]:state['error']=name;uc.emu_stop();return False
  return True
 def hook(_,a,size,__):
  r=a-base;c=reg(x.UC_X86_REG_RCX)
  if a==stop:uc.emu_stop();return
  if a==stop+0x100:
   if emit('mode'):ret(opt.get('mode',1))
  elif a==stop+0x110:
   assert c==score and reg(x.UC_X86_REG_R8)==0xabcdef
   if emit('update_full_score',day=i32(reg(x.UC_X86_REG_RDX)&0xffffffff)):ret()
  elif r==0x281d90:
   assert c in types.values()
   if emit('class_init'):d(c+0xe0,1);ret()
  elif r==0x3b1cb0:
   assert c==asc
   index=state['counts'].get('counts',0)
   if emit('counts'):
    sizes=opt.get('sizes',[7]);value=sizes[min(index,len(sizes)-1)]
    if value is None:ret(0)
    else:d(count+0x18,value);ret(count)
  elif r==0x1c86600:
   if emit('random',minimum=i32(c&0xffffffff),maximum=i32(reg(x.UC_X86_REG_RDX)&0xffffffff)):ret(opt.get('random',123))
  elif r==0xb16640:
   assert reg(x.UC_X86_REG_RDX)==rels
   if emit('enumerator'):uc.mem_write(c,bytes(24));ret(c)
  elif r==0x9693d0:
   index=state['counts'].get('move_next',0)
   if emit('move_next'):
    values=opt.get('factors',[])
    if index==len(values):ret(0)
    else:q(c+0x10,0 if values[index]=='null_relic' else arena+0x10000+index*0x100);ret(1)
  elif r==0x33ed50:
   if emit('dispose'):ret()
  elif r==0x2b7d90:state['error']='null';uc.emu_stop()
  elif r in decoded:visited.add(r)
  else:raise AssertionError(('unexpected gateway',hex(a)))
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 def run(name,options):
  opt.clear();opt.update(options);state.clear();state.update(events=[],counts={},error=None)
  uc.mem_write(arena+0x3000,bytes(0x1d000))
  for n,p in types.items():d(p+0xe0,0 if opt.get('cold') else 1)
  for n,p in [('Gameplay',gp),('GameData',gd),('ProjectContext',pc)]:q(types[n]+0xb8,p)
  q(gp,0 if opt.get('null')=='relics' else rels);q(gp+8,0 if opt.get('null')=='score' else score)
  q(score,sklass);q(sklass+0x1a8,stop+0x110);q(sklass+0x1b0,0xabcdef)
  q(gd+0x10,0 if opt.get('null')=='mode' else mode);q(mode,klass);q(klass+0x178,stop+0x100)
  q(pc,0 if opt.get('null')=='project' else project);q(project+0x20,0 if opt.get('null')=='data' else data);q(data+0x78,0 if opt.get('null')=='ascension' else asc)
  d(obj+0x78,opt.get('level',0));d(obj+0x7c,opt.get('day',0))
  for k,v in enumerate(opt.get('factors',[])):
   p=arena+0x10000+k*0x100;q(p+0x30,0 if v=='null_info' else p+0x50)
   if isinstance(v,int):d(p+0x64,v)
  # Provider-returned count object is the only permitted object write below.
  before=bytes(uc.mem_read(arena+0x3000,0x1d000))
  sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,obj);uc.reg_write(x.UC_X86_REG_MXCSR,0x1f80)
  uc.emu_start(base+ENTRIES[name][0],stop+0x200,count=10000)
  after=bytes(uc.mem_read(arena+0x3000,0x1d000));skip=count+0x18-(arena+0x3000)
  assert before[:skip]==after[:skip] and before[skip+4:]==after[skip+4:]
  value=None if state['error'] else (reg(x.UC_X86_REG_XMM0)&0xffffffff if name=='GetScoreMultiplayer' else reg(x.UC_X86_REG_RAX)&0xffffffff)
  if name=='GetScore' and not state['error']:value='null' if reg(x.UC_X86_REG_RAX)==0 else 'score';assert reg(x.UC_X86_REG_RAX) in (0,score)
  return {'method':name,'input':dict(options),'value':value,'error':state['error'],'events':state['events'][:],**({'accumulator_bits':reg(x.UC_X86_REG_XMM6)&0xffffffff} if name=='GetScoreMultiplayer' and state['error'] in ('null','move_next','dispose') else {})}
 cases=[]
 def check(name,options,value,error=None,events=None):
  result=run(name,options);assert result['error']==error,(result,error)
  if not error and name!='UpdateScore':assert result['value']==value,(result,value)
  if events is not None:assert result['events']==events,(result,events)
  cases.append(result)
 for cold,null in itertools.product([False,True],[False,True]):
  check('GetScore',{'cold':cold,'null':'score' if null else None},'null' if null else 'score',events=[{'kind':'class_init'}] if cold else [])
 check('GetScore',{'cold':True,'fail':['class_init',1]},None,'class_init')
 for day,cold in itertools.product([-2**31,-1,0,1,2**31-1],[False,True]):
  ev=([{'kind':'class_init'}] if cold else [])+[{'kind':'update_full_score','day':day}]
  check('UpdateScore',{'day':day,'cold':cold},None,events=ev)
  check('UpdateScore',{'day':day,'cold':cold,'fail':['update_full_score',1]},None,'update_full_score',ev)
 check('UpdateScore',{'null':'score'},None,'null',[])
 for n in [-2**31,-1,0,1,7,2**31-1]:check('GetMaxDay',{'sizes':[n]},i32(n-1)&0xffffffff,events=[{'kind':'counts'}])
 for method in ['GetMaxDay','GetCharactersLevel']:
  for missing in ['project','data','ascension']+(['mode'] if method=='GetCharactersLevel' else []):check(method,{'null':missing},None,'null')
  check(method,{'sizes':[None]},None,'null')
  check(method,{'fail':['counts',1]},None,'counts')
 for mode_value,level,first,second in itertools.product([0,1,-1],[-2**31,-1,0,6,7,2**31-1],[-2**31,0,7,2**31-1],[-2**31,5]):
  ev=[{'kind':'mode'},{'kind':'counts'}]
  if mode_value==0:expected=123;ev+=[{'kind':'random','minimum':0,'maximum':first}]
  elif level<first:expected=level
  else:expected=i32(second-1);ev+=[{'kind':'counts'}]
  check('GetCharactersLevel',{'mode':mode_value,'level':level,'sizes':[first,second]},expected&0xffffffff,events=ev)
 for fail in [['mode',1],['counts',2],['random',1]]:
  check('GetCharactersLevel',{'mode':0 if fail[0]=='random' else 1,'level':7,'fail':fail},None,fail[0])
 check('GetCharactersLevel',{'level':7,'sizes':[7,None]},None,'null')
 check('GetCharactersLevel',{'cold':True,'fail':['class_init',1]},None,'class_init')
 factors=[0,0x80000000,0x3f800000,0xbf800000,0x3f800001,0x7f7fffff,1,0x7f800000,0xff800000,0x7fc12345,0x7f812345]
 sequences=[[]]+[[v] for v in factors]+[list(v) for v in itertools.product(factors,repeat=2)]+[[0x7f7fffff,0x40000000,0x3f000000],[0x3f000000,0x40000000,0x7f7fffff]]
 for values in sequences:
  expected=0x3f800000
  for v in values:expected=mul(expected,v)
  check('GetScoreMultiplayer',{'factors':values},expected,events=[{'kind':'enumerator'}]+[{'kind':'move_next'}]*(len(values)+1)+[{'kind':'dispose'}])
 for values in [['null_relic'],[0x40000000,'null_info']]:
  check('GetScoreMultiplayer',{'factors':values},None,'null', [{'kind':'enumerator'}]+[{'kind':'move_next'}]*len(values))
  assert cases[-1]['accumulator_bits']==(0x3f800000 if len(values)==1 else 0x40000000)
 check('GetScoreMultiplayer',{'null':'relics'},None,'null',[])
 for name,index in [('enumerator',1),('move_next',1),('move_next',2),('dispose',1),('class_init',1)]:
  check('GetScoreMultiplayer',{'factors':[0x40000000],'cold':name=='class_init','fail':[name,index]},None,name)
 return {'schema_version':1,'build_id':BUILD,'cases_passed':len(cases),'distinct_native_instructions':len(visited),'metadata_verified':verified,'field_assertions':fields,'cases':cases,'boundary':'Metadata warmed; optional class initialization, enumerator, mode, character-count provider, Random.Range and Score.UpdateFullScore are explicit preserving gateways. Native exception unwinding, provider bodies and UI effects are not executed.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();result=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['cases','metadata_verified']}))
