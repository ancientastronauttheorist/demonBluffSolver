"""Native reveal wrappers under an explicit explicit diagnostic formatting and callback gateways."""
import argparse,hashlib,json,re,struct
from pathlib import Path
from audit_character_assets import BUILD
ENTRIES={'RevealAllDebug':(0x36da40,0x36df21),'RevealAll':(0x36df30,0x36e485)}
def audit(game_root,dumper_root):
 import capstone,pefile,unicorn
 from unicorn import x86_const as x
 assert unicorn.__version__=='2.1.4'
 repo=Path(__file__).parents[1];lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'));ex=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
 def pin(path,digest):
  raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==digest.upper();return raw
 raw=pin(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256']);m=json.loads(pin(Path(dumper_root)/'script.json',ex['outputs']['script_json']['sha256']).decode('utf-8-sig'));dump=pin(Path(dumper_root)/'dump.cs',ex['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
 for field in ['public static List<Character> CurrentCharacters; // 0x18','public string characterName; // 0x28','public const ECharacterState Hidden = 5;','public const ECharacterState Dead = 20;','public const ECharacterState Revealed = 30;','public ECharacterState prevState; // 0xE0','public ECharacterState state; // 0xE4','public Action onStateChange; // 0x180']:assert field in dump
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
 decoded={};verified=[]
 for name,(start,end) in ENTRIES.items():
  row=next(r for r in m['ScriptMethod'] if r['Name']=='Characters$$'+name);assert row['Address']==start and row['Signature']==f'void Characters__{name} (Characters_o* __this, const MethodInfo* method);';verified.append(row)
  nxt=min(r['Address'] for r in m['ScriptMethod'] if r['Address']>start);ins=list(cs.disasm(pe.get_data(start,nxt-start),start))
  while ins[-1].mnemonic=='int3':ins.pop()
  assert ins[-1].address+ins[-1].size==end and all(a.address+a.size==b.address for a,b in zip(ins,ins[1:]));decoded.update({i.address:i for i in ins})
 for name,rva in [('Character$$RevealAllReal',0x367e80),('UnityEngine.Debug$$Log',0x1c4b450)]:
  row=next(r for r in m['ScriptMethod'] if r['Name']==name);assert row['Address']==rva;verified.append(row)
 def instruction(a,mn,op):assert a in decoded and (decoded[a].mnemonic,decoded[a].op_str)==(mn,op),(hex(a),decoded[a].mnemonic,decoded[a].op_str)
 for a,mn,op in [(0x36e3ad,'mov','dword ptr [rdi + 0xe0], eax'),(0x36e3b3,'mov','dword ptr [rdi + 0xe4], 0x1e'),(0x36e3d5,'call','rax'),(0x36e3dc,'call','0x367e80'),(0x36e3ed,'mov','dword ptr [rdi + 0xe0], eax'),(0x36e3f3,'mov','dword ptr [rdi + 0xe4], ecx'),(0x36e415,'call','rax'),(0x36deb1,'call','0x367e80')]:instruction(a,mn,op)
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
 arena,stack,stop=0x200000000,0x300000000,0x400000000;uc.mem_map(arena,0x20000);uc.mem_map(stack,0x10000);uc.mem_map(stop,0x1000)
 def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
 def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
 def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
 def rd(a):return struct.unpack('<I',uc.mem_read(a,4))[0]
 def reg(r):return uc.reg_read(r)
 def ret(v=0):
  sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
 types={'Gameplay':arena+0x1000,'UnityEngine.Debug':arena+0x2000,'UnityEngine.Object':arena+0x2400,'System.Int32':arena+0x2800}
 for name,p in types.items():
  row=next(r for r in m['ScriptMetadata'] if r['Name']==('int' if name=='System.Int32' else name)+'_TypeInfo');q(base+row['Address'],p)
 q(types['Gameplay']+0xb8,arena+0x3000)
 for i in decoded.values():
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
 obj,globalboard,localboard=arena+0x4000,arena+0x5000,arena+0x6000
 strings={};literal_ptrs={};literal_expected={0x26d4028:"act: '",0x26f34a0:"'",0x270d1f0:';; \n',0x26fde70:', H',0x26fe9f0:', realRole: {0}; ',0x26edab8:'{0}: {1}',0x26df1b8:'',0x26fddf0:', D',0x270ccf0:'; '}
 for addr,value in literal_expected.items():
  assert next(r['Value'] for r in m['ScriptString'] if r['Address']==addr)==value
  pointer=0x500000000+len(literal_ptrs)*0x100;literal_ptrs[pointer]=value;q(base+addr,pointer)
 for rva,name in [(0x35de70,'Acted$$GetActed'),(0xf750a0,'System.String$$Format'),(0xf74df0,'System.String$$Format'),(0xf713f0,'System.String$$Concat'),(0xf71c60,'System.String$$Concat'),(0xf7bab0,'System.String$$op_Inequality'),(0x1c822c0,'UnityEngine.Object$$op_Equality'),(0x1c82480,'UnityEngine.Object$$op_Inequality')]:
  row=next(r for r in m['ScriptMethod'] if r['Address']==rva and r['Name']==name);verified.append(row)
 for field in ['public CharacterData dataRef; // 0x50','public CharacterData bluff; // 0x58','public Acted acteds; // 0xA8','public bool revealed; // 0xD8','public int id; // 0x118']:assert field in dump
 opt={};state={};visited=set()
 def string(value):
  p=0x600000000+len(strings)*0x100;strings[p]=value;return p
 def text(p):return '' if p==0 else strings[p]
 def roleptr(n,bluff=False):return arena+0x10000+n*0x400+(0x100 if bluff else 0)
 def roleidentity(p):
  if p==0:return None
  for n in range(3):
   for bluff in [False,True]:
    if p==roleptr(n,bluff):return ('bluff' if bluff else 'real')+str(n)
  raise AssertionError(hex(p))
 def char(n):return arena+0x8000+n*0x400
 def snap():return [[rd(char(n)+0xe0),rd(char(n)+0xe4),0 if rq(char(n)+0x180)==0 else (rq(char(n)+0x180)-char(n)-0x200)//0x80+1] for n in range(3)]
 def emit(kind,**kw):
  state['events'].append({'kind':kind,**kw,'state':snap()});state['counts'][kind]=state['counts'].get(kind,0)+1
  if opt.get('fail')==[kind,state['counts'][kind]]:state['error']=kind;uc.emu_stop();return False
  return True
 def mutate(n,kind):
  change=opt.get('mutations',{}).get(kind+str(state['counts'][kind]))
  if change:
   if 'previous' in change:d(char(n)+0xe0,change['previous'])
   if 'current' in change:d(char(n)+0xe4,change['current'])
   if 'delegate' in change:q(char(n)+0x180,0 if change['delegate']==0 else char(n)+0x200+(change['delegate']-1)*0x80)
 def hook(_,a,size,__):
  r=a-base;c=reg(x.UC_X86_REG_RCX);dx=reg(x.UC_X86_REG_RDX)
  if a==stop:uc.emu_stop();return
  if r==0xb16640:
   source='global' if dx==globalboard else 'local';assert dx in (globalboard,localboard)
   if emit('enumerator',source=source):uc.mem_write(c,bytes(24));q(c,dx);state['source']=source;state['index']=0;ret(c)
  elif r==0x9693d0:
   if emit('move_next',source=state['source']):
    values=opt['global'] if state['source']=='global' else opt['local'];idx=state['index'];state['index']+=1
    if idx==len(values):ret(0)
    else:q(c+0x10,0 if values[idx] is None else char(values[idx]));ret(1)
  elif r==0x281d90:
   name=next(k for k,v in types.items() if v==c)
   if emit('class_init',type=name):d(c+0xe0,1);ret()
  elif r==0x1c4b450:
   if emit('log',text=text(c)):ret()
  elif r==0x282580:
   assert c==types['System.Int32'];value=struct.unpack('<i',uc.mem_read(dx,4))[0]
   if emit('box',value=value):ret(string(str(value)))
  elif r in [0x1c822c0,0x1c82480]:
   assert dx==0;identity=roleidentity(c);nonnull=identity is not None and identity not in opt.get('destroyed',[]);result=not nonnull if r==0x1c822c0 else nonnull
   if emit('unity_equal' if r==0x1c822c0 else 'unity_nonnull',role=identity,result=result):ret(0xabc000|int(result))
  elif r in [0xf750a0,0xf74df0]:
   template=text(c);args=[text(dx),text(reg(x.UC_X86_REG_R8))] if r==0xf750a0 else [roleidentity(dx) or ''];value=template.format(*args)
   if emit('format',template=template,args=args):ret(string(value))
  elif r in [0xf71c60,0xf713f0]:
   args=[text(c),text(dx)]+([text(reg(x.UC_X86_REG_R8)),text(reg(x.UC_X86_REG_R9))] if r==0xf713f0 else [])
   if emit('concat',args=args):ret(string(''.join(args)))
  elif r==0x35de70:
   n=(c-arena-0x11000)//0x100;assert n in range(3);index=state['acted_calls'].get(n,0);state['acted_calls'][n]=index+1;values=opt['acted'][n];value=values[min(index,len(values)-1)]
   if emit('get_acted',character=n,value=value):ret(0 if value is None else string(value))
  elif r==0xf7bab0:
   # .NET string null and empty are distinct for equality, although Concat treats null as empty.
   value=(None if c==0 else text(c))!=(None if dx==0 else text(dx))
   if emit('string_unequal',a=None if c==0 else text(c),b=None if dx==0 else text(dx),result=value):ret(int(value))
  elif r==0x367e80:
   n=(c-char(0))//0x400;assert n in range(3) and dx==0
   if emit('reveal_all_real',character=n):mutate(n,'reveal_all_real');ret()
  elif a in [stop+0x100,stop+0x110]:
   n=(c-char(0))//0x400;delegate=(a-stop-0x100)//0x10+1;assert dx==0xabcdef00+delegate
   if emit('callback',character=n,delegate=delegate):mutate(n,'callback');ret()
  elif r==0x33ed50:
   if emit('dispose',source=state['source']):ret()
  elif r==0x2b7d90:state['error']='null';uc.emu_stop()
  elif r in decoded:visited.add(r)
  else:raise AssertionError(('unexpected gateway',hex(a)))
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 def run(method,options):
  opt.clear();opt.update(global_=[],acted=[[''],[''],['']],local=[0,1,2],initial=[[99,5,1],[88,20,0],[77,40,2]]);opt.update(options);opt.setdefault('global',opt.pop('global_'));state.clear();state.update(events=[],counts={},acted_calls={},error=None);strings.clear();strings.update(literal_ptrs)
  uc.mem_write(arena+0x4000,bytes(0x1c000));q(arena+0x3018,0 if opt.get('null_global') else globalboard);q(obj+0x20,0 if opt.get('null_local') else localboard)
  for p in types.values():d(p+0xe0,0 if opt.get('cold') else 1)
  for n in range(3):
   p=char(n);q(p+0x50,0 if opt.get('null_real')==n else roleptr(n));q(p+0x58,roleptr(n,True) if n in opt.get('bluffs',[]) else 0);q(p+0xa8,0 if opt.get('null_acted')==n else arena+0x11000+n*0x100);uc.mem_write(p+0xd8,bytes([opt.get('revealed',[0,0,0])[n]]));d(p+0x118,opt.get('ids',[1,2,3])[n])
   q(roleptr(n)+0x28,0 if opt.get('null_name')==n else string('realName'+str(n)));q(roleptr(n,True)+0x28,string('bluffName'+str(n)))
  for n,(prev,current,delegate) in enumerate(opt['initial']):
   p=char(n);d(p+0xe0,prev);d(p+0xe4,current);q(p+0x180,0 if delegate==0 else p+0x200+(delegate-1)*0x80)
   for k in [1,2]:t=p+0x200+(k-1)*0x80;q(t+0x18,stop+0x100+(k-1)*0x10);q(t+0x28,0xabcdef00+k);q(t+0x40,p)
  before=bytearray(uc.mem_read(arena+0x4000,0x1c000));sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,obj);uc.reg_write(x.UC_X86_REG_RDX,0);uc.reg_write(x.UC_X86_REG_MXCSR,0x1f80)
  preserved={r:0x55660000+idx*0x100 for idx,r in enumerate([x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15])}
  for r,v in preserved.items():uc.reg_write(r,v)
  uc.emu_start(base+ENTRIES[method][0],stop+0x200,count=10000)
  if state['error'] is None:
   assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8,'did not return normally'
   assert all(reg(r)==v for r,v in preserved.items()),'nonvolatile register changed'
  after=bytearray(uc.mem_read(arena+0x4000,0x1c000))
  for n in range(3):
   for off,length in [(0xe0,8),(0x180,8)]:index=char(n)+off-arena-0x4000;before[index:index+length]=after[index:index+length]
  assert before==after,'unexpected managed state write'
  return {'method':method,'input':dict(opt),'events':state['events'][:],'error':state['error'],'final':snap()}
 cases=[]
 def check(method,options):
  actual=run(method,options);o=actual['input'];fields=[row[:] for row in o['initial']];events=[];counts={};error=None
  class Halt(Exception):pass
  def event(kind,**kw):
   events.append({'kind':kind,**kw,'state':[row[:] for row in fields]});counts[kind]=counts.get(kind,0)+1
   if o.get('fail')==[kind,counts[kind]]:raise Halt(kind)
  def mutation(n,kind):
   change=o.get('mutations',{}).get(kind+str(counts[kind]),{})
   for key,index in [('previous',0),('current',1),('delegate',2)]:
    if key in change:fields[n][index]=change[key]&0xffffffff
  def require(v):
   if not v:raise Halt('null')
  try:
   accumulated='';acted_calls={};object_cold=o.get('cold',False)
   def concat(*args):
    event('concat',args=list(args));return ''.join(args)
   if o.get('cold'):event('class_init',type='Gameplay')
   require(not o.get('null_global'));event('enumerator',source='global')
   for n in o['global']:
    event('move_next',source='global');require(n is not None);identifier=o.get('ids',[1,2,3])[n];identifier=(identifier+2**31)%2**32-2**31;event('box',value=identifier)
    bluff='bluff'+str(n) if n in o.get('bluffs',[]) else None;nonnull=bluff is not None and bluff not in o.get('destroyed',[]);use_real=fields[n][1] in [20,30] or o.get('revealed',[0,0,0])[n]!=0
    if not use_real:
     if object_cold:event('class_init',type='UnityEngine.Object');object_cold=False
     event('unity_equal',role=bluff,result=not nonnull);use_real=not nonnull
    if use_real:require(o.get('null_real')!=n)
    name=('' if o.get('null_name')==n else 'realName'+str(n)) if use_real else 'bluffName'+str(n)
    event('format',template='{0}: {1}',args=[str(identifier),name]);accumulated=concat(accumulated,str(identifier)+': '+name)
    if fields[n][1]==20:accumulated=concat(accumulated,', D')
    if fields[n][1]==5:accumulated=concat(accumulated,', H')
    if object_cold:event('class_init',type='UnityEngine.Object');object_cold=False
    event('unity_nonnull',role=bluff,result=nonnull)
    if nonnull:
     role='' if o.get('null_real')==n else 'real'+str(n);event('format',template=', realRole: {0}; ',args=[role]);accumulated=concat(accumulated,', realRole: '+role+'; ')
    else:accumulated=concat(accumulated,'; ')
    require(o.get('null_acted')!=n);idx=acted_calls.get(n,0);acted_calls[n]=idx+1;values=o['acted'][n];value=values[min(idx,len(values)-1)];event('get_acted',character=n,value=value);event('string_unequal',a=value,b='',result=value!='')
    if value!='':
     idx=acted_calls[n];acted_calls[n]=idx+1;value=values[min(idx,len(values)-1)];event('get_acted',character=n,value=value);accumulated=concat(accumulated,"act: '",value or '',"'")
    accumulated=concat(accumulated,';; \n')
   event('move_next',source='global');event('dispose',source='global')
   if o.get('cold'):event('class_init',type='UnityEngine.Debug')
   event('log',text=accumulated);require(not o.get('null_local'));event('enumerator',source='local')
   for n in o['local']:
    event('move_next',source='local');require(n is not None)
    if method=='RevealAll':
     fields[n][0]=fields[n][1];fields[n][1]=30
     if fields[n][2]:event('callback',character=n,delegate=fields[n][2]);mutation(n,'callback')
    event('reveal_all_real',character=n);mutation(n,'reveal_all_real')
    if method=='RevealAll':
     fields[n][0],fields[n][1]=fields[n][1],fields[n][0]
     if fields[n][2]:event('callback',character=n,delegate=fields[n][2]);mutation(n,'callback')
   event('move_next',source='local');event('dispose',source='local')
  except Halt as h:error=str(h)
  assert (actual['events'],actual['error'],actual['final'])==(events,error,fields),(actual,events,error,fields)
  cases.append(actual);return actual
 for method in ENTRIES:
  for values in [[],[0],[0,1,2],[2,2,0],[0,None,2],[None]]:check(method,{'local':values})
  for flag in ['null_global','null_local']:check(method,{flag:True,'cold':True})
  for value in [0,5,20,30,40,0x7fffffff,0x80000000,0xffffffff]:
   for delegate in [0,1,2]:check(method,{'initial':[[123,value,delegate]]*3})
  baseline=check(method,{'cold':True});counts={}
  for e in baseline['events']:
   k=e['kind'];counts[k]=counts.get(k,0)+1;check(method,{'cold':True,'fail':[k,counts[k]]})
  for kind in ['callback','reveal_all_real']:
   for change in [{'previous':0x80000000,'current':0xffffffff},{'delegate':0},{'delegate':2},{'previous':71,'current':72,'delegate':2}]:
    o={'local':[0,0],'mutations':{kind+'1':change}};result=check(method,o);counts={}
    for e in result['events']:
     k=e['kind'];counts[k]=counts.get(k,0)+1
     if k in ['callback','reveal_all_real']:check(method,{**o,'fail':[k,counts[k]]})
 for method in ENTRIES:
  for sequence in [[0],[0,1,2],[2,2,0],[0,None,2],[None]]:check(method,{'global':sequence})
  for value in [0,5,20,30,40,0xffffffff]:
   for revealed in [0,1,255]:
    for bluff in [False,True]:check(method,{'global':[0],'initial':[[123,value,1]]*3,'revealed':[revealed]*3,'bluffs':[0] if bluff else [],'cold':True})
  for extra in [{'destroyed':['bluff0'],'bluffs':[0]},{'null_real':0},{'null_real':0,'bluffs':[0]},{'null_name':0},{'null_acted':1},{'ids':[-2**31,0,2**31-1]},{'acted':[['first','second'],[''],['third']]},{'acted':[[None],[''],['']]},{'acted':[['first',None],[''],['']]}]:check(method,{'global':[0,1,2],**extra})
  opts={'global':[0,1,2],'bluffs':[0,1],'acted':[['first','second'],[''],['third']],'cold':True};result=check(method,opts);counts={}
  for e in result['events']:
   kind=e['kind'];counts[kind]=counts.get(kind,0)+1;check(method,{**opts,'fail':[kind,counts[kind]]})
 return {'schema_version':1,'build_id':BUILD,'cases_passed':len(cases),'distinct_native_instructions':len(visited),'complete_entry_instruction_count':len(decoded),'literal_assertions':{hex(k):v for k,v in literal_expected.items()},'metadata_verified':verified,'cases':cases,'boundary':'Actual complete native entries, warmed metadata, independent Gameplay.CurrentCharacters diagnostic and Characters.characters reveal lists. Explicit enumeration/disposal, class initialization, Debug.Log, Character.RevealAllReal and delegate bodies; injected failures stop at gateway entry without exception unwinding. Callback/reveal mutations explicitly declared. Actual nonempty diagnostic traversal with explicit boxing, synthetic-role formatting, concat, Unity equality and Acted.GetActed gateways; framework/string/role ToString bodies are not executed.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();result=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['cases','metadata_verified']}))
