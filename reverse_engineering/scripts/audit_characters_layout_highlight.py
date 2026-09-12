"""Native Characters layout/highlight loops with explicit engine/UI callbacks."""
import argparse,hashlib,itertools,json,math,re,struct
from pathlib import Path
from audit_character_assets import BUILD
from audit_gameplay_score_resources import mul,f32
ENTRIES={'UpdateCharacterPositions':(0x36e4e0,0x36e700),'HighlightCharacters':(0x36caf0,0x36cc38),'DisableHighlightAll':(0x369d60,0x369ea8)}
def bits(v):return struct.unpack('<I',struct.pack('<f',v))[0]
def divide_size(size):return 0x7f800000 if size==0 else bits(360.0/f32(bits(size)))
def audit(game_root,dumper_root):
 import capstone,pefile,unicorn
 from unicorn import x86_const as x
 assert unicorn.__version__=='2.1.4'
 repo=Path(__file__).parents[1];lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'));ex=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
 def pin(path,digest):
  raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==digest.upper();return raw
 raw=pin(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256']);m=json.loads(pin(Path(dumper_root)/'script.json',ex['outputs']['script_json']['sha256']).decode('utf-8-sig'));dump=pin(Path(dumper_root)/'dump.cs',ex['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
 for name,needle in [('Characters','public List<Character> characters; // 0x20'),('Character','public Transform icon; // 0x20'),('Character','public CardHighlight highlight; // 0x90')]:
  block=re.search(r'public class '+name+r' : MonoBehaviour[^\n]*\n\{(.*?)\n\}',dump,re.S).group(1);assert needle in block
 assert 'private static readonly Vector3 zeroVector; // 0x0' in dump
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
 decoded={};verified=[]
 for name,(start,end) in ENTRIES.items():
  row=next(r for r in m['ScriptMethod'] if r['Name']=='Characters$$'+name);assert row['Address']==start
  args='Characters_o* __this'+(', System_Collections_Generic_List_Character__o* chList' if name=='HighlightCharacters' else '')+', const MethodInfo* method'
  assert row['Signature']==f'void Characters__{name} ({args});';verified.append(row)
  next_entry=min(r['Address'] for r in m['ScriptMethod'] if r['Address']>start);ins=list(cs.disasm(pe.get_data(start,next_entry-start),start))
  while ins[-1].mnemonic=='int3':ins.pop()
  assert ins[-1].address+ins[-1].size==end and all(a.address+a.size==b.address for a,b in zip(ins,ins[1:]));decoded.update({i.address:i for i in ins})
 for name,rva in [('UnityEngine.Transform$$set_localEulerAngles',0x1c92030),('UnityEngine.Transform$$set_eulerAngles',0x1c91ec0),('UnityEngine.Object$$op_Inequality',0x1c82480),('CardHighlight$$ShowHighlight',0x397090),('CardHighlight$$DisableHighlight',0x396f40)]:
  row=next(r for r in m['ScriptMethod'] if r['Name']==name);assert row['Address']==rva;verified.append(row)
 def instruction(a,mn,op):assert a in decoded and (decoded[a].mnemonic,decoded[a].op_str)==(mn,op)
 instruction(0x36e568,'divss','xmm6, xmm0');instruction(0x36e5f7,'mulss','xmm0, xmm6')
 instruction(0x36e626,'call','0x1c92030');instruction(0x36e68f,'call','0x1c91ec0')
 instruction(0x36cbce,'je','0x36cb90');instruction(0x369e3e,'je','0x369e00')
 i=decoded[0x36e560];assert struct.unpack('<f',pe.get_data(i.address+i.size+i.operands[1].mem.disp,4))[0]==360.0
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
 arena,stack,stop=0x200000000,0x300000000,0x400000000;uc.mem_map(arena,0x20000);uc.mem_map(stack,0x10000);uc.mem_map(stop,0x1000)
 def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
 def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
 def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
 def reg(r):return uc.reg_read(r)
 def ret(v=0):
  sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v&0xffffffffffffffff);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
 types={'UnityEngine.Object':arena+0x1000,'UnityEngine.Vector3':arena+0x2000}
 for name,p in types.items():
  row=next(r for r in m['ScriptMetadata'] if r['Name']==name+'_TypeInfo');q(base+row['Address'],p)
 q(types['UnityEngine.Vector3']+0xb8,arena+0x3000)
 for i in decoded.values():
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
 obj,board,selection=arena+0x4000,arena+0x5000,arena+0x6000
 state={};opt={};identities={};visited=set()
 def emit(name,**kw):
  state['events'].append({'kind':name,**kw});state['counts'][name]=state['counts'].get(name,0)+1
  if opt.get('fail')==[name,state['counts'][name]]:state['error']=name;uc.emu_stop();return False
  return True
 def hook(_,a,size,__):
  r=a-base;c=reg(x.UC_X86_REG_RCX);dx=reg(x.UC_X86_REG_RDX)
  if a==stop:uc.emu_stop();return
  if r==0xb16640:
   source='selected' if dx==selection else 'board';assert dx in (board,selection)
   if emit('enumerator',source=source):uc.mem_write(c,bytes(24));q(c,dx);state['source']=source;state['index']=0;ret(c)
  elif r==0x9693d0:
   if emit('move_next'):
    values=opt[state['source']];index=state['index'];state['index']+=1
    if index==len(values):ret(0)
    else:q(c+0x10,0 if values[index] is None else arena+0x8000+values[index]*0x400);ret(1)
  elif r==0x1c7a010:
   target=identities[c]
   if emit('transform',target=target):ret(0 if opt.get('missing_transform')==target else c+0x100)
  elif r in (0x1c92030,0x1c91ec0):
   target=identities[c-0x100];values=list(struct.unpack('<III',uc.mem_read(dx,12)));name='local_euler' if r==0x1c92030 else 'world_euler'
   if emit(name,target=target,bits=values):state['rotations'][target]=values;ret()
  elif r==0x281d90:
   assert c==types['UnityEngine.Object']
   if emit('class_init'):d(c+0xe0,1);ret()
  elif r==0x1c82480:
   assert dx==0
   target=None if c==0 else identities[c];value=opt.get('nonnull_override') if c==0 and 'nonnull_override' in opt else c!=0 and target not in opt.get('destroyed',[])
   if emit('unity_nonnull',target=target,result=bool(value)):ret(0xabc000|int(value))
  elif r in (0x397090,0x396f40):
   target=identities[c];name='show' if r==0x397090 else 'disable'
   if emit(name,target=target):state['highlight_calls'].append([name,target]);ret()
  elif r==0x33ed50:
   if emit('dispose'):ret()
  elif r==0x2b7d90:state['error']='null';uc.emu_stop()
  elif r in decoded:visited.add(r)
  else:raise AssertionError(('unexpected gateway',hex(a)))
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 def run(method,options):
  opt.clear();opt.update(board=[0,1,2],selected=[2,0]);opt.update(options);state.clear();state.update(events=[],counts={},error=None,rotations={},highlight_calls=[]);identities.clear()
  uc.mem_write(arena+0x4000,bytes(0x1c000));d(types['UnityEngine.Object']+0xe0,0 if opt.get('cold') else 1)
  uc.mem_write(arena+0x3000,struct.pack('<III',*opt.get('zero_bits',[0,0,0])));q(obj+0x20,0 if opt.get('null_board') else board);d(board+0x18,opt.get('size',len(opt['board'])))
  for n in range(8):
   p=arena+0x8000+n*0x400;identities[p]=f'character{n}';identities[p+0x180]=f'icon{n}';identities[p+0x300]=f'highlight{n}'
   q(p+0x20,0 if opt.get('null_icon')==n else p+0x180);q(p+0x90,0 if opt.get('null_highlight')==n else p+0x300)
  before=bytes(uc.mem_read(arena+0x4000,0x1c000));sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,obj);uc.reg_write(x.UC_X86_REG_RDX,0 if opt.get('null_selected') else selection);uc.reg_write(x.UC_X86_REG_MXCSR,0x1f80)
  uc.emu_start(base+ENTRIES[method][0],stop+0x200,count=10000)
  assert before==bytes(uc.mem_read(arena+0x4000,0x1c000))
  return {'method':method,'input':dict(opt),'events':state['events'][:],'error':state['error'],'rotations':dict(state['rotations']),'highlight_calls':state['highlight_calls'][:]}
 cases=[]
 def check(method,options):
  result=run(method,options);opt=result['input'];events=[];rotations={};calls=[];counts={};error=None
  class Halt(Exception):pass
  def event(name,**kw):
   events.append({'kind':name,**kw});counts[name]=counts.get(name,0)+1
   if opt.get('fail')==[name,counts[name]]:raise Halt(name)
  def require(v):
   if not v:raise Halt('null')
  try:
   source='selected' if method=='HighlightCharacters' else 'board';require(not opt.get('null_'+source));event('enumerator',source=source)
   cold=opt.get('cold',False);step=divide_size(opt.get('size',len(opt['board'])))
   for index,n in enumerate(opt[source]):
    event('move_next')
    if method=='UpdateCharacterPositions':
     require(n is not None);ch=f'character{n}';icon=f'icon{n}';event('transform',target=ch);require(opt.get('missing_transform')!=ch)
     values=[0,0,mul(bits(index),step)];event('local_euler',target=ch,bits=values);rotations[ch]=values
     require(opt.get('null_icon')!=n);event('transform',target=icon);require(opt.get('missing_transform')!=icon)
     values=opt.get('zero_bits',[0,0,0]);event('world_euler',target=icon,bits=values);rotations[icon]=values
    else:
     if cold:event('class_init');cold=False
     ch=None if n is None else f'character{n}';nonnull=opt.get('nonnull_override',False) if n is None else ch not in opt.get('destroyed',[])
     event('unity_nonnull',target=ch,result=nonnull)
     if not nonnull:continue
     require(n is not None);require(opt.get('null_highlight')!=n);name='show' if method=='HighlightCharacters' else 'disable';event(name,target=f'highlight{n}');calls.append([name,f'highlight{n}'])
   event('move_next');event('dispose')
  except Halt as halt:error=str(halt)
  assert result['error']==error and result['events']==events and result['rotations']==rotations and result['highlight_calls']==calls,(result,error,events,rotations,calls)
  cases.append(result);return result
 for method in ENTRIES:
  for values in [[],[0],[0,1,2],[2,2,0],[0,None,2],[None]]:
   options={'board':values} if method!='HighlightCharacters' else {'selected':values}
   check(method,options)
  check(method,{'null_selected':True} if method=='HighlightCharacters' else {'null_board':True})
  baseline=check(method,{'cold':True});counts={}
  for event in baseline['events']:
   name=event['kind'];counts[name]=counts.get(name,0)+1;check(method,{'cold':True,'fail':[name,counts[name]]})
 for size in [-2**31,-1,0,1,3,7,2**31-1]:check('UpdateCharacterPositions',{'size':size})
 check('UpdateCharacterPositions',{'zero_bits':[0x80000000,0x7fc12345,0x3f800000]})
 for n in [0,1,2]:
  check('UpdateCharacterPositions',{'null_icon':n});check('UpdateCharacterPositions',{'missing_transform':f'character{n}'});check('UpdateCharacterPositions',{'missing_transform':f'icon{n}'})
 for method in ['HighlightCharacters','DisableHighlightAll']:
  for destroyed in [[],['character0'],['character0','character1','character2']]:check(method,{'destroyed':destroyed,'cold':True})
  for n in [0,1,2]:check(method,{'null_highlight':n})
  check(method,{'board':[None],'selected':[None],'nonnull_override':True})
 return {'schema_version':1,'build_id':BUILD,'cases_passed':len(cases),'distinct_native_instructions':len(visited),'metadata_verified':verified,'cases':cases,'boundary':'Three actual Characters callers; explicit stable enumeration, Unity null comparison, class initialization, transform lookup/setters and CardHighlight callbacks. No transform/tween/coroutine engine bodies or exception unwinding. MXCSR0x1F80; malformed size/provider combinations are adversarial fixtures.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();result=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['cases','metadata_verified']}))
