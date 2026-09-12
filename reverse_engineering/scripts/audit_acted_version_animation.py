"""Actual ActedVersion Show/Animate callers with engine and tween gateways."""
import argparse,hashlib,json,math,re,struct
from pathlib import Path
from audit_character_assets import BUILD
from audit_gameplay_score_resources import f32
ENTRIES={'Animate':(0x35d690,0x35d858),'Show':(0x35d920,0x35dbec)}
def bits(v):return struct.unpack('<I',struct.pack('<f',v))[0]
def audit(game_root,dumper_root):
 import capstone,pefile,unicorn
 from unicorn import x86_const as x
 assert unicorn.__version__=='2.1.4'
 repo=Path(__file__).parents[1];lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'));ex=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
 def pin(path,digest):
  raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==digest.upper();return raw
 raw=pin(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256']);m=json.loads(pin(Path(dumper_root)/'script.json',ex['outputs']['script_json']['sha256']).decode('utf-8-sig'));dump=pin(Path(dumper_root)/'dump.cs',ex['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
 for field in ['public const Ease OutBack = 27;','public TextMeshProUGUI blankText; // 0x28','public TextMeshProUGUI text; // 0x30','private string animationId; // 0x38','private Vector3 savedScale; // 0x40']:assert field in dump
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True;decoded={};verified=[]
 for name,(start,end) in ENTRIES.items():
  row=next(r for r in m['ScriptMethod'] if r['Name']=='ActedVersion$$'+name);assert row['Address']==start and row['Signature']==f"void ActedVersion__{name} (ActedVersion_o* __this, "+('System_String_o* description, ' if name=='Show' else '')+'const MethodInfo* method);';verified.append(row)
  nxt=min(r['Address'] for r in m['ScriptMethod'] if r['Address']>start);ins=list(cs.disasm(pe.get_data(start,nxt-start),start))
  while ins[-1].mnemonic=='int3':ins.pop()
  assert ins[-1].address+ins[-1].size==end and all(a.address+a.size==b.address for a,b in zip(ins,ins[1:]));decoded.update({i.address:i for i in ins})
 for addr,name in [(0x513bf0,'DG.Tweening.ShortcutExtensions$$DOScale'),(0x3562e0,'DG.Tweening.ShortcutExtensionsTMPText$$DOText'),(0x6bc7a0,'DG.Tweening.TweenSettingsExtensions$$SetEase<object>'),(0x6bc9d0,'DG.Tweening.TweenSettingsExtensions$$SetId<object>'),(0x5044d0,'DG.Tweening.DOTween$$Kill'),(0x1c91990,'UnityEngine.Transform$$get_localScale'),(0x1c92270,'UnityEngine.Transform$$set_localScale'),(0x1c79fd0,'UnityEngine.Component$$get_gameObject'),(0x1c7d810,'UnityEngine.GameObject$$SetActive'),(0x1be7620,'TMPro.TMP_Text$$set_text')]:
  row=next(r for r in m['ScriptMethod'] if r['Name']==name and r['Address']==addr);verified.append(row)
 def constant(a):
  i=decoded[a];o=i.operands[1];return struct.unpack('<I',pe.get_data(i.address+i.size+o.mem.disp,4))[0]
 assert constant(0x35d73a)==constant(0x35dac9)==0x2edbe6fe
 assert constant(0x35d808)==constant(0x35d9f3)==constant(0x35db97)==0x3e4ccccd
 assert (decoded[0x35d759].mnemonic,decoded[0x35d759].op_str)==('jbe','0x35d78d') and (decoded[0x35dae8].mnemonic,decoded[0x35dae8].op_str)==('jbe','0x35db1c')
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
 arena,stack,stop=0x200000000,0x300000000,0x400000000;uc.mem_map(arena,0x20000);uc.mem_map(stack,0x10000);uc.mem_map(stop,0x1000)
 def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
 def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
 def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
 def reg(r):return uc.reg_read(r)
 def vec(a):return list(struct.unpack('<III',uc.mem_read(a,12)))
 def putvec(a,v):uc.mem_write(a,struct.pack('<III',*v))
 def ret(v=0):
  sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
 types={'UnityEngine.Vector3':arena+0x1000,'DG.Tweening.DOTween':arena+0x2000}
 for name,p in types.items():row=next(r for r in m['ScriptMetadata'] if r['Name']==name+'_TypeInfo');q(base+row['Address'],p)
 generic_specs={41008256:'Method$DG.Tweening.TweenSettingsExtensions.SetEase<TweenerCore<Vector3, Vector3, VectorOptions>>()',41009344:'Method$DG.Tweening.TweenSettingsExtensions.SetId<TweenerCore<string, string, StringOptions>>()',41009616:'Method$DG.Tweening.TweenSettingsExtensions.SetId<TweenerCore<Vector3, Vector3, VectorOptions>>()'}
 generic={}
 for addr,name in generic_specs.items():
  row=next(r for r in m['ScriptMetadataMethod'] if r['Address']==addr);assert row['Name']==name;generic[addr]=arena+0xb000+len(generic)*0x100;q(base+addr,generic[addr])
 q(types['UnityEngine.Vector3']+0xb8,arena+0x3000)
 for i in decoded.values():
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
 empty=next(r for r in m['ScriptString'] if r['Address']==0x35d9df+7+0x23817d2);assert empty['Value']=='';q(base+empty['Address'],arena+0xf100)
 obj,transform,game,blank,txt,klass=arena+0x4000,arena+0x5000,arena+0x6000,arena+0x7000,arena+0x8000,arena+0x9000
 q(klass+0x558,stop+0x100);q(klass+0x560,0xabcdef);state={};opt={};visited=set()
 def emit(kind,**kw):
  state['events'].append({'kind':kind,**kw,'saved':vec(obj+0x40)});state['counts'][kind]=state['counts'].get(kind,0)+1
  if opt.get('fail')==[kind,state['counts'][kind]]:state['error']=kind;uc.emu_stop();return False
  return True
 def hook(_,a,size,__):
  r=a-base;c=reg(x.UC_X86_REG_RCX);dx=reg(x.UC_X86_REG_RDX)
  if a==stop:uc.emu_stop();return
  if r==0x281d90:
   assert c==types['DG.Tweening.DOTween']
   if emit('class_init'):d(c+0xe0,1);ret()
  elif r==0x5044d0:
   assert c==rq(obj+0x38) and dx&255==1
   if emit('kill',complete=True):ret()
  elif r==0x1c79fd0:
   assert c==obj
   if emit('game_object'):ret(0 if opt.get('null_game') else game)
  elif r==0x1c7d810:
   assert c==game and dx&255==1
   if emit('activate'):ret()
  elif a==stop+0x100:
   assert reg(x.UC_X86_REG_R8)==0xabcdef;target='blank' if c==blank else 'text';assert c in [blank,txt]
   value=None if dx==0 else '' if dx==arena+0xf100 else 'description';assert dx in [0,arena+0xf100,arena+0xf200]
   if emit('set_text',target=target,value=value):state[target]=value;ret()
  elif r==0x3562e0:
   assert c==txt and reg(x.UC_X86_REG_R9)&255==1 and reg(x.UC_X86_REG_XMM2)&0xffffffff==0x3e4ccccd
   sp=reg(x.UC_X86_REG_RSP);assert struct.unpack('<I',uc.mem_read(sp+0x28,4))[0]==0 and rq(sp+0x30)==0 and rq(sp+0x38)==0
   if emit('do_text',description=None if dx==0 else 'description',duration_bits=0x3e4ccccd):ret(0 if opt.get('null_tween')=='text' else arena+0xa000)
  elif r==0x1c7a010:
   assert c==obj
   if emit('transform'):ret(0 if opt.get('null_transform')==state['counts']['transform'] else transform)
  elif r==0x1c91990:
   assert dx==transform
   if emit('get_scale'):
    result=arena+0xc000 if opt.get('alternate_return') else c;putvec(result,opt['current']);ret(result)
  elif r==0x1c92270:
   assert c==transform
   if emit('set_scale',bits=vec(dx)):state['scale']=vec(dx);ret()
  elif r==0x513bf0:
   assert c in [0,transform] and reg(x.UC_X86_REG_XMM2)&0xffffffff==0x3e4ccccd
   if emit('do_scale',null_target=c==0,bits=vec(dx),duration_bits=0x3e4ccccd):ret(0 if opt.get('null_tween')=='scale' else arena+0xa100)
  elif r==0x6bc7a0:
   assert dx==27 and reg(x.UC_X86_REG_R8)==generic[41008256]
   if emit('ease',null_tween=c==0,value=27):ret(c)
  elif r==0x6bc9d0:
   assert dx==rq(obj+0x38) and reg(x.UC_X86_REG_R8)==generic[41009616 if state['counts'].get('do_scale',0) else 41009344]
   if emit('set_id',null_tween=c==0):ret(c)
  elif r==0x2b7d90:state['error']='null';uc.emu_stop()
  elif r in decoded:visited.add(r)
  else:raise AssertionError(hex(a))
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 def run(method,options):
  opt.clear();opt.update(saved=[0,0,0],current=[0x3f800000]*3,zero=[0,0,0]);opt.update(options);state.clear();state.update(events=[],counts={},error=None,scale=opt['current'][:],blank='old_blank',text='old_text')
  uc.mem_write(obj,bytes(0x100));q(obj+0x28,0 if opt.get('null_blank') else blank);q(obj+0x30,0 if opt.get('null_text') else txt);q(obj+0x38,0 if opt.get('null_id') else arena+0xf300);putvec(obj+0x40,opt['saved']);putvec(arena+0x3000,opt['zero']);q(blank,klass);q(txt,klass);d(types['DG.Tweening.DOTween']+0xe0,0 if opt.get('cold') else 1)
  before=bytearray(uc.mem_read(obj,0x100));sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,obj);uc.reg_write(x.UC_X86_REG_RDX,0 if opt.get('null_description') else arena+0xf200);uc.reg_write(x.UC_X86_REG_MXCSR,0x1f80)
  regs={rr:0x123400+rr for rr in [x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RDI,x.UC_X86_REG_RSI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]}
  for rr,v in regs.items():uc.reg_write(rr,v)
  uc.emu_start(base+ENTRIES[method][0],stop+0x200,count=10000)
  if state['error'] is None:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8 and all(reg(rr)==v for rr,v in regs.items())
  after=bytearray(uc.mem_read(obj,0x100));before[0x40:0x4c]=after[0x40:0x4c];assert before==after
  return {'method':method,'input':dict(opt),'events':state['events'][:],'error':state['error'],'saved':vec(obj+0x40),'scale':state['scale'][:],'blank':state['blank'],'text':state['text']}
 def rounded(v):
  try:return f32(bits(v))
  except OverflowError:return math.copysign(math.inf,v)
 cases=[]
 def check(method,options):
  a=run(method,options);o=a['input'];saved=o['saved'][:];scale=o['current'][:];blankvalue='old_blank';textvalue='old_text';events=[];counts={};error=None
  class Halt(Exception):pass
  def event(kind,**kw):
   events.append({'kind':kind,**kw,'saved':saved[:]});counts[kind]=counts.get(kind,0)+1
   if o.get('fail')==[kind,counts[kind]]:raise Halt(kind)
  def require(v):
   if not v:raise Halt('null')
  def trans():event('transform');return o.get('null_transform')!=counts['transform']
  try:
   if method=='Show':
    if o.get('cold'):event('class_init')
    event('kill',complete=True);event('game_object');require(not o.get('null_game'));event('activate');require(not o.get('null_blank'));value=None if o.get('null_description') else 'description';event('set_text',target='blank',value=value);blankvalue=value;require(not o.get('null_text'));event('set_text',target='text',value='');textvalue='';event('do_text',description=value,duration_bits=0x3e4ccccd);event('set_id',null_tween=o.get('null_tween')=='text')
   diff=[rounded(f32(s)-f32(z)) for s,z in zip(saved,o['zero'])];square=[rounded(v*v) for v in diff];distance=rounded(rounded(square[1]+square[0])+square[2])
   if distance<f32(0x2edbe6fe):require(trans());event('get_scale');saved=o['current'][:]
   require(trans());event('set_scale',bits=o['zero']);scale=o['zero'][:];nonnull=trans();event('do_scale',null_target=not nonnull,bits=saved[:],duration_bits=0x3e4ccccd);event('ease',null_tween=o.get('null_tween')=='scale',value=27);event('set_id',null_tween=o.get('null_tween')=='scale')
  except Halt as e:error=str(e)
  assert (a['events'],a['error'],a['saved'],a['scale'],a['blank'],a['text'])==(events,error,saved,scale,blankvalue,textvalue),(a,events,error,saved,scale)
  cases.append(a);return a
 for method in ENTRIES:
  for saved in [[0,0,0],[0x80000000]*3,[0x3f800000]*3,[0x7fc12345,0,0],[0x7f800000,0,0],[1,0,0],[bits(0.000009),0,0],[bits(0.000011),0,0],[bits(0.00001)-1,0,0],[bits(0.00001),0,0],[bits(0.00001)+1,0,0]]:check(method,{'saved':saved})
  check(method,{'zero':[0x3f800000]*3,'saved':[0x3f800000]*3,'current':[0x80000000,0x7fc12345,0x7f800000]})
  check(method,{'alternate_return':True});baseline=check(method,{'cold':True});counts={}
  for e in baseline['events']:k=e['kind'];counts[k]=counts.get(k,0)+1;check(method,{'cold':True,'fail':[k,counts[k]]})
  for n in [1,2,3]:check(method,{'null_transform':n})
  for key in ['null_game','null_blank','null_text','null_description','null_id']:check(method,{key:True})
  for tween in ['text','scale']:check(method,{'null_tween':tween})
 return {'build_id':BUILD,'cases_passed':len(cases),'native_instructions_executed':len(visited),'metadata_verified':verified,'generic_method_bindings':{str(k):v for k,v in generic_specs.items()},'cases':cases,'scope':'Actual Show and Animate complete native bodies; initialized metadata, explicit TMP setter, object/transform, DOTween and type initialization gateways. Raw float bits and MXCSR0x1F80; no tween engine, elapsed time, callback execution or managed unwind claim.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(r['cases_passed'],r['native_instructions_executed'])
