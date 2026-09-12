"""Native CardHighlight caller, iterator and animation-ID boundaries."""
import argparse,hashlib,itertools,json,struct
from pathlib import Path
from audit_character_assets import BUILD

def audit(game_root,dumper_root):
 import capstone,pefile,unicorn
 from unicorn import x86_const as x
 repo=Path(__file__).parents[1];lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'));ext=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'));assert unicorn.__version__=='2.1.4'
 def pin(p,h):
  b=p.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==h.upper();return b
 raw=pin(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256']);meta=json.loads(pin(Path(dumper_root)/'script.json',ext['outputs']['script_json']['sha256']).decode('utf-8-sig'))
 dump=pin(Path(dumper_root)/'dump.cs',ext['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
 entries={'CardHighlight$$Awake':(0x396EB0,0x396F3B),'CardHighlight$$DisableHighlight':(0x396F40,0x39701D),'CardHighlight$$HighlightCoroutine':(0x397020,0x397086),'CardHighlight$$ShowHighlight':(0x397090,0x39714C),'CardHighlight$$.ctor':(0x397150,0x39719E),'CardHighlight.<HighlightCoroutine>d__4$$MoveNext':(0x3A9F50,0x3AA080),'CardHighlight.<HighlightCoroutine>d__4$$System.Collections.IEnumerator.Reset':(0x3AA090,0x3AA0CE)}
 exact=[]
 for name,(start,end) in entries.items():
  rows=[r for r in meta['ScriptMethod'] if r['Name']==name];assert len(rows)==1 and rows[0]['Address']==start;exact+=rows
 expected=['void CardHighlight__Awake (CardHighlight_o* __this, const MethodInfo* method);','void CardHighlight__DisableHighlight (CardHighlight_o* __this, const MethodInfo* method);','System_Collections_IEnumerator_o* CardHighlight__HighlightCoroutine (CardHighlight_o* __this, const MethodInfo* method);','void CardHighlight__ShowHighlight (CardHighlight_o* __this, const MethodInfo* method);','void CardHighlight___ctor (CardHighlight_o* __this, const MethodInfo* method);','bool CardHighlight__HighlightCoroutine_d__4__MoveNext (CardHighlight__HighlightCoroutine_d__4_o* __this, const MethodInfo* method);','void CardHighlight__HighlightCoroutine_d__4__System_Collections_IEnumerator_Reset (CardHighlight__HighlightCoroutine_d__4_o* __this, const MethodInfo* method);']
 assert [r['Signature'] for r in exact]==expected
 block=dump.split('public class CardHighlight ',1)[1].split('// Namespace:',1)[0]
 assert 'public CanvasGroup[] highlights; // 0x20' in block and 'private string highlightAnimation; // 0x28' in block
 gateway_names={'UnityEngine.Component$$get_gameObject':0x1C79FD0,'UnityEngine.Object$$GetInstanceID':0x1C81060,'System.String$$Format':0xF74DF0,'UnityEngine.MonoBehaviour$$StopAllCoroutines':0x1C7F4A0,'UnityEngine.MonoBehaviour$$StartCoroutine':0x1C7F160,'DG.Tweening.DOTween$$Kill':0x5044D0,'DG.Tweening.DOTweenModuleUI$$DOFade':0x349E80,'DG.Tweening.TweenSettingsExtensions$$SetId<object>':0x6BC9D0,'UnityEngine.WaitForSeconds$$.ctor':0x1C961F0}
 verified_gateways=[]
 for n,a in gateway_names.items():
  rows=[r for r in meta['ScriptMethod'] if r['Name']==n and r['Address']==a];assert len(rows)==1,(n,rows);verified_gateways+=rows
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True;decoded={}
 for start,end in entries.values():
  nxt=min(r['Address'] for r in meta['ScriptMethod'] if r['Address']>start);ins=list(cs.disasm(pe.get_data(start,nxt-start),start))
  while ins[-1].mnemonic=='int3':ins.pop()
  assert ins[-1].address+ins[-1].size==end,(hex(start),hex(ins[-1].address+ins[-1].size))
  assert all(a.address+a.size==b.address for a,b in zip(ins,ins[1:]));decoded.update({i.address:i for i in ins})
 decoded.update({i.address:i for i in cs.disasm(pe.get_data(0x33ED50,3),0x33ED50)})
 assert (decoded[0x33ED50].mnemonic,decoded[0x33ED50].op_str)==('ret','0')
 for a,bits in [(0x396FBC,0x3DCCCCCD),(0x3A9FA8,0x3ED70A3D),(0x3AA00E,0x3F000000),(0x3AA016,0x3F800000)]:
  i=decoded[a];r=i.address+i.size+i.operands[1].mem.disp;buf=pe.get_data(r,4);assert len(buf)==4 and struct.unpack('<I',buf)[0]==bits
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image());arena,stack,stop=0x200000000,0x300000000,0x400000000
 uc.mem_map(arena,0x100000);uc.mem_map(stack,0x20000);uc.mem_map(stop,0x1000)
 def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
 def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
 def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
 def rd(a):return struct.unpack('<I',uc.mem_read(a,4))[0]
 def reg(r):return uc.reg_read(r)
 def ret(v=0):
  sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
 # Bind exact named metadata referred to by these bodies, including shared generic tokens.
 refs=set()
 for i in decoded.values():
  for o in i.operands:
   if o.type==capstone.CS_OP_MEM and o.mem.base==capstone.x86.X86_REG_RIP:refs.add(i.address+i.size+o.mem.disp)
 bindings={};rows=meta['ScriptMetadata']+meta['ScriptMetadataMethod']
 for r in rows:
  if r['Address'] in refs:
   p=arena+0x1000+len(bindings)*0x300;bindings[r['Name']]=p;q(base+r['Address'],p);d(p+0xE0,1)
 for r in meta['ScriptString']:
  if r['Address'] in refs:
   p=arena+0x1000+len(bindings)*0x300;bindings['literal:'+r['Value']]=p;q(base+r['Address'],p)
 required=('DG.Tweening.DOTween_TypeInfo','UnityEngine.WaitForSeconds_TypeInfo','CardHighlight.<HighlightCoroutine>d__4_TypeInfo','int_TypeInfo','System.NotSupportedException_TypeInfo','Method$CardHighlight.<HighlightCoroutine>d__4.System.Collections.IEnumerator.Reset()','Method$DG.Tweening.TweenSettingsExtensions.SetId<TweenerCore<float, float, FloatOptions>>()','literal:','literal:{0}_highlight')
 for n in required:assert n in bindings,n
 flags=[]
 for i in decoded.values():
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:flags.append(base+i.address+i.size+i.operands[0].mem.disp)
 owner,array,alternate,iterator,wait,game,boxed,exception= [arena+n for n in (0x10000,0x11000,0x12000,0x13000,0x14000,0x15000,0x16000,0x17000)]
 canvases=[arena+0x20000+n*0x100 for n in range(3)];ids=[arena+0x21000+n*0x100 for n in range(3)];tweens=[arena+0x22000+n*0x100 for n in range(3)]
 labels={0:None,owner:'owner',iterator:'iterator',wait:'wait',game:'game',boxed:'boxed',exception:'exception',bindings['literal:']:'empty',bindings['literal:{0}_highlight']:'format'}|{p:'canvas'+str(i) for i,p in enumerate(canvases)}|{p:'id'+str(i) for i,p in enumerate(ids)}|{p:'tween'+str(i) for i,p in enumerate(tweens)}
 opt={};state={};visited=set()
 def snap():return {'animation':labels[rq(owner+0x28)],'iterator_state':rd(iterator+0x10),'current':labels[rq(iterator+0x18)],'capture':labels[rq(iterator+0x20)]}
 def event(n,**kw):
  state['counts'][n]=state['counts'].get(n,0)+1;state['events'].append({'event':n,**kw,'fields':snap()})
  if opt.get('fail')==[n,state['counts'][n]]:state['error']=n;uc.emu_stop();return False
  mutation=opt.get('mutations',{}).get(n+str(state['counts'][n]),{})
  if 'animation' in mutation:q(owner+0x28,0 if mutation['animation'] is None else ids[mutation['animation']])
  if mutation.get('swap_array'):q(owner+0x20,alternate)
  return True
 def hook(_,a,size,__):
  r=a-base;c,t,m=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
  if a==stop:uc.emu_stop();return
  if r==0x2B7B40:
   assert c-base in refs
   if event('metadata'):ret(rq(c))
  elif r==0x1C79FD0:
   assert c==owner and t==0
   if event('game_object'):ret(0 if opt.get('null_game') else game)
  elif r==0x1C81060:
   assert c==game and t==0
   if event('instance_id'):ret(opt.get('instance_id',-7)&0xffffffff)
  elif r==0x282580:
   assert c==bindings['int_TypeInfo']
   if event('box',value=rd(t)):d(boxed+0x10,rd(t));ret(boxed)
  elif r==0xF74DF0:
   assert c==bindings['literal:{0}_highlight'] and t==boxed and m==0
   if event('format',value=rd(boxed+0x10)):ret(ids[1])
  elif r==0x1C79770:
   assert c==owner and t==0
   if event('base_constructor'):ret()
  elif r==0x1C7F4A0:
   assert c==owner and t==0
   if event('stop_all'):ret()
  elif r==0x281D90:
   assert c==bindings['DG.Tweening.DOTween_TypeInfo']
   if event('class_init'):d(c+0xE0,1);ret()
  elif r==0x5044D0:
   assert m==0
   if event('kill',animation=labels[c],complete=bool(t&0xff)):ret(7)
  elif r==0x2B7D40:
   mapping={bindings['CardHighlight.<HighlightCoroutine>d__4_TypeInfo']:iterator,bindings['UnityEngine.WaitForSeconds_TypeInfo']:wait,bindings['System.NotSupportedException_TypeInfo']:exception};assert c in mapping
   p=mapping[c]
   if event('allocate',kind=labels[p]):uc.mem_write(p,bytes(0x80));q(p,c);ret(p)
  elif r==0x1C961F0:
   assert c==wait and m==0
   value=reg(x.UC_X86_REG_XMM1)&0xffffffff
   if event('wait_constructor',bits=value):d(wait+0x10,value);ret()
  elif r==0x2B6FF0:
   assert rq(c)==t
   if event('barrier',value=labels[t]):ret()
  elif r==0x1C7F160:
   assert c==owner and t==iterator and m==0
   if event('start_coroutine'):ret(arena+0x30000)
  elif r==0x349E80:
   assert reg(x.UC_X86_REG_R9)==0
   if event('fade',canvas=labels[c],alpha_bits=reg(x.UC_X86_REG_XMM1)&0xffffffff,duration_bits=reg(x.UC_X86_REG_XMM2)&0xffffffff):ret(0 if opt.get('null_tween') else tweens[(canvases.index(c) if c else 0)])
  elif r==0x6BC9D0:
   assert m==bindings['Method$DG.Tweening.TweenSettingsExtensions.SetId<TweenerCore<float, float, FloatOptions>>()']
   if event('set_id',tween=labels[c],animation=labels[t]):ret(c)
  elif r==0x111B720:
   assert c==exception and t==0
   if event('exception_constructor'):ret()
  elif r==0x2B7D50:
   assert c==exception and t==bindings['Method$CardHighlight.<HighlightCoroutine>d__4.System.Collections.IEnumerator.Reset()'];event('throw');state['error']='not_supported';uc.emu_stop()
  elif r==0x2B7D90:state['error']='null';uc.emu_stop()
  elif r==0x2B7D80:state['error']='bounds';uc.emu_stop()
  elif r in decoded:visited.add(r)
  else:raise AssertionError(hex(r))
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 cases=[]
 def run(name,options=None):
  opt.clear();opt.update(options or {});state.clear();state.update(error=None,events=[],counts={})
  for f in flags:uc.mem_write(f,b'\0' if opt.get('cold_metadata') else b'\1')
  d(bindings['DG.Tweening.DOTween_TypeInfo']+0xE0,0 if opt.get('cold') else 1)
  for p in (owner,array,alternate,iterator,wait,boxed,exception):uc.mem_write(p,bytes(0x100))
  values=opt.get('canvases',[0,1]);q(owner+0x20,0 if values is None else array);d(array+0x18,len(values or []))
  for i,n in enumerate(values or []):q(array+0x20+i*8,0 if n is None else canvases[n])
  d(alternate+0x18,1);q(alternate+0x20,canvases[2]);q(owner+0x28,0 if opt.get('null_id') else ids[0]);d(iterator+0x10,opt.get('state',0));q(iterator+0x18,ids[2]);q(iterator+0x20,0 if opt.get('null_capture') else owner)
  sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,iterator if name in ('move','reset') else 0 if opt.get('null_receiver') else owner);uc.reg_write(x.UC_X86_REG_RDX,0)
  keep=[x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
  for rr in keep:uc.reg_write(rr,0xABC000+rr)
  for rr in (x.UC_X86_REG_XMM6,x.UC_X86_REG_XMM7):uc.reg_write(rr,0x123456789ABCDEF0011223344556677)
  address={'awake':0x396EB0,'disable':0x396F40,'factory':0x397020,'show':0x397090,'ctor':0x397150,'move':0x3A9F50,'reset':0x3AA090}[name]
  before_owner=bytes(uc.mem_read(owner,0x100));before_array=bytes(uc.mem_read(array,0x100))
  uc.emu_start(base+address,stop,count=10000)
  after_owner=bytes(uc.mem_read(owner,0x100));assert before_owner[:0x20]==after_owner[:0x20] and before_owner[0x30:]==after_owner[0x30:] and before_array==bytes(uc.mem_read(array,0x100))
  if state['error'] is None:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8 and all(reg(rr)==0xABC000+rr for rr in keep) and all(reg(rr)==0x123456789ABCDEF0011223344556677 for rr in (x.UC_X86_REG_XMM6,x.UC_X86_REG_XMM7))
  result={'method':name,'options':dict(opt),'events':state['events'][:],'fields':snap(),'error':state['error'],'move_result':bool(reg(x.UC_X86_REG_RAX)&0xff) if name=='move' and state['error'] is None else None};cases.append(result);return result
 for name in ('awake','ctor','factory','show','disable','move','reset'):
  options={'state':1} if name=='move' else {}
  baseline=run(name,options)
  expected={'awake':['game_object','instance_id','box','format','barrier'],'ctor':['barrier','base_constructor'],'factory':['allocate','barrier'],'show':['stop_all','kill','allocate','barrier','start_coroutine'],'disable':['stop_all','kill','fade','set_id','fade','set_id'],'move':['fade','set_id','fade','set_id'],'reset':['metadata','allocate','exception_constructor','metadata','throw']}[name]
  assert [e['event'] for e in baseline['events']]==expected,(name,baseline)
  if name=='awake':assert baseline['fields']['animation']=='id1' and baseline['events'][2]['value']==0xfffffff9
  if name=='ctor':assert baseline['fields']['animation']=='empty'
  if name in ('factory','show'):assert baseline['fields']['iterator_state']==0 and baseline['fields']['capture']=='owner' and baseline['fields']['current'] is None
  if name in ('disable','move'):
   for e in baseline['events']:
    if e['event']=='fade':assert e['alpha_bits']==(0 if name=='disable' else 0x3F800000) and e['duration_bits']==(0x3DCCCCCD if name=='disable' else 0x3F000000)
  if name=='show':assert baseline['events'][1]['complete'] is True
  if name=='disable':assert baseline['events'][1]['complete'] is False
  counts={}
  for i,e in enumerate(baseline['events']):
   n=e['event'];counts[n]=counts.get(n,0)+1
   if n=='throw':continue
   failed=run(name,{**options,'fail':[n,counts[n]]});assert failed['error']==n and failed['events']==baseline['events'][:i+1] and failed['fields']==e['fields']
 for name in ('awake','ctor','factory','show','disable','move'):
  r=run(name,{'cold_metadata':True,'cold':True});assert r['error'] is None and any(e['event']=='metadata' for e in r['events'])
 for value in (-1,0,1,2,2147483647,-2147483648):
  r=run('move',{'state':value});assert r['error'] is None
  if value==0:
   assert r['move_result'] and r['fields']['iterator_state']==1 and r['fields']['current']=='wait'
   assert r['events'][1]['event']=='wait_constructor' and r['events'][1]['bits']==0x3ED70A3D
  elif value==1:assert not r['move_result'] and r['fields']['iterator_state']==0xffffffff and r['fields']['current']=='id2'
  else:assert not r['move_result'] and r['fields']['iterator_state']==value&0xffffffff and r['fields']['current']=='id2' and not r['events']
 baseline=run('move',{'state':0});counts={}
 for i,e in enumerate(baseline['events']):
  n=e['event'];counts[n]=counts.get(n,0)+1;r=run('move',{'state':0,'fail':[n,counts[n]]});assert r['error']==n and r['fields']==e['fields'] and r['events']==baseline['events'][:i+1]
 for name in ('disable','move'):
  for values in ([],[2,2,0],[None],[0,None,2],None):
   r=run(name,{'state':1,'canvases':values});assert r['error']==('null' if values is None else None)
   assert [e['canvas'] for e in r['events'] if e['event']=='fade']==(['canvas'+str(n) if n is not None else None for n in values] if values is not None else [])
  r=run(name,{'state':1,'null_tween':True,'null_id':True});assert r['error'] is None and all(e['tween'] is None and e['animation'] is None for e in r['events'] if e['event']=='set_id')
  r=run(name,{'state':1,'mutations':{'fade1':{'animation':1,'swap_array':True},'set_id1':{'animation':2}}});assert r['error'] is None
  assert [e['canvas'] for e in r['events'] if e['event']=='fade']==['canvas0','canvas1']
  assert [e['animation'] for e in r['events'] if e['event']=='set_id']==['id1','id2']
 for name in ('show','disable'):
  r=run(name,{'cold':True,'mutations':{'stop_all1':{'animation':1},'class_init1':{'animation':2}}});assert r['error'] is None
  assert next(e['animation'] for e in r['events'] if e['event']=='kill')=='id1'
  r=run(name,{'cold':True,'fail':['class_init',1]});assert r['error']=='class_init' and [e['event'] for e in r['events']]==['stop_all','class_init']
 assert run('awake',{'null_game':True})['error']=='null'
 assert run('factory',{'null_receiver':True})['fields']['capture'] is None
 r=run('move',{'state':0,'null_capture':True});assert r['move_result'] and r['fields']['capture'] is None
 r=run('move',{'state':1,'null_capture':True});assert r['error']=='null' and r['fields']['iterator_state']==0xffffffff and r['fields']['current']=='id2'
 for value in (-1,0,1):
  r=run('reset',{'state':value});assert r['error']=='not_supported' and r['fields']['iterator_state']==value&0xffffffff and r['fields']['current']=='id2'
 return {'schema_version':1,'build_id':BUILD,'exact_declarations':exact,'cases_passed':len(cases),'distinct_native_instructions':len(visited),'metadata_bindings':sorted(bindings),'verified_gateways':verified_gateways,'cases':cases,'scope':'Seven native CardHighlight declarations; timer descriptor only, explicit external MoveNext entry, no scheduler. Engine stop/start, DOTween Kill/DOFade/SetId, object ID/format, allocation and constructors are explicit services. Failure stops preserve exact native write prefixes; no managed unwind or tween effect implementation claimed.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(f"Passed {r['cases_passed']} CardHighlight cases")
