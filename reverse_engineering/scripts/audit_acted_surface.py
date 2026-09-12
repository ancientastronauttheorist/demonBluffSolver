"""Native immediate Acted surface and exact color return-buffer ordering."""
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
 entries={'Acted$$GetActed':(0x35DE70,0x35DEF5),'Acted$$UpdateActed':(0x35E010,0x35E0F3),'Acted$$Act':(0x35DD10,0x35DDB4),'Acted$$Hide':(0x35DF00,0x35DF3C),'Acted$$Highlight':(0x35DF40,0x35DFBA),'Acted$$UnHighlight':(0x35DFC0,0x35E00E),'Acted$$.ctor':(0x33E820,0x33E827)}
 exact=[]
 for name,(a,b) in entries.items():
  rows=[r for r in meta['ScriptMethod'] if r['Name']==name and r['Address']==a];assert len(rows)==1;exact+=rows
  managed=name.split('$$')[1];ret='System_String_o*' if managed=='GetActed' else 'void';symbol='Acted___ctor' if managed=='.ctor' else 'Acted__'+managed;extra=', System_String_o* description' if managed in ('Act','UpdateActed') else ''
  assert rows[0]['Signature']==f'{ret} {symbol} (Acted_o* __this{extra}, const MethodInfo* method);'
 for cls,fields in [('Acted',['public ActedVersion acted; // 0x20','public RectTransform[] layoutsToRebuild; // 0x28','public GameObject highlight; // 0x30','public Image arrowImage; // 0x38','private Color savedArrowColor; // 0x40']),('ActedVersion',['public TextMeshProUGUI blankText; // 0x28','public TextMeshProUGUI text; // 0x30'])]:
  block=dump.split('public class '+cls+' ',1)[1].split('// Namespace:',1)[0]
  for field in fields:assert field in block
 gateway_names={'UnityEngine.Component$$get_gameObject':0x1C79FD0,'UnityEngine.GameObject$$get_activeSelf':0x1C7DC50,'UnityEngine.Object$$get_name':0x1C82250,'System.String$$Concat':0xF71C60,'UnityEngine.Debug$$Log':0x1C4B380,'ActedVersion$$Show':0x35D920,'UnityEngine.UI.LayoutRebuilder$$ForceRebuildLayoutImmediate':0x1EC1010,'UnityEngine.MonoBehaviour$$StopAllCoroutines':0x1C7F4A0,'UnityEngine.GameObject$$SetActive':0x1C7D810,'UnityEngine.MonoBehaviour$$.ctor':0x1C79770}
 verified_gateways=[]
 for n,a in gateway_names.items():
  rows=[r for r in meta['ScriptMethod'] if r['Name']==n and r['Address']==a];assert len(rows)==1,(n,rows);verified_gateways+=rows
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True;decoded={}
 for a,b in entries.values():
  nxt=min(r['Address'] for r in meta['ScriptMethod'] if r['Address']>a);ins=list(cs.disasm(pe.get_data(a,nxt-a),a))
  while ins[-1].mnemonic=='int3':ins.pop()
  assert ins[-1].address+ins[-1].size==b,(hex(a),hex(ins[-1].address+ins[-1].size));assert all(i.address+i.size==j.address for i,j in zip(ins,ins[1:]));decoded.update({i.address:i for i in ins})
 for a,expected in {0x35DF88:('movups','xmm0, xmmword ptr [rax]'),0x35DF8B:('movups','xmmword ptr [rbx + 0x40], xmm0'),0x35DF9C:('movdqa','xmmword ptr [rsp + 0x20], xmm1'),0x35DECE:('mov','rcx, qword ptr [rax + 0x28]'),0x35E0C7:('mov','rcx, qword ptr [rcx + 0x28]')}.items():assert a in decoded and (decoded[a].mnemonic,decoded[a].op_str)==expected
 i=decoded[0x35DF81];constant=list(struct.unpack('<IIII',pe.get_data(i.address+i.size+i.operands[1].mem.disp,16)))
 assert constant==[0,0x3F800000,0x3F800000,0x3F800000]
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
 for n in ('UnityEngine.UI.LayoutRebuilder_TypeInfo','UnityEngine.Debug_TypeInfo','literal:'):assert n in bindings,n
 literals={p:n[8:] for n,p in bindings.items() if n.startswith('literal:')};assert len(literals)==2
 logprefix=next(v for v in literals.values() if v);assert logprefix=='Character: '
 owner,acted0,acted1,text0,text1,visible0,game0,game1,highlight,arrow0,arrow1,layouts,alternate,rect0,rect1,texttype,imagetype=[arena+0x10000+n*0x1000 for n in range(17)]
 strings={arena+0x30000:'description',arena+0x30100:'first',arena+0x30200:'second',arena+0x30300:'Game',arena+0x30400:logprefix+'Game'}|literals
 names={0:None,owner:'owner',acted0:'acted0',acted1:'acted1',text0:'blank0',text1:'blank1',visible0:'visible0',game0:'game0',game1:'game1',highlight:'highlight',arrow0:'arrow0',arrow1:'arrow1',layouts:'layouts',alternate:'alternate',rect0:'rect0',rect1:'rect1'}
 saved=[0x80000000,0x7FC12345,0x3F800000,0x00000001];returned=[0x3E800000,0x3F000000,0x3F400000,0x3F800000]
 opt={};state={};visited=set()
 def color(p):return list(struct.unpack('<IIII',uc.mem_read(p,16)))
 def snap():return {'acted':names[rq(owner+0x20)],'layouts':names[rq(owner+0x28)],'arrow':names[rq(owner+0x38)],'saved_color':color(owner+0x40),'text':[state['texts'][text0],state['texts'][text1]],'active':dict(state['active']),'colors':dict(state['colors'])}
 def event(n,**kw):
  state['counts'][n]=state['counts'].get(n,0)+1;state['events'].append({'event':n,**kw,'fields':snap()})
  if opt.get('fail')==[n,state['counts'][n]]:state['error']=n;uc.emu_stop();return False
  change=opt.get('mutations',{}).get(n+str(state['counts'][n]),{})
  for field,pointers,offset in [('acted',[0,acted0,acted1],0x20),('arrow',[0,arrow0,arrow1],0x38),('layouts',[0,layouts,alternate],0x28)]:
   if field in change:q(owner+offset,pointers[change[field]])
  return True
 def hook(_,a,size,__):
  r=a-base;c,t,m=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
  if a==stop:uc.emu_stop();return
  if r==0x2B7B40:
   assert c-base in refs
   if event('metadata'):ret(rq(c))
  elif r==0x1C79FD0:
   assert c in (owner,acted0,acted1) and t==0
   if event('game_object',target=names[c]):ret(0 if opt.get('null_game_call')==state['counts']['game_object'] else game1 if state['counts']['game_object']==2 else game0)
  elif r==0x1C7DC50:
   assert c==game0 and t==0
   if event('active_self'):ret(0xABC000|int(opt.get('active',True)))
  elif r==0x1C82250:
   assert c==game0 and t==0
   if event('name'):ret(arena+0x30300)
  elif r==0xF71C60:
   assert strings[c]==logprefix and strings[t]=='Game' and m==0
   if event('concat',parts=[strings[c],strings[t]]):ret(arena+0x30400)
  elif r==0x1C4B380:
   assert m==0
   if event('log',text=strings[c],context=names[t]):ret()
  elif r==0x281D90:
   name=next(n for n,p in bindings.items() if p==c)
   if event('class_init',type=name):d(c+0xE0,1);ret()
  elif r==0x35D920:
   assert c in (acted0,acted1) and m==0
   if event('show',target=names[c],description=None if t==0 else strings[t]):ret()
  elif r==0x1EC1010:
   assert t==0
   if event('rebuild',target=names[c]):ret()
  elif r==0x1C7F4A0:
   assert c==owner and t==0
   if event('stop_all'):ret()
  elif r==0x1C7D810:
   assert c in (game0,game1,highlight) and m==0
   if event('set_active',target=names[c],value=bool(t&0xff)):state['active'][names[c]]=bool(t&0xff);ret()
  elif r==0x1C79770:
   assert c==owner and t==0
   if event('base_constructor'):ret()
  elif a==stop+0x100:
   assert c in (text0,text1) and t==0xABC1
   if event('get_text',target=names[c]):ret(0 if opt.get('null_text_return') else arena+0x30100 if c==text0 else arena+0x30200)
  elif a==stop+0x110:
   assert c in (text0,text1) and m==0xABC2
   if event('set_text',target=names[c],value=None if t==0 else strings[t]):state['texts'][c]=None if t==0 else strings[t];ret()
  elif a==stop+0x120:
   assert t in (arrow0,arrow1) and m==0xABC3
   if event('get_color',target=names[t]):
    p=arena+0x31000 if opt.get('external_color_buffer') else c;uc.mem_write(p,struct.pack('<IIII',*returned));ret(p)
  elif a==stop+0x130:
   assert c in (arrow0,arrow1) and m==0xABC4
   if event('set_color',target=names[c],bits=color(t)):state['colors'][names[c]]=color(t);ret()
  elif r in (0x2B7D90,0x2B7D80):state['error']='null' if r==0x2B7D90 else 'bounds';uc.emu_stop()
  elif r in decoded:visited.add(r)
  else:raise AssertionError(hex(r))
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 cases=[]
 def run(method,options=None):
  opt.clear();opt.update(options or {});state.clear();state.update(counts={},events=[],error=None,texts={text0:'first',text1:'second'},active={},colors={})
  for i in decoded.values():
   if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
  for n in ('UnityEngine.UI.LayoutRebuilder_TypeInfo','UnityEngine.Debug_TypeInfo'):d(bindings[n]+0xE0,0 if opt.get('cold') else 1)
  for p in (owner,acted0,acted1,text0,text1,visible0,layouts,alternate):uc.mem_write(p,bytes(0x100))
  q(owner+0x20,0 if opt.get('null_acted') else acted0);q(owner+0x28,0 if opt.get('null_layouts') else layouts);q(owner+0x30,0 if opt.get('null_highlight') else highlight);q(owner+0x38,0 if opt.get('null_arrow') else arrow0);uc.mem_write(owner+0x40,struct.pack('<IIII',*saved))
  for a,t in [(acted0,text0),(acted1,text1)]:q(a+0x28,0 if opt.get('null_blank') else t);q(a+0x30,visible0)
  for p in (text0,text1):q(p,texttype)
  for p in (arrow0,arrow1):q(p,imagetype)
  for offset,pointer in [(0x548,stop+0x100),(0x550,0xABC1),(0x558,stop+0x110),(0x560,0xABC2)]:q(texttype+offset,pointer)
  for offset,pointer in [(0x298,stop+0x120),(0x2A0,0xABC3),(0x2A8,stop+0x130),(0x2B0,0xABC4)]:q(imagetype+offset,pointer)
  values=opt.get('rects',[0,1]);d(layouts+0x18,len(values))
  for i,n in enumerate(values):q(layouts+0x20+i*8,0 if n is None else rect0 if n==0 else rect1)
  d(alternate+0x18,1);q(alternate+0x20,rect1)
  sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,owner);uc.reg_write(x.UC_X86_REG_RDX,0 if opt.get('null_description') else arena+0x30000)
  keep=[x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
  for rr in keep:uc.reg_write(rr,0xABC000+rr)
  before=bytes(uc.mem_read(owner,0x100));entry=entries['Acted$$'+method][0];uc.emu_start(base+entry,stop+0x200,count=10000)
  after=bytes(uc.mem_read(owner,0x100));assert before[:0x20]==after[:0x20] and before[0x30:0x38]==after[0x30:0x38] and before[0x50:]==after[0x50:]
  if state['error'] is None:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8 and all(reg(rr)==0xABC000+rr for rr in keep)
  result={'method':method,'options':dict(opt),'events':state['events'][:],'fields':snap(),'error':state['error'],'return_text':(None if reg(x.UC_X86_REG_RAX)==0 else strings[reg(x.UC_X86_REG_RAX)]) if method=='GetActed' and state['error'] is None else None};cases.append(result);return result
 expected={'GetActed':['game_object','active_self','get_text'],'UpdateActed':['game_object','name','concat','game_object','log','set_text'],'Act':['show','rebuild','rebuild'],'Hide':['stop_all','game_object','set_active'],'Highlight':['set_active','get_color','set_color'],'UnHighlight':['set_active','set_color'],'.ctor':['base_constructor']}
 for method in expected:
  baseline=run(method);assert baseline['error'] is None and [e['event'] for e in baseline['events']]==expected[method]
  if method=='GetActed':assert baseline['return_text']=='first'
  if method=='UpdateActed':assert baseline['fields']['text']==['description','second'] and baseline['events'][4]['context']=='game1'
  if method=='Highlight':assert baseline['fields']['saved_color']==returned and baseline['fields']['colors']['arrow0']==constant and baseline['fields']['active']=={'highlight':True}
  if method=='UnHighlight':assert baseline['fields']['colors']['arrow0']==saved and baseline['fields']['active']=={'highlight':False}
  counts={}
  for i,e in enumerate(baseline['events']):
   n=e['event'];counts[n]=counts.get(n,0)+1;r=run(method,{'fail':[n,counts[n]]});assert r['error']==n and r['events']==baseline['events'][:i+1] and r['fields']==e['fields']
 r=run('GetActed',{'active':False,'null_blank':True});assert r['return_text']=='' and [e['event'] for e in r['events']]==['game_object','active_self']
 r=run('GetActed',{'null_text_return':True});assert r['return_text'] is None and r['error'] is None
 for method in ('GetActed','UpdateActed','Act','Hide'):
  assert run(method,{'null_acted':True})['error']=='null'
 for method in ('GetActed','UpdateActed'):
  assert run(method,{'null_blank':True})['error']=='null'
  assert run(method,{'null_game_call':1})['error']=='null'
 r=run('UpdateActed',{'null_game_call':2,'null_description':True});assert r['error'] is None and r['fields']['text']==[None,'second'] and next(e for e in r['events'] if e['event']=='log')['context'] is None
 for method in ('Highlight','UnHighlight'):
  for key in ('null_highlight','null_arrow'):assert run(method,{key:True})['error']=='null'
 r=run('Highlight',{'external_color_buffer':True});assert r['fields']['saved_color']==returned and r['fields']['colors']['arrow0']==constant
 for change in (0,2):
  r=run('Highlight',{'mutations':{'get_color1':{'arrow':change}}});assert r['fields']['saved_color']==returned
  assert r['error']==('null' if change==0 else None)
  if change==2:assert r['fields']['colors']=={'arrow1':constant}
 for method,mutation_event in [('GetActed','active_self1'),('UpdateActed','log1'),('Hide','stop_all1')]:
  for changed in (0,2):
   r=run(method,{'mutations':{mutation_event:{'acted':changed}}});assert r['error']==('null' if changed==0 else None)
   if changed==2 and method=='GetActed':assert r['return_text']=='second'
   if changed==2 and method=='UpdateActed':assert r['fields']['text']==['first','description']
 for values in ([],[0,0,1],[None],[0,None,1]):
  r=run('Act',{'rects':values,'null_description':True});assert r['error'] is None
  assert [e['target'] for e in r['events'] if e['event']=='rebuild']==[None if n is None else 'rect'+str(n) for n in values]
 assert run('Act',{'null_layouts':True})['error']=='null'
 r=run('Act',{'mutations':{'show1':{'layouts':2}}});assert [e['target'] for e in r['events'] if e['event']=='rebuild']==['rect1']
 r=run('Act',{'mutations':{'rebuild1':{'layouts':2}}});assert [e['target'] for e in r['events'] if e['event']=='rebuild']==['rect0','rect1']
 for method in ('Act','UpdateActed'):
  r=run(method,{'cold':True});assert r['error'] is None and sum(e['event']=='class_init' for e in r['events'])==1
  r=run(method,{'cold':True,'fail':['class_init',1]});assert r['error']=='class_init'
 r=run('GetActed',{'active':False,'mutations':{'active_self1':{'acted':0}}});assert r['error'] is None and r['return_text']==''
 for method in ('Highlight','UnHighlight'):
  r=run(method,{'mutations':{'set_active1':{'arrow':2}}});assert r['error'] is None and 'arrow1' in r['fields']['colors']
 r=run('Hide',{'null_game_call':1});assert r['error']=='null' and not r['fields']['active']
 r=run('Act',{'cold':True,'mutations':{'class_init1':{'layouts':2}}});assert [e['target'] for e in r['events'] if e['event']=='rebuild']==['rect0','rect1']
 return {'schema_version':1,'build_id':BUILD,'exact_declarations':exact,'cases_passed':len(cases),'distinct_native_instructions':len(visited),'highlight_color_bits':constant,'log_prefix':logprefix,'metadata_bindings':sorted(bindings),'verified_gateways':verified_gateways,'cases':cases,'scope':'Seven immediate Acted callers, exact blankText virtual slots and hidden color return-buffer ABI, saved-before-overwrite order and controlled field rereads. Engine/text/layout/ActedVersion.Show/format and virtual-color services explicit; no tween, delayed-coroutine or managed unwind reconstruction.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(f"Passed {r['cases_passed']} Acted surface cases")
