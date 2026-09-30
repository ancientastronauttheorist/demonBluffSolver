"""Native complete ManageCharacters caller with explicitly supplied managed gateways."""
import argparse,hashlib,itertools,json,struct
from pathlib import Path
from audit_character_assets import BUILD

def audit(game_root,dumper_root):
 import capstone,pefile,unicorn
 from unicorn import x86_const as x
 assert unicorn.__version__=='2.1.4';repo=Path(__file__).parents[1]
 lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'));ext=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
 def pin(p,h):
  b=p.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==h.upper();return b
 raw=pin(game_root/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256']);meta=json.loads(pin(dumper_root/'script.json',ext['outputs']['script_json']['sha256']).decode('utf-8-sig'))
 dump=pin(dumper_root/'dump.cs',ext['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
 fields=[('public class Characters : MonoBehaviour //', ['public List<Character> characters; // 0x20','public CharacterData[] startGameActOrder; // 0x28','public Action onSetup; // 0x58']),('public class Character : MonoBehaviour, ICard //',['public CharacterData dataRef; // 0x50']),('public class CharacterData : ScriptableObject, ICharacterLocData, ICardData //',['public Role role; // 0x140']),('private sealed class Characters.<ShuffleDeck>d__16 :',['private int <>1__state; // 0x10'])]
 for declaration,expected in fields:
  assert dump.count(declaration)==1;block=dump.split(declaration,1)[1].split('// Methods',1)[0]
  for field in expected:assert field in block,field
 exact=[]
 for a,name in [(0x36ce30,'Characters$$ManageCharacters'),(0x36e4e0,'Characters$$UpdateCharacterPositions'),(0x36d3a0,'Characters$$PickRoundBluffs'),(0x36d720,'Characters$$PickRoundDuplicates'),(0x365a20,'Character$$Init'),(0x3645c0,'Character$$Act'),(0x3811b0,'Gameplay$$UpdateCharacters')]:
  rows=[r for r in meta['ScriptMethod'] if r['Address']==a and r['Name']==name];assert len(rows)==1;exact.append(rows[0])
 assert exact[0]['Signature']=='void Characters__ManageCharacters (Characters_o* __this, System_Collections_Generic_List_CharacterData__o* charactersList, const MethodInfo* method);'
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
 # Decode from the verified full method entry, including its cold path and null stubs.
 rawins=list(cs.disasm(pe.get_data(0x36ce30,0x36d3a0-0x36ce30),0x36ce30))
 while rawins[-1].mnemonic=='int3':rawins.pop()
 assert all(a.address+a.size==b.address for a,b in zip(rawins,rawins[1:]));decoded={i.address:i for i in rawins}
 checks={0x36cf00:('call','0x36e4e0'),0x36cf0a:('call','0x36d3a0'),0x36cf14:('call','0x36d720'),0x36cf19:('mov','rdx, qword ptr [r12 + 0x20]'),0x36cf7a:('mov','rax, qword ptr [r12 + 0x20]'),0x36cf88:('mov','esi, dword ptr [rax + 0x18]'),0x36cfa0:('sub','esi, edi'),0x36cfc0:('call','0xb22150'),0x36cfda:('call','0x365a20')}
 checks.update({0x36d01e:('mov','rbx, qword ptr [r12 + 0x20]'),0x36d03d:('call','0x3811b0'),0x36d0ba:('call','0x3645c0'),0x36d0f9:('mov','r13, qword ptr [r12 + 0x28]'),0x36d12c:('mov','r14, qword ptr [r13 + r14*8 + 0x20]'),0x36d131:('mov','rdx, qword ptr [r12 + 0x20]'),0x36d1c9:('call','0x1c822c0'),0x36d1dc:('call','0x3645c0'),0x36d1ea:('mov','rdx, qword ptr [r14 + 0x140]'),0x36d2db:('mov','rax, qword ptr [r12 + 0x58]'),0x36d2ed:('call','qword ptr [rax + 0x18]'),0x36d313:('call','0x2b7d40'),0x36d325:('mov','dword ptr [rbx + 0x10], 0'),0x36d335:('call','0x1c7f160')})
 for a,v in checks.items():assert (decoded[a].mnemonic,decoded[a].op_str)==v
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
 arena,stack,stop=0x200000000,0x300000000,0x400000000
 uc.mem_map(arena,0x40000);uc.mem_map(stack,0x10000);uc.mem_map(stop,0x1000)
 def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
 def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
 def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
 def rd(a):return struct.unpack('<I',uc.mem_read(a,4))[0]
 def reg(r):return uc.reg_read(r)
 def ret(v=0):
  sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
 refs=set();flags=set()
 for i in rawins:
  for o in i.operands:
   if o.type==capstone.CS_OP_MEM and o.mem.base==capstone.x86.X86_REG_RIP:refs.add(i.address+i.size+o.mem.disp)
  if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:flags.add(i.address+i.size+i.operands[0].mem.disp)
 bindings={};slots={}
 for r in meta['ScriptMetadata']+meta['ScriptMetadataMethod']:
  if r['Address'] in refs:p=arena+0x1000+len(bindings)*0x200;bindings[r['Name']]=p;slots[base+r['Address']]=p;q(base+r['Address'],p)
 required_tokens={0x36cf27:'Method$System.Collections.Generic.List<Character>.GetEnumerator()',0x36cf60:'Method$System.Collections.Generic.List.Enumerator<Character>.MoveNext()',0x36cfb4:'Method$System.Collections.Generic.List<CharacterData>.get_Item()',0x36cfe6:'Method$System.Collections.Generic.List.Enumerator<Character>.Dispose()'}
 for address,name in required_tokens.items():
  i=decoded[address];slot=base+i.address+i.size+i.operands[1].mem.disp
  assert name in bindings and slots[slot]==bindings[name]
 owner,board,alternate,roster,order,other_order,callback,iterator=[arena+a for a in range(0x8000,0x10000,0x1000)]
 chars=[arena+0x10000+i*0x200 for i in range(5)];datas=[arena+0x12000+i*0x200 for i in range(5)];roles=[arena+0x14000+i*0x200 for i in range(5)]
 cb_entry=stop+0x100
 labels={0:None,owner:'owner',board:'board',alternate:'alternate',roster:'roster',order:'order',other_order:'other_order',callback:'callback',iterator:'iterator',**{p:f'card{i}' for i,p in enumerate(chars)},**{p:f'data{i}' for i,p in enumerate(datas)}}
 state={};opt={};visited=set();lists={};enumerators={}
 def snap():return {'board':labels[rq(owner+0x20)],'order':labels[rq(owner+0x28)],'identities':[labels[rq(p+0x50)] for p in chars],'effects':state['effects'][:],'iterator_state':rd(iterator+0x10)}
 def emit(kind,**kw):
  state['counts'][kind]=state['counts'].get(kind,0)+1;state['events'].append({'kind':kind,**kw,'snapshot':snap()})
  if opt.get('fail')==[kind,state['counts'][kind]]:state['error']=kind;uc.emu_stop();return False
  return True
 def mutate(kind):
  for op in opt.get('mutations',[]):
   if op[:2]==[kind,state['counts'][kind]]:
    target,value=op[2:]
    if target=='board':q(owner+0x20,{'alternate':alternate,'null':0}[value])
    elif target=='order':q(owner+0x28,other_order)
    elif target=='order_length':d(order+0x18,value)
    elif target=='role':q(datas[0]+0x140,roles[value])
    elif target=='callback':q(owner+0x58,0 if value==0 else callback)
    elif target=='identity':q(chars[1]+0x50,datas[value])
 def finish(kind):state['effects'].append(kind);mutate(kind)
 def hook(_,a,size,__):
  r=a-base;c,t,m=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
  if a==cb_entry:
   assert c==owner and t==callback
   if emit('callback'):finish('callback');ret()
  elif r in [0x36e4e0,0x36d3a0,0x36d720]:
   assert c==owner and t==0;kind={0x36e4e0:'positions',0x36d3a0:'unique',0x36d720:'duplicates'}[r]
   if emit(kind):finish(kind);ret()
  elif r==0x2b7b40:
   assert c in slots
   if emit('metadata'):ret(slots[c])
  elif r==0x281d90:
   assert c in bindings.values()
   if emit('class_init',name=next(n for n,p in bindings.items() if p==c)):d(c+0xe0,1);mutate('class_init');ret()
  elif r==0xb16640:
   assert t in lists and m==bindings[required_tokens[0x36cf27]]
   if emit('enumerator',source=labels[t]):
    ident=len(enumerators)+1;enumerators[ident]=[t,0];uc.mem_write(c,bytes(24));q(c,ident);ret(c)
  elif r==0x9693d0:
   assert t==bindings[required_tokens[0x36cf60]];entry=enumerators[rq(c)];source,index=entry
   if emit('move_next',source=labels[source],index=index):
    available=index<len(lists[source]);q(c+0x10,lists[source][index] if available else 0);entry[1]+=int(available);mutate('move_next');ret(0xabc000|int(available))
  elif r==0xb22150:
   assert c==roster and m==bindings[required_tokens[0x36cfb4]]
   if emit('get_item',index=t):
    if t>=len(state['roster']):state['error']='bounds';uc.emu_stop()
    else:ret(state['roster'][t])
  elif r==0x365a20:
   assert c in chars and reg(x.UC_X86_REG_R9)==0
   if emit('init',card=labels[c],data=labels[t],display_id=m&0xffffffff):q(c+0x50,t);finish('init');ret()
  elif r==0x3811b0:
   assert t==0
   if emit('publish',source=labels[c]):finish('publish');ret()
  elif r==0x3645c0:
   assert c in chars and t in [3,5] and m==0;kind='act_init' if t==3 else 'act_start'
   if emit(kind,card=labels[c]):finish(kind);ret()
  elif r==0x1c822c0:
   assert m==0
   if emit('equal',left=labels[c],right=labels[t]):ret(int(c==t))
  elif r==0x33ed50:
   kind='iterator_ctor' if c==iterator else 'dispose'
   assert (t==0 if kind=='iterator_ctor' else t==bindings[required_tokens[0x36cfe6]])
   if emit(kind):ret()
  elif r==0x2b7d40:
   assert c==bindings['Characters.<ShuffleDeck>d__16_TypeInfo']
   if emit('allocate'):ret(iterator)
  elif r==0x1c7f160:
   assert c==owner and t==iterator and m==0 and rd(iterator+0x10)==0
   if emit('start_coroutine'):finish('start_coroutine');ret()
  elif r in [0x2b7d90,0x2b7d80]:state['error']='null' if r==0x2b7d90 else 'bounds';uc.emu_stop()
  else:assert r in decoded and decoded[r].size==size,hex(r);visited.add(r)
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 nv=[x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
 def run(options=None):
  opt.clear();opt.update(options or {});state.clear();state.update(counts={},events=[],effects=[],error=None,roster=[datas[i] if i is not None else 0 for i in opt.get('roster',[0,0,1])]);enumerators.clear();lists.clear()
  lists[board]=[chars[i] if i is not None else 0 for i in opt.get('board',[0,1,2])];lists[alternate]=[chars[3],chars[4]]
  uc.mem_write(owner,bytes(0x80));q(owner+0x20,0 if opt.get('null_board') else board);q(owner+0x28,0 if opt.get('null_order') else order);q(owner+0x58,0 if opt.get('no_callback') else callback)
  d(board+0x18,len(lists[board]));d(alternate+0x18,2);q(callback+0x18,cb_entry);q(callback+0x28,callback);q(callback+0x40,owner);d(iterator+0x10,0xeeeeeeee)
  for i,p in enumerate(chars):q(p+0x50,datas[i])
  for i,p in enumerate(datas):q(p+0x140,roles[i])
  for p in bindings.values():d(p+0xe0,0 if opt.get('cold') else 1);uc.mem_write(p+0x130,b'\x01');q(p+0xc8,p+0x180);q(p+0x180,p)
  for i,p in enumerate(roles):
   klass=arena+0x16000+i*0x200;q(p,klass);uc.mem_write(klass+0x130,b'\x01');q(klass+0xc8,klass+0x180);q(klass+0x180,klass)
  for i,name in enumerate(['Alchemist_TypeInfo','Poisoner_TypeInfo','Puzzlemaster_TypeInfo'],1):q(roles[i],bindings[name])
  q(datas[0]+0x140,0 if opt.get('role')=='null' else roles[opt.get('role',0)])
  if opt.get('subclass'):
   klass=arena+0x18000;q(roles[0],klass);uc.mem_write(klass+0x130,b'\x02');q(klass+0xc8,klass+0x180);q(klass+0x180,bindings[['Alchemist_TypeInfo','Poisoner_TypeInfo','Puzzlemaster_TypeInfo'][opt['subclass']-1]]);q(klass+0x188,klass)
  seq=opt.get('order',[0,1]);d(order+0x18,len(seq));d(other_order+0x18,0)
  for i,value in enumerate(seq):q(order+0x20+8*i,0 if value is None else datas[value])
  for flag in flags:uc.mem_write(base+flag,bytes([0 if opt.get('cold') else 1]))
  sp=stack+0x8008;q(sp,stop)
  for i,r in enumerate(nv):uc.reg_write(r,0x123400+i)
  uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,owner);uc.reg_write(x.UC_X86_REG_RDX,0 if opt.get('null_roster') else roster)
  uc.emu_start(base+0x36ce30,stop,count=100000)
  returned=reg(x.UC_X86_REG_RIP)==stop
  assert state['error'] or returned
  if returned:
   assert reg(x.UC_X86_REG_RSP)==sp+8
   assert [reg(r) for r in nv]==[0x123400+i for i in range(len(nv))]
  return {'input':dict(opt),'events':state['events'][:],'final':snap(),'error':state['error'],'returned':returned}
 cases=[]
 def add(o=None):r=run(o);cases.append(r);return r
 def calls(r,kind):return [e for e in r['events'] if e['kind']==kind]
 for cold in [False,True]:
  r=add({'cold':cold});assert r['returned'];assert [e['display_id'] for e in calls(r,'init')]==[3,2,1];assert [e['card'] for e in calls(r,'act_init')]==['card0','card1','card2'];assert [e['card'] for e in calls(r,'act_start')]==['card0','card2']
  counts={}
  for i,e in enumerate(r['events']):
   k=e['kind'];counts[k]=counts.get(k,0)+1;f=add({'cold':cold,'fail':[k,counts[k]]});assert f['events']==r['events'][:i+1] and f['final']==e['snapshot'] and f['error']==k
 for subclass in [1,2,3]:
  r=add({'subclass':subclass});assert len(calls(r,'act_start'))==3
 for role in [1,2,3,'null']:
  r=add({'role':role});assert len(calls(r,'act_start'))==(2 if role=='null' else 3)
 r=add({'board':[0,0,2]});assert [e['card'] for e in calls(r,'init')]==['card0','card0','card2'];assert len(calls(r,'act_init'))==3
 r=add({'board':[],'order':[],'null_roster':True});assert r['returned'] and not calls(r,'init')
 r=add({'order':[0,0]});assert [e['card'] for e in calls(r,'act_start')]==['card0','card0']
 r=add({'order':[4]});assert not calls(r,'act_start') and r['returned']
 r=add({'no_callback':True});assert r['returned'] and not calls(r,'callback')
 r=add({'roster':[None,0,1],'order':[None]});assert r['error']=='null' and [e['card'] for e in calls(r,'act_start')]==['card0']
 r=add({'mutations':[['act_init',1,'callback',0]]});assert r['returned'] and not calls(r,'callback')
 r=add({'no_callback':True,'mutations':[['act_start',1,'callback',1]]});assert len(calls(r,'callback'))==1
 for o in [{'null_board':True},{'null_roster':True},{'null_order':True},{'board':[None]},{'roster':[]}]:assert add(o)['error']
 for kind,count in [('init',1),('publish',1),('act_init',1),('act_start',1),('callback',1)]:
  r=add({'mutations':[[kind,count,'board','alternate']]});assert r['returned']
  if kind=='init':assert [e['display_id'] for e in calls(r,'init')]==[3,1,0] and calls(r,'publish')[0]['source']=='alternate'
  if kind in ['init','publish']:assert [e['card'] for e in calls(r,'act_init')]==['card3','card4']
  if kind=='act_init':assert [e['card'] for e in calls(r,'act_init')]==['card0','card1','card2'] and not calls(r,'act_start')
  if kind=='act_start':assert [e['card'] for e in calls(r,'act_start')]==['card0']
 r=add({'mutations':[['act_start',1,'role',1]]});assert len(calls(r,'act_start'))==3
 r=add({'role':1,'mutations':[['act_start',1,'role',0]]});assert len(calls(r,'act_start'))==2
 r=add({'mutations':[['act_start',1,'order',0]]});assert len(calls(r,'act_start'))==2
 r=add({'mutations':[['act_start',1,'order_length',1]]});assert len(calls(r,'act_start'))==1
 r=add({'role':1,'mutations':[['act_start',1,'identity',1]]});assert [e['card'] for e in calls(r,'act_start')]==['card0','card1','card2']
 for kind in ['init','publish','act_init','act_start']:
  r=add({'mutations':[[kind,1,'board','null']]});assert r['error']=='null'
 return {'build_id':BUILD,'field_declarations_verified':fields,'metadata_verified':exact,'required_generic_tokens':list(required_tokens.values()),'cases_passed':len(cases),'native_assertions':len(checks),'native_instructions_executed':len(visited),'cases':cases,'scope':'Complete native ManageCharacters caller. Layout, pool builders, Character.Init (only synthetic dataRef write), publication, Character.Act, delegate, coroutine allocation/registration, collection, equality, metadata and class initialization are explicit supplied gateways. No callee body, Unity lifecycle, scheduler, real managed collection mutation/version checks, exception unwinding, or full engine replay is claimed.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',required=True,type=Path)
 a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(r['cases_passed'],r['native_instructions_executed'])
