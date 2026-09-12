"""Native ManageCharacters prefix, stopping before first Character.Init/publication."""
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
 exact=[]
 for a,name in [(0x36ce30,'Characters$$ManageCharacters'),(0x36e4e0,'Characters$$UpdateCharacterPositions'),(0x36d3a0,'Characters$$PickRoundBluffs'),(0x36d720,'Characters$$PickRoundDuplicates'),(0x365a20,'Character$$Init')]:
  rows=[r for r in meta['ScriptMethod'] if r['Address']==a and r['Name']==name];assert len(rows)==1;exact.append(rows[0])
 assert exact[0]['Signature']=='void Characters__ManageCharacters (Characters_o* __this, System_Collections_Generic_List_CharacterData__o* charactersList, const MethodInfo* method);'
 pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
 # Decode from the verified full method entry, including its cold path and null stubs.
 rawins=list(cs.disasm(pe.get_data(0x36ce30,0x36d3a0-0x36ce30),0x36ce30))
 while rawins[-1].mnemonic=='int3':rawins.pop()
 assert all(a.address+a.size==b.address for a,b in zip(rawins,rawins[1:]));decoded={i.address:i for i in rawins}
 checks={0x36cf00:('call','0x36e4e0'),0x36cf0a:('call','0x36d3a0'),0x36cf14:('call','0x36d720'),0x36cf19:('mov','rdx, qword ptr [r12 + 0x20]'),0x36cf7a:('mov','rax, qword ptr [r12 + 0x20]'),0x36cf88:('mov','esi, dword ptr [rax + 0x18]'),0x36cfa0:('sub','esi, edi'),0x36cfc0:('call','0xb22150'),0x36cfda:('call','0x365a20')}
 for a,v in checks.items():assert (decoded[a].mnemonic,decoded[a].op_str)==v
 uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
 arena,stack,stop=0x200000000,0x300000000,0x400000000
 uc.mem_map(arena,0x20000);uc.mem_map(stack,0x10000);uc.mem_map(stop,0x1000)
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
 owner,board,alternate,roster,character,other_character,data= [arena+a for a in range(0x8000,0xf000,0x1000)]
 labels={0:None,owner:'owner',board:'board',alternate:'alternate',roster:'roster',character:'character',other_character:'other_character',data:'data'}
 state={};opt={};visited=set()
 def snap():return {'board':labels[rq(owner+0x20)],'board_size':rd(board+0x18),'alternate_size':rd(alternate+0x18),'effects':state['effects'][:]}
 def emit(kind,**kw):
  state['counts'][kind]=state['counts'].get(kind,0)+1;state['events'].append({'kind':kind,**kw,'snapshot':snap()})
  if opt.get('fail')==[kind,state['counts'][kind]]:state['error']=kind;uc.emu_stop();return False
  return True
 def hook(_,a,size,__):
  r=a-base;c,t,m=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
  if r==0x36d01e:state['boundary']='empty_before_publication';uc.emu_stop();return
  if r in [0x36e4e0,0x36d3a0,0x36d720]:
   assert c==owner and t==0;kind={0x36e4e0:'positions',0x36d3a0:'unique',0x36d720:'duplicates'}[r]
   if emit(kind):
    state['effects'].append(kind)
    if opt.get('replace_after')==kind:q(owner+0x20,alternate)
    ret()
  elif r==0x2b7b40:
   assert c in slots
   if emit('metadata'):ret(slots[c])
  elif r==0x281d90:
   assert c in bindings.values()
   if emit('class_init',class_name=next(n for n,p in bindings.items() if p==c)):d(c+0xe0,1);ret()
  elif r==0xb16640:
   assert t in [board,alternate] and m==bindings[required_tokens[0x36cf27]]
   if emit('enumerator',source=labels[t]):uc.mem_write(c,bytes(24));q(c,t);ret(c)
  elif r==0x9693d0:
   assert t==bindings[required_tokens[0x36cf60]]
   source=rq(c)
   if emit('move_next',source=labels[source]):
    if opt.get('replace_after')=='move_next':q(owner+0x20,alternate)
    if opt.get('null_after_move'):q(owner+0x20,0)
    q(c+0x10,0 if opt.get('null_character') else character if source==board else other_character)
    ret(0xabc000|int(not opt.get('empty_board')))
  elif r==0xb22150:
   assert c==roster and t==0 and m==bindings[required_tokens[0x36cfb4]]
   if emit('get_item',index=t):
    if opt.get('empty_roster'):state['error']='bounds';uc.emu_stop()
    else:ret(0 if opt.get('null_data') else data)
  elif r==0x365a20:
   assert reg(x.UC_X86_REG_R9)==0
   state['boundary']='before_first_init';state['init_arguments']={'character':labels[c],'data':labels[t],'display_id_bits':m&0xffffffff};uc.emu_stop()
  elif r==0x33ed50:
   assert t==bindings[required_tokens[0x36cfe6]]
   if emit('dispose'):ret()
  elif r in [0x2b7d90,0x2b7d80]:state['error']='null' if r==0x2b7d90 else 'bounds';uc.emu_stop()
  else:assert r in decoded and decoded[r].size==size,hex(r);visited.add(r)
 uc.hook_add(unicorn.UC_HOOK_CODE,hook)
 def run(options=None):
  opt.clear();opt.update(options or {});state.clear();state.update(counts={},events=[],effects=[],error=None,boundary=None,init_arguments=None)
  uc.mem_write(owner,bytes(0x80));q(owner+0x20,0 if opt.get('null_board') else board);d(board+0x18,opt.get('board_size',3));d(alternate+0x18,opt.get('alternate_size',5))
  for p in bindings.values():d(p+0xe0,0 if opt.get('cold') else 1)
  for flag in flags:uc.mem_write(base+flag,bytes([0 if opt.get('cold') else 1]))
  sp=stack+0x8008;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,owner);uc.reg_write(x.UC_X86_REG_RDX,0 if opt.get('null_roster') else roster)
  before=bytes(uc.mem_read(owner,0x80));uc.emu_start(base+0x36ce30,stop,count=10000)
  after=bytes(uc.mem_read(owner,0x80));assert before[:0x20]==after[:0x20] and before[0x28:]==after[0x28:]
  assert state['error'] or state['boundary']
  return {'input':dict(opt),'events':state['events'][:],'final':snap(),'error':state['error'],'boundary':state['boundary'],'init_arguments':state['init_arguments']}
 cases=[]
 for cold in [False,True]:
  baseline=run({'cold':cold});assert baseline['boundary']=='before_first_init' and baseline['init_arguments']=={'character':'character','data':'data','display_id_bits':3};cases.append(baseline);counts={}
  for index,event in enumerate(baseline['events']):
   kind=event['kind'];counts[kind]=counts.get(kind,0)+1;r=run({'cold':cold,'fail':[kind,counts[kind]]})
   assert r['error']==kind and r['events']==baseline['events'][:index+1] and r['final']==event['snapshot'];cases.append(r)
 for options,error in [({'null_board':True},'null'),({'null_roster':True},'null'),({'empty_roster':True},'bounds'),({'null_character':True},'null'),({'null_after_move':True},'null')]:
  r=run(options);assert r['error']==error and r['final']['effects']==['positions','unique','duplicates'];cases.append(r)
 for value in [0,1,7,-1,-2**31]:
  r=run({'board_size':value});assert r['init_arguments']['display_id_bits']==abs(value)&0xffffffff;cases.append(r)
 for stage in ['positions','unique','duplicates','move_next']:
  r=run({'replace_after':stage});assert r['init_arguments']=={'character':'character' if stage=='move_next' else 'other_character','data':'data','display_id_bits':5};cases.append(r)
 r=run({'null_character':True,'empty_roster':True});assert r['error']=='bounds';cases.append(r)
 r=run({'null_data':True});assert r['init_arguments']['data'] is None;cases.append(r)
 r=run({'empty_board':True,'board_size':0,'null_roster':True});assert r['boundary']=='empty_before_publication' and r['error'] is None and r['events'][-1]['kind']=='dispose';cases.append(r)
 return {'build_id':BUILD,'metadata_verified':exact,'required_generic_tokens':list(required_tokens.values()),'cases_passed':len(cases),'native_assertions':len(checks),'native_instructions_executed':len(visited),'cases':cases,'scope':'Actual ManageCharacters prefix only. Stops before Character.Init or before publication for empty enumeration. Positions/unique/duplicates and collection/metadata/class-init are supplied preserving services with optional controlled board replacement. No per-card Init, later callback passes, shuffle, unwind or complete ManageCharacters replay.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',required=True,type=Path)
 a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(r['cases_passed'],r['native_instructions_executed'])
