"""Audit the pinned ProjectContext -> GameData -> ascension asset graph.

Read typed serialized values through complete objects, retaining order and
multiplicity. This is configuration evidence, not a claim that any mode chooses
all these profiles or that runtime clones preserve their initial contents.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from audit_character_assets import Cursor, BUILD, ASSET_HASHES

COUNT_FIELDS = ('allCharCount','town','demon','outs','minion','dTown','dDemon','dOuts','dMinion')
SCRIPT_LISTS = ('startingTownsfolks','startingOutsiders','startingMinions','startingDemons','mustInclude')
ASCENSION_LISTS = ('unlockedCharacters','mustInlcude','alwaysInDeck','startingTownsfolks','startingOutsiders','startingMinions','startingDemons')
POOL_LISTS = ('townsfolks','outsiders','minions','demons')


def pointers(cursor): return cursor.array(cursor.pointer)
def character_count(cursor): return dict(zip(COUNT_FIELDS,cursor.read('iiiiiiiii')))
def script_info(cursor):
    value={name:pointers(cursor) for name in SCRIPT_LISTS}
    value['characterCounts']=cursor.array(lambda:character_count(cursor));return value

def finish(cursor):
    if cursor.offset!=len(cursor.data):raise ValueError(f'incomplete object: {cursor.offset}/{len(cursor.data)}')

def parse_game_data(raw):
    c=Cursor(raw);v={'name':c.string(),'version':c.string(),'language':c.read('i'),'localizationData':c.pointer(),'saveData':c.pointer()}
    for name in ('allCharacterData','allRelicsData'):v[name]=pointers(c)
    v['advancedAscension']=c.pointer();v['roguelikeStandardAscensions']=c.array(lambda:pointers(c))
    for name in ('roguelikeAscensions','standardAscensions'):v[name]=pointers(c)
    for name in ('allCharactersAscension','debugAscension','currentTemporaryAscension'):v[name]=c.pointer()
    v['allAchievements']=pointers(c);v['showUpdateScript']=c.boolean();finish(c);return v

def parse_ascension(raw):
    c=Cursor(raw);v={'name':c.string(),'possibleScriptsData':pointers(c),'possibleScripts':c.array(lambda:script_info(c))}
    for name in ASCENSION_LISTS:v[name]=pointers(c)
    v['currentPickedScript']=script_info(c)
    for name in POOL_LISTS:v[name]=pointers(c)
    v['characterCounts']=c.array(lambda:character_count(c))
    # Action delegates are not serialized here. The remaining five integer
    # fields consume the entire CardAdditionInfo payload in every pinned object.
    v['cardAdditions']=c.array(lambda:c.array(lambda:dict(zip(('rewardType','charType','addType','amount','day'),c.read('iiiii')))))
    finish(c);return v

def parse_custom_script(raw):
    c=Cursor(raw);v={'name':c.string(),'scriptInfo':script_info(c)};finish(c);return v

def parse_compendium(raw):
    c=Cursor(raw);v={'name':c.string(),'cards':pointers(c)}
    v['pages']=c.array(lambda:{'pageName':c.string(),'characterDatas':pointers(c)})
    for name in ('pagesText','completion','completionBg','pageName'):v[name]=c.pointer()
    v['currentPage']=c.read('i');finish(c);return v

def all_pointers(value):
    if isinstance(value,tuple) and len(value)==2:yield value
    elif isinstance(value,list):
        for child in value:yield from all_pointers(child)
    elif isinstance(value,dict):
        for child in value.values():yield from all_pointers(child)


def audit(game_root,dumper_root):
    import UnityPy
    root=Path(game_root)/'Demon Bluff_Data';repo=Path(__file__).parents[1]
    def pinned(path,digest):
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest().upper()!=digest.upper():raise ValueError(f'fingerprint changed: {path.name}')
        return raw
    for name in ('sharedassets0.assets','globalgamemanagers.assets','level0'):pinned(root/name,ASSET_HASHES[name])
    extraction=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    dump=pinned(Path(dumper_root)/'dump.cs',extraction['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    expected={
        'GameData':['version','language','localizationData','saveData','allCharacterData','allRelicsData','advancedAscension','roguelikeStandardAscensions','roguelikeAscensions','standardAscensions','allCharactersAscension','debugAscension','currentTemporaryAscension','allAchievements','showUpdateScript'],
        'ProjectContext':['gameData','cursorPosition'],
        'AscensionsData':['possibleScriptsData','possibleScripts',*ASCENSION_LISTS,'currentPickedScript',*POOL_LISTS,'characterCounts','cardAdditions'],
        'ScriptInfo':[*SCRIPT_LISTS,'characterCounts'],'CharactersCount':list(COUNT_FIELDS),
        'AscensionsList':['ascensions'],'CustomScriptData':['scriptInfo'],'CardAdditionPerDay':['additions'],
        'CardAdditionInfo':['action','rewardType','charType','addType','amount','day'],
        'Compendium':['cards','pages','pagesText','completion','completionBg','pageName','currentPage'],
        'CharactersCompendiumPage':['pageName','characterDatas']}
    for name,fields in expected.items():
        block=re.search(r'^public class '+name+r'\b[^\n]*\n\{(.*?)\n\}',dump,re.M|re.S)
        if not block:raise ValueError(f'missing declaration: {name}')
        actual=re.findall(r'^\s*public (?!static\b)[^;\n]+ (\w+); // 0x[0-9A-F]+$',block[1].split('// Methods')[0],re.M)
        if actual!=fields:raise ValueError(f'serialized field declaration changed: {name}')
    manager=UnityPy.load(str(root/'globalgamemanagers.assets'));mono={}
    for obj in manager.objects:
        if obj.type.name!='MonoScript':continue
        data=obj.read_typetree()
        if data.get('m_AssemblyName')=='Assembly-CSharp' and data.get('m_Namespace')=='':mono[data['m_ClassName']]=obj.path_id
    shared=UnityPy.load(str(root/'sharedassets0.assets'));level=UnityPy.load(str(root/'level0'))
    if next(iter(shared.files.values())).externals[0].path!='globalgamemanagers.assets':raise ValueError('shared script-file binding changed')
    if [x.path for x in next(iter(level.files.values())).externals[:2]]!=['globalgamemanagers.assets','sharedassets0.assets']:raise ValueError('scene file binding changed')
    def objects_of(environment,class_name):
        return [o for o in environment.objects if o.type.name=='MonoBehaviour'
                and o.read_typetree(check_read=False).get('m_Script')=={'m_FileID':1,'m_PathID':mono[class_name]}]
    def record(obj,parser):
        raw=obj.get_raw_data();result=parser(raw)
        if result['name']!=obj.read_typetree(check_read=False)['m_Name']:raise ValueError('header name mismatch')
        return {'path_id':obj.path_id,'object_size':len(raw),'object_sha256':hashlib.sha256(raw).hexdigest().upper(),'data':result}
    contexts=objects_of(level,'ProjectContext')
    if len(contexts)!=1:raise ValueError('ambiguous ProjectContext')
    context_raw=contexts[0].get_raw_data();c=Cursor(context_raw);name=c.string();game_pointer=c.pointer();cursor_pointer=c.pointer();finish(c)
    games=objects_of(shared,'GameData')
    if len(games)!=1 or game_pointer!=(2,games[0].path_id):raise ValueError('ProjectContext GameData link changed')
    game=record(games[0],parse_game_data)
    ascensions=[record(o,parse_ascension) for o in objects_of(shared,'AscensionsData')]
    customs=[record(o,parse_custom_script) for o in objects_of(shared,'CustomScriptData')]
    compendia=[record(o,parse_compendium) for o in objects_of(level,'Compendium')]
    for group in (ascensions,customs,compendia):group.sort(key=lambda r:r['path_id'])
    if len(ascensions)!=46 or len(customs)!=12 or len(compendia)!=1:raise ValueError('profile/custom/catalogue inventory changed')
    ascension_ids={r['path_id'] for r in ascensions};custom_ids={r['path_id'] for r in customs}
    references=[game['data']['advancedAscension'],*game['data']['roguelikeAscensions'],*game['data']['standardAscensions'],
                game['data']['allCharactersAscension'],game['data']['debugAscension'],game['data']['currentTemporaryAscension']]
    references += [p for group in game['data']['roguelikeStandardAscensions'] for p in group]
    if any(file!=0 or path not in ascension_ids for file,path in references):raise ValueError('unresolved ascension reference')
    if any(file!=0 or path not in custom_ids for r in ascensions for file,path in r['data']['possibleScriptsData']):raise ValueError('unresolved custom script')
    catalogue = game['data']['allCharacterData']
    compendium_catalogue = [p for page in compendia[0]['data']['pages'] for p in page['characterDatas']]
    if len(catalogue) != 41 or any(file != 0 or not 21590 <= path <= 21635 for file, path in catalogue):
        raise ValueError('unexpected GameData character catalogue')
    if any(file != 2 for file, path in compendium_catalogue) or {path for file, path in catalogue} != {path for file, path in compendium_catalogue}:
        raise ValueError('Compendium and GameData catalogue sets differ')
    mutant_users=[{'kind':kind,'path_id':r['path_id']} for kind,group in [('game_data',[game]),('ascension',ascensions),('custom_script',customs)]
                  for r in group if (0,21592) in all_pointers(r['data'])]
    return {'schema_version':1,'build_id':BUILD,'source_hashes':{n:ASSET_HASHES[n] for n in ('sharedassets0.assets','globalgamemanagers.assets','level0')},
            'project_context':{'path_id':contexts[0].path_id,'object_sha256':hashlib.sha256(context_raw).hexdigest().upper(),'game_data':game_pointer},
            'mono_script_bindings':{name:mono[name] for name in ('ProjectContext','GameData','AscensionsData','CustomScriptData','Compendium')},
            'game_data':game,'ascensions':ascensions,'custom_scripts':customs,'compendia':compendia,
            'complete_objects':1+1+len(ascensions)+len(customs)+len(compendia),
            'game_data_catalogue_size':len(game['data']['allCharacterData']),
            'compendium_matches_game_data_catalogue_set':True,
            'public_mutant_direct_config_users':mutant_users,
            'scope':'Complete pinned serialized configuration objects and direct references; runtime profile selection, clone/writer effects, dynamic loads and global reachability remain separate',
            'catalogue_correction':'level0 path 139347 is a Compendium object, not the runtime ascension source. The actual chain starts at ProjectContext 138195 -> GameData 21636.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('game_root',type=Path)
    parser.add_argument('--dumper-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.game_root,args.dumper_root)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f"Verified {result['complete_objects']} complete objects, {len(result['ascensions'])} profiles, {len(result['custom_scripts'])} custom scripts and {result['game_data_catalogue_size']} catalogue entries")
