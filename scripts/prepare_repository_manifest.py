"""Record the staged publication boundary; never read credential stores."""
from __future__ import annotations
import hashlib,json,re,subprocess
from pathlib import Path


def digest(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    raw=subprocess.check_output(['git','ls-files','--cached','-z'],cwd=root)
    paths=sorted(set(raw.decode('utf-8').strip('\0').split('\0')))
    own='reproducibility/repository_manifest.json'
    paths=[name for name in paths if name and name!=own]
    files={};tokens=[]
    patterns=[re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),re.compile(rb'github_pat_[A-Za-z0-9_]{30,}'),
        re.compile(rb'AKIA[0-9A-Z]{16}'),re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
    for name in paths:
        path=root/name
        assert path.resolve().is_relative_to(root)
        assert '.venv' not in path.parts and '__pycache__' not in path.parts
        if name.startswith('data/'):
            assert not name.endswith(('.csv','.csv.gz','.npz'))
        assert path.stat().st_size<50*1024**2,f'oversized staged file: {name}'
        content=path.read_bytes()
        if path.suffix in ('.py','.md','.json','.toml','.txt','.yml','.yaml'):
            if any(pattern.search(content) for pattern in patterns):tokens.append(name)
        files[name]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
    if tokens:raise ValueError('possible credential material in staged files: '+', '.join(tokens))
    external=[]
    for base in ('data','results'):
        for path in (root/base).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                name=path.relative_to(root).as_posix()
                if name not in files:external.append({'path':name,'bytes':path.stat().st_size})
    manifest={'project':'Hawkes Criticality Lab','distribution':'hawkes-criticality-lab',
        'repository':'Alexandch/hawkes-criticality-lab','visibility':'private',
        'purpose':'byte-exact compact research snapshot; full raw data and large simulation caches are external',
        'files':files,'external_artifacts':sorted(external,key=lambda x:x['path']),
        'credential_scan':'no recognized GitHub/AWS tokens or private-key blocks in staged text files'}
    (root/own).write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Prepared publication manifest: {len(files)} files, {sum(item["bytes"] for item in files.values())/2**20:.1f} MiB, {len(external)} external artifacts; credential scan passed.')


if __name__=='__main__':main()
