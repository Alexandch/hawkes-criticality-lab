"""Verify a compact GitHub checkout without claiming absent raw-data checks."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
from urllib.parse import unquote


def digest(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'reproducibility/repository_manifest.json').read_text(encoding='utf-8'))
    assert manifest['project']=='Hawkes Criticality Lab'
    for name,record in manifest['files'].items():
        path=root/name
        assert path.resolve().is_relative_to(root) and path.is_file(),name
        assert path.stat().st_size==record['bytes'] and digest(path)==record['sha256'],name
    external={item['path'] for item in manifest['external_artifacts']}
    missing_links=[];links=0
    for document in [root/'README.md',*(root/'report').glob('*.md')]:
        content=document.read_text(encoding='utf-8')
        assert content.count('```')%2==0,document.name
        for match in re.findall(r'\[[^\]]*\]\(([^)]+)\)',content):
            match=match.strip('<>')
            if re.match(r'[a-z]+://',match) or match.startswith('#'):continue
            path=(document.parent/unquote(match.split('#',1)[0])).resolve()
            assert path.is_relative_to(root),match
            if not path.exists():
                relative=path.relative_to(root).as_posix()
                assert relative in external,f'broken link: {document.name}: {match}'
                missing_links.append(relative)
            links+=1
    assets=json.loads((root/'report/assets/manifest.json').read_text(encoding='utf-8'))
    for item in assets['assets']:assert digest(root/item['source'])==digest(root/item['copy'])==item['sha256']
    for folder in ('dynamic_estimation','extended_neural','empirical_transfer'):
        protocol=json.loads((root/'results'/folder/'protocol.json').read_text(encoding='utf-8'))
        for name,value in protocol['code_sha256'].items():assert digest(root/'src/critical_processes'/name)==value,(folder,name)
    metrics=json.loads((root/'results/extended_neural/metrics.json').read_text(encoding='utf-8'))
    assert metrics['warning__ensemble__family_confident']['detected_trajectories']==5
    assert metrics['warning__ensemble__family_confident']['positive_trajectories']==600
    assert metrics['warning__ensemble__family_nominal_sensitivity']['detected_trajectories']==27
    tests=json.loads((root/'reproducibility/test_result.json').read_text(encoding='utf-8'))
    assert tests['successful'] and tests['tests_run']==82
    print(f'Verified {len(manifest["files"])} published files, {len(assets["assets"])} figures, {links} documentation links and frozen source hashes.')
    print(f'External artifacts intentionally not part of Git: {len(external)}; missing documentation targets declared external: {len(missing_links)}. Full raw-data verification was not run.')


if __name__=='__main__':main()
