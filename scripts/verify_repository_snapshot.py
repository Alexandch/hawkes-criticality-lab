"""Verify a compact GitHub checkout without claiming absent raw-data checks."""
from __future__ import annotations
import csv,hashlib,json,re
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
    for folder in ('dynamic_estimation','extended_neural','empirical_transfer','spatiotemporal_forecast','offline_readiness'):
        protocol=json.loads((root/'results'/folder/'protocol.json').read_text(encoding='utf-8'))
        for name,value in protocol['code_sha256'].items():assert digest(root/'src/critical_processes'/name)==value,(folder,name)
    metrics=json.loads((root/'results/extended_neural/metrics.json').read_text(encoding='utf-8'))
    assert metrics['warning__ensemble__family_confident']['detected_trajectories']==5
    assert metrics['warning__ensemble__family_confident']['positive_trajectories']==600
    assert metrics['warning__ensemble__family_nominal_sensitivity']['detected_trajectories']==27
    tests=json.loads((root/'reproducibility/test_result.json').read_text(encoding='utf-8'))
    assert tests['successful'] and tests['tests_run']==99 and tests['failures']==tests['errors']==0
    spatial=root/'results/spatiotemporal_forecast'
    complete=json.loads((spatial/'complete.json').read_text(encoding='utf-8'))
    for name,value in complete['artifact_sha256'].items():
        relative=(spatial/name).relative_to(root).as_posix()
        if relative in manifest['files']:assert digest(spatial/name)==value,name
        else:assert relative in external,name
    with (spatial/'test_metrics.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==40
    main_rows={r['city']:r for r in rows if r['information_lag_days']=='7' and r['split']=='test2026' and r['method']=='neural_ensemble'}
    assert abs(float(main_rows['chicago']['mae'])-1.8578675762290284)<1e-12
    assert abs(float(main_rows['san_francisco']['mae'])-1.4367698107119444)<1e-12
    summary=root/'results/spatiotemporal_summary'
    provenance=json.loads((summary/'provenance.json').read_text(encoding='utf-8'))
    for name,value in provenance['source_sha256'].items():assert digest(root/name)==value,name
    for name,value in provenance['artifact_sha256'].items():assert digest(summary/name)==value,name
    offline=root/'results/offline_readiness'
    finished=json.loads((offline/'complete.json').read_text(encoding='utf-8'))
    for name,value in finished['artifact_sha256'].items():
        relative=(offline/name).relative_to(root).as_posix()
        if relative in manifest['files']:assert digest(offline/name)==value,name
        else:assert relative in external,name
    readiness=json.loads((offline/'readiness.json').read_text(encoding='utf-8'))
    assert not readiness['real_pilot_performed'] and not readiness['historical_publication_times_verified']
    derived=root/'results/offline_summary'
    provenance=json.loads((derived/'provenance.json').read_text(encoding='utf-8'))
    for name,value in provenance['source_sha256'].items():assert digest(root/name)==value,name
    for name,value in provenance['artifact_sha256'].items():assert digest(derived/name)==value,name
    print(f'Verified {len(manifest["files"])} published files, {len(assets["assets"])} figures, {links} documentation links and frozen source hashes.')
    print(f'External artifacts intentionally not part of Git: {len(external)}; missing documentation targets declared external: {len(missing_links)}. Full raw-data verification was not run.')


if __name__=='__main__':main()
