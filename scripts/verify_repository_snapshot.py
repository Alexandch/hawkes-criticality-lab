"""Verify a compact GitHub checkout without claiming absent raw-data checks."""
from __future__ import annotations
import csv,hashlib,json,re
from pathlib import Path
from urllib.parse import unquote
from critical_processes.documentation_archive import archive_index,read_artifact,verify_archive


def digest(path):
    return hashlib.sha256(read_artifact(path)).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'reproducibility/repository_manifest.json').read_text(encoding='utf-8'))
    assert manifest['project']=='Hawkes Criticality Lab'
    assert [name for name in manifest['files'] if name.lower().endswith('.md')]==['README.md']
    archived=archive_index(root)['files']
    assert verify_archive(root)==37
    for name,record in manifest['files'].items():
        path=root/name
        assert path.resolve().is_relative_to(root) and path.is_file(),name
        assert path.stat().st_size==record['bytes'] and digest(path)==record['sha256'],name
    external={item['path'] for item in manifest['external_artifacts']}
    missing_links=[];links=0
    for document in [root/'README.md']:
        content=document.read_text(encoding='utf-8')
        assert content.count('```')%2==0,document.name
        targets=re.findall(r'\[[^\]]*\]\(([^)]+)\)',content)+re.findall(r'<img\s+src="([^"]+)"',content)
        for match in targets:
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
    for folder in ('dynamic_estimation','extended_neural','empirical_transfer','spatiotemporal_forecast','offline_readiness','expanded_district_detector','adaptive_seattle_detector','delay_forecast_detector','selective_austin_detector','cad_context_detector'):
        protocol=json.loads((root/'results'/folder/'protocol.json').read_text(encoding='utf-8'))
        for name,value in protocol['code_sha256'].items():assert digest(root/'src/critical_processes'/name)==value,(folder,name)
    metrics=json.loads((root/'results/extended_neural/metrics.json').read_text(encoding='utf-8'))
    assert metrics['warning__ensemble__family_confident']['detected_trajectories']==5
    assert metrics['warning__ensemble__family_confident']['positive_trajectories']==600
    assert metrics['warning__ensemble__family_nominal_sensitivity']['detected_trajectories']==27
    tests=json.loads((root/'reproducibility/test_result.json').read_text(encoding='utf-8'))
    assert tests['successful'] and tests['tests_run']==139 and tests['failures']==tests['errors']==0
    spatial=root/'results/spatiotemporal_forecast'
    complete=json.loads((spatial/'complete.json').read_text(encoding='utf-8'))
    for name,value in complete['artifact_sha256'].items():
        relative=(spatial/name).relative_to(root).as_posix()
        if relative in manifest['files'] or relative in archived:assert digest(spatial/name)==value,name
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
        if relative in manifest['files'] or relative in archived:assert digest(offline/name)==value,name
        else:assert relative in external,name
    readiness=json.loads((offline/'readiness.json').read_text(encoding='utf-8'))
    assert not readiness['real_pilot_performed'] and not readiness['historical_publication_times_verified']
    derived=root/'results/offline_summary'
    provenance=json.loads((derived/'provenance.json').read_text(encoding='utf-8'))
    for name,value in provenance['source_sha256'].items():assert digest(root/name)==value,name
    for name,value in provenance['artifact_sha256'].items():assert digest(derived/name)==value,name
    expanded=root/'results/expanded_district_detector'
    for name,value in json.loads((expanded/'complete.json').read_text(encoding='utf-8'))['artifact_sha256'].items():
        relative=(expanded/name).relative_to(root).as_posix()
        if relative in manifest['files']:assert digest(expanded/name)==value,name
        else:assert relative in external,name
    check=json.loads((expanded/'verification.json').read_text(encoding='utf-8'))
    assert check['successful'] and sum(check['districts'].values())==114
    figure_manifest=json.loads((root/'report/assets/expanded_manifest.json').read_text(encoding='utf-8'))
    for name,value in figure_manifest['source_sha256'].items():assert digest(root/name)==value
    for item in figure_manifest['assets']:assert digest(root/item['file'])==item['sha256']
    pdf_build=json.loads((root/'output/pdf/r12_build.json').read_text(encoding='utf-8'))
    assert pdf_build['total_pages']==53 and pdf_build['base_pages']==48
    assert digest(root/'scripts/append_expanded_pdf.py')==pdf_build['builder_sha256']
    for name,value in pdf_build['source_sha256'].items():assert digest(root/name)==value
    qa=json.loads((root/'output/pdf/r12_qa.json').read_text(encoding='utf-8'))
    assert qa['successful'] and qa['sha256']==pdf_build['output_sha256']
    adaptive=root/'results/adaptive_seattle_detector'
    for name,value in json.loads((adaptive/'complete.json').read_text(encoding='utf-8'))['artifact_sha256'].items():
        relative=(adaptive/name).relative_to(root).as_posix()
        if relative in manifest['files']:assert digest(adaptive/name)==value,name
        else:assert relative in external,name
    check=json.loads((adaptive/'verification.json').read_text(encoding='utf-8'))
    assert check['successful'] and check['development_candidates']==27 and check['new_october_cases']==3564
    graphs=json.loads((root/'report/assets/adaptive_manifest.json').read_text(encoding='utf-8'))
    for name,value in graphs['source_sha256'].items():assert digest(root/name)==value
    for item in graphs['assets']:assert digest(root/item['file'])==item['sha256']
    last_pdf=json.loads((root/'output/pdf/r13_build.json').read_text(encoding='utf-8'))
    assert last_pdf['base_sha256']==pdf_build['output_sha256']
    assert last_pdf['base_pages']==53 and last_pdf['total_pages']==57
    assert digest(root/'scripts/append_adaptive_pdf.py')==last_pdf['builder_sha256']
    for name,value in last_pdf['source_sha256'].items():assert digest(root/name)==value
    last_qa=json.loads((root/'output/pdf/r13_qa.json').read_text(encoding='utf-8'))
    assert last_qa['successful'] and last_qa['sha256']==last_pdf['output_sha256']
    delay=root/'results/delay_forecast_detector'
    for name,value in json.loads((delay/'complete.json').read_text(encoding='utf-8'))['artifact_sha256'].items():
        relative=(delay/name).relative_to(root).as_posix()
        if relative in manifest['files']:assert digest(delay/name)==value,name
        else:assert relative in external,name
    protocol=json.loads((delay/'protocol.json').read_text(encoding='utf-8'))
    assert digest(root/'scripts/prepare_delay_forecast_data.py')==protocol['preparation_sha256']
    check=json.loads((delay/'verification.json').read_text(encoding='utf-8'))
    assert check['successful'] and check['development_candidates']==12 and check['new_test_cases']==33233 and check['models']==9
    assert digest(root/'scripts/verify_delay_forecast.py')==check['verifier_sha256']
    with (delay/'metrics.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==30 and not any(r['full_screen_passed']=='True' for r in rows)
    row=next(r for r in rows if r['scenario']=='primary' and r['role']=='selected' and r['group']=='all')
    assert row['true_positives']=='290' and row['positives']=='1497' and row['false_positives']=='1527'
    graphs=json.loads((root/'report/assets/delay_forecast_manifest.json').read_text(encoding='utf-8'))
    assert digest(root/'scripts/summarize_delay_forecast.py')==graphs['renderer_sha256']
    for name,value in graphs['source_sha256'].items():assert digest(root/name)==value
    for item in graphs['assets']:assert digest(root/item['file'])==item['sha256']
    current_pdf=json.loads((root/'output/pdf/r14_build.json').read_text(encoding='utf-8'))
    assert current_pdf['base_sha256']==last_pdf['output_sha256']
    assert current_pdf['base_pages']==57 and current_pdf['total_pages']==62
    assert digest(root/'scripts/append_delay_pdf.py')==current_pdf['builder_sha256']
    for name,value in current_pdf['source_sha256'].items():assert digest(root/name)==value
    current_qa=json.loads((root/'output/pdf/r14_qa.json').read_text(encoding='utf-8'))
    assert current_qa['successful'] and current_qa['sha256']==current_pdf['output_sha256']
    selective=root/'results/selective_austin_detector'
    for name,value in json.loads((selective/'complete.json').read_text(encoding='utf-8'))['artifact_sha256'].items():
        relative=(selective/name).relative_to(root).as_posix()
        if relative in manifest['files']:assert digest(selective/name)==value,name
        else:assert relative in external,name
    protocol=json.loads((selective/'protocol.json').read_text(encoding='utf-8'))
    assert digest(root/'scripts/prepare_selective_data.py')==protocol['preparation_sha256']
    assert digest(root/'scripts/prepare_delay_forecast_data.py')==protocol['r14_preparation_sha256']
    check=json.loads((selective/'verification.json').read_text(encoding='utf-8'))
    assert check['successful'] and check['development_candidates']==6 and check['new_test_cases']==19713
    assert digest(root/'scripts/verify_selective_alarm.py')==check['verifier_sha256']
    with (selective/'metrics.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==30 and not any(r['full_screen_passed']=='True' for r in rows)
    row=next(r for r in rows if r['scenario']=='primary' and r['role']=='selected' and r['group']=='all')
    assert row['true_positives']=='48' and row['positives']=='935' and row['false_positives']=='172'
    graphs=json.loads((root/'report/assets/selective_manifest.json').read_text(encoding='utf-8'))
    assert digest(root/'scripts/summarize_selective_alarm.py')==graphs['renderer_sha256']
    for name,value in graphs['source_sha256'].items():assert digest(root/name)==value
    for item in graphs['assets']:assert digest(root/item['file'])==item['sha256']
    newest_pdf=json.loads((root/'output/pdf/r15_build.json').read_text(encoding='utf-8'))
    assert newest_pdf['base_sha256']==current_pdf['output_sha256']
    assert newest_pdf['base_pages']==62 and newest_pdf['total_pages']==66
    assert digest(root/'scripts/append_selective_pdf.py')==newest_pdf['builder_sha256']
    for name,value in newest_pdf['source_sha256'].items():assert digest(root/name)==value
    newest_qa=json.loads((root/'output/pdf/r15_qa.json').read_text(encoding='utf-8'))
    assert newest_qa['successful'] and newest_qa['sha256']==newest_pdf['output_sha256']
    cad=root/'results/cad_context_detector'
    for name,value in json.loads((cad/'complete.json').read_text(encoding='utf-8'))['artifact_sha256'].items():
        relative=(cad/name).relative_to(root).as_posix()
        if relative in manifest['files']:assert digest(cad/name)==value,name
        else:assert relative in external,name
    protocol=json.loads((cad/'protocol.json').read_text(encoding='utf-8'))
    assert protocol['independent_test'] is False and digest(root/'scripts/prepare_cad_data.py')==protocol['preparation_sha256']
    check=json.loads((cad/'verification.json').read_text(encoding='utf-8'))
    assert check['successful'] and check['models']==6 and check['included_cad_incidents']==1614776 and check['independent_test'] is False
    assert digest(root/'scripts/verify_cad_context.py')==check['verifier_sha256']
    with (cad/'metrics.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==36 and all(r['independent_test']=='False' and r['full_screen_passed']=='False' for r in rows)
    row=next(r for r in rows if r['split']=='reused2026' and r['scenario']=='primary' and r['method']=='cad_nb' and r['group']=='all')
    assert row['true_positives']=='189' and row['false_positives']=='1058' and row['positives']=='935'
    graphs=json.loads((root/'report/assets/cad_manifest.json').read_text(encoding='utf-8'))
    assert digest(root/'scripts/summarize_cad_context.py')==graphs['renderer_sha256']
    for name,value in graphs['source_sha256'].items():assert digest(root/name)==value
    for item in graphs['assets']:assert digest(root/item['file'])==item['sha256']
    cad_pdf=json.loads((root/'output/pdf/r16_build.json').read_text(encoding='utf-8'))
    assert cad_pdf['base_sha256']==newest_pdf['output_sha256']
    assert cad_pdf['base_pages']==66 and cad_pdf['total_pages']==70
    assert digest(root/'output/pdf/hawkes_criticality_report.pdf')==cad_pdf['output_sha256']
    assert digest(root/'scripts/append_cad_pdf.py')==cad_pdf['builder_sha256']
    for name,value in cad_pdf['source_sha256'].items():assert digest(root/name)==value
    cad_qa=json.loads((root/'output/pdf/r16_qa.json').read_text(encoding='utf-8'))
    assert cad_qa['successful'] and cad_qa['sha256']==cad_pdf['output_sha256']
    print(f'Verified {len(manifest["files"])} published files, {len(assets["assets"])+10} figures, {links} documentation links and frozen source hashes.')
    print(f'External artifacts intentionally not part of Git: {len(external)}; missing documentation targets declared external: {len(missing_links)}. Full raw-data verification was not run.')


if __name__=='__main__':main()
