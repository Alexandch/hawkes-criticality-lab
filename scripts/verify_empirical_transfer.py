from __future__ import annotations
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from critical_processes.empirical_transfer import block_intervals,prepare_block,describe_block
from critical_processes.extended_training import extended_scores,fused_features
from critical_processes.selected_features import build_selection_features
from critical_processes.selected_training import score_selected_stream,sha


def main():
    import torch
    torch.set_num_threads(min(4,torch.get_num_threads()))
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('results/empirical_transfer'));args=parser.parse_args();out=args.out
    panel=Path('data/processed/expanded_20261110');raw=Path('data/raw/expanded_20261110')
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'));complete=json.loads((out/'complete.json').read_text(encoding='utf-8'))
    for name,digest in complete['artifact_sha256'].items():assert sha(out/name)==digest,name
    for name,digest in complete['prepared_source_sha256'].items():assert sha(name)==digest,name
    for name,digest in complete['model_sha256'].items():assert sha(name)==digest,name
    module=Path(__file__).resolve().parents[1]/'src/critical_processes'
    for name,digest in protocol['code_sha256'].items():assert sha(module/name)==digest,name
    manifests=json.loads((panel/'panel_manifest.json').read_text(encoding='utf-8'))['streams']
    assert sha(panel/'panel_manifest.json')==protocol['panel_manifest_sha256']
    old_root,new_root=Path('results/selected_detectors'),Path('results/extended_neural')
    old=json.loads((old_root/'frozen_plan.json').read_text(encoding='utf-8'));new=json.loads((new_root/'frozen_plan.json').read_text(encoding='utf-8'))
    new_protocol=json.loads((new_root/'protocol.json').read_text(encoding='utf-8'))
    assert sha(old_root/'frozen_plan.json')==protocol['old_plan_sha256'];assert sha(new_root/'frozen_plan.json')==protocol['new_plan_sha256']
    downloads=json.loads((raw/'download_summary.json').read_text(encoding='utf-8'));assert not downloads['errors']
    hashes={}
    for record in downloads['records']:
        if 'city' in record:
            year=int(record['start'][:4]);path=raw/'cities'/f"{record['city']}_{year}.csv.gz"
            assert record['rows']==record['count_before']==record['count_after']
            actual=sum(len(chunk) for chunk in pd.read_csv(path,chunksize=250000))
            assert actual==record['rows']
        else:path=raw/record['name']
        assert sha(path)==record['sha256'];hashes[str(path)]=record['sha256']
    with np.load(new_root/'data/training_features.npz') as data:
        reference=np.quantile(data['matrix'].reshape(-1,128),[.005,.995],axis=0)
    lower,upper=reference
    saved=pd.read_csv(out/'decisions.csv');blocks=pd.read_csv(out/'blocks.csv');skipped=json.loads((out/'skipped.json').read_text(encoding='utf-8'))
    assert saved.true_n.isna().all() and saved.true_transition.isna().all()
    checked=0
    for manifest in manifests:
        source=manifest['source'];stem='code_'+source.split(':',1)[1].lower().replace(' ','_') if source.startswith('CODE') else 'portal_'+source.split(':',1)[1]
        csv_path=panel/(stem+'.csv.gz');npz_path=panel/(stem+'.csv.npz')
        assert sha(csv_path)==manifest['output_sha256'];assert sha(npz_path)==manifest['events_npz_sha256']
        for name,digest in manifest['source_files_sha256'].items():
            if name not in hashes:hashes[name]=sha(name)
            assert hashes[name]==digest
        normalized=pd.read_csv(csv_path,dtype={'event_id':str})
        with np.load(npz_path) as data:
            ns=data['event_ns'];ids=data['event_ids']
        np.testing.assert_array_equal(pd.to_datetime(normalized.event_time,format='mixed').to_numpy(dtype='datetime64[ns]').astype(np.int64),ns)
        np.testing.assert_array_equal(normalized.event_id.to_numpy(dtype=str),ids)
        assert len(ns)==manifest['events'] and normalized.event_id.nunique()==len(ns) and (np.diff(ns)>=0).all()
        if source.startswith('CODE'):assert normalized.event_id.str.match(r'^20(18|19|20|21|22):').all()
        if not manifest['eligible_for_full_history']:
            assert source in [item['source'] for item in skipped];continue
        for tag,start,end in block_intervals(manifest):
            key=stem+'_'+tag;events,counts=prepare_block(ns,start,end)
            matrix,fits=build_selection_features(counts,events,[90,180,360],730,90,3,2)
            with np.load(out/'data'/(key+'.npz')) as data:
                np.testing.assert_array_equal(counts,data['counts']);np.testing.assert_allclose(matrix,data['matrix'],rtol=1e-10,atol=1e-10);np.testing.assert_allclose(fits,data['fits'],rtol=1e-10,atol=1e-10)
            batch=SimpleNamespace(matrix=matrix[None],fits=fits[None]);predicted=extended_scores(batch,[90,180,360],new_root,new_protocol['model_seeds'])
            decisions=saved[saved.block_id==key].reset_index(drop=True);assert len(decisions)==10
            x=fused_features(batch,[90,180,360])[0];outside=((x<lower)|(x>upper)).mean(-1)
            np.testing.assert_allclose(outside,decisions.fraction_features_outside_training_central99pct,rtol=1e-12,atol=1e-12)
            for task in ('state','warning'):
                # Float32 BLAS may differ by a few ULP across thread counts.
                score=predicted[task+'__ensemble'][0];np.testing.assert_allclose(score,decisions['new_'+task+'_score'],rtol=1e-6,atol=2e-7)
                np.testing.assert_array_equal(score>=new['thresholds'][task+'__ensemble__family_confident'],decisions['new_'+task+'_alarm'])
            for original in (entry for entry in old['streams'] if entry['primary']):
                entry=dict(original);entry['window_index']=[90,180,360].index(entry['window_days']);score,valid=score_selected_stream(batch,entry,old_root)
                np.testing.assert_allclose(score[0],decisions['old_'+entry['task']+'_score'],rtol=1e-6,atol=2e-7)
                np.testing.assert_array_equal(valid[0]&(score[0]>=entry['threshold']),decisions['old_'+entry['task']+'_alarm'])
            meta=blocks[blocks.block_id==key].iloc[0];assert meta.events==len(events)
            np.testing.assert_allclose(meta.mean_daily_rate,counts.mean(),atol=1e-10)
            assert decisions.future180days_observed.sum()==8
            for model in ('old','new'):
                assert meta[model+'_state_threshold_crossings']==decisions[model+'_state_alarm'].sum()
                assert meta[model+'_warning_threshold_crossings_full_horizon']==decisions.loc[decisions.future180days_observed,model+'_warning_alarm'].sum()
            checked+=1
    assert checked==len(blocks)
    print(f'Verified {len(downloads["records"])} source downloads, {len(manifests)} prepared streams, {checked} real-data feature replays and {len(saved)} causal decisions; absent criticality labels retained, no accuracy invented.')


if __name__=='__main__':main()
