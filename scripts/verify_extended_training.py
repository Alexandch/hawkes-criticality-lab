from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from critical_processes.confident_calibration import calibrate_confident
from critical_processes.dynamic_experiment import reserved_seeds
from critical_processes.extended_training import assemble_training,extended_scores,fused_features
from critical_processes.selected_features import load_selection_batch,selection_tasks,simulate_selection_trajectory,build_selection_features
from critical_processes.selected_training import selection_targets,sha,score_selected_stream
from critical_processes.task_protocol import calibrate_task_threshold,evaluate_task_scores


def main():
    import torch
    torch.set_num_threads(min(4,torch.get_num_threads()))
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('results/extended_neural'));args=parser.parse_args();out=args.out
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    frozen=json.loads((out/'frozen_plan.json').read_text(encoding='utf-8'))
    complete=json.loads((out/'complete.json').read_text(encoding='utf-8'))
    metrics=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    for name,digest in complete['artifact_sha256'].items():assert sha(out/name)==digest,name
    for name,digest in protocol['old_snapshot_sha256'].items():assert sha(name)==digest,name
    module=Path(__file__).resolve().parents[1]/'src/critical_processes'
    for name,digest in protocol['code_sha256'].items():assert sha(module/name)==digest,name
    assert frozen['protocol_sha256']==sha(out/'protocol.json')
    assert not frozen['test_used_for_selection']
    assert json.loads((out/'test_started.json').read_text(encoding='utf-8'))['frozen_plan_sha256']==sha(out/'frozen_plan.json')
    sources=[Path(p) for p in protocol['sources']]
    used=reserved_seeds(sources);old_used=set(used);batches={};tasks_saved={};new_seeds=[]
    for index,(split,size) in enumerate(protocol['new_sizes'].items()):
        seed=protocol['test_seed'] if split=='test' else int(np.random.SeedSequence([protocol['data_seed'],index]).generate_state(1)[0])
        tasks=selection_tasks(seed,'extended_'+split,size,protocol['generator'],used)
        batch=load_selection_batch(out/'data'/f'{split}.npz')
        np.testing.assert_array_equal(batch.ids,[t[0] for t in tasks])
        np.testing.assert_array_equal(batch.parameter_seeds,[t[2] for t in tasks]);np.testing.assert_array_equal(batch.simulator_seeds,[t[3] for t in tasks])
        assert sha(out/'data'/f'{split}.npz')==complete['dataset_sha256'][split]
        values=batch.parameter_seeds.tolist()+batch.simulator_seeds.tolist();used.extend(values);new_seeds.extend(values)
        batches[split]=batch;tasks_saved[split]=tasks
    assert len(set(new_seeds))==len(new_seeds) and not set(new_seeds).intersection(old_used)
    x,ratios,ids,provenance=assemble_training(batches['train'],protocol['generator'],sources)
    assert provenance==frozen['source_train_sha256'] and len(x)==frozen['training_trajectories']
    with np.load(out/'data/training_features.npz') as saved:
        np.testing.assert_array_equal(saved['matrix'],x);np.testing.assert_array_equal(saved['ratios'],ratios);np.testing.assert_array_equal(saved['ids'],ids)
    flat=x.reshape(-1,128);mean,scale=flat.mean(0),flat.std(0);scale[scale<1e-8]=1.
    for seed in protocol['model_seeds']:
        checkpoint=torch.load(out/'models'/f'multitask_s{seed}.pt',map_location='cpu',weights_only=False)
        np.testing.assert_allclose(checkpoint['mean'],mean,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(checkpoint['scale'],scale,rtol=1e-12,atol=1e-12)
        history=pd.read_csv(out/'models'/f'multitask_s{seed}.csv')
        chosen=history.dropna(subset=['validation_loss']).sort_values(['validation_loss','epoch']).iloc[0]
        assert checkpoint['selected_epoch']==chosen.epoch
    scores={};targets={}
    for split in ('calibration','test'):
        scores[split]=extended_scores(batches[split],protocol['generator']['windows'],out,protocol['model_seeds'])
        targets[split]=selection_targets(batches[split],protocol['generator'])
        with np.load(out/'data'/f'{split}_scores.npz') as saved:
            for key,value in scores[split].items():np.testing.assert_array_equal(saved[key],value)
    batch=batches['test']
    for stream,score in scores['test'].items():
        task='warning' if stream.startswith('warning__') else 'state'
        for policy in ('family_confident','family_nominal_sensitivity'):
            cutoff,_=(calibrate_confident(scores['calibration'][stream],targets['calibration'][task],batches['calibration'].families,.05,.05/protocol['protected_score_streams'])
                if policy=='family_confident' else calibrate_task_threshold(scores['calibration'][stream],targets['calibration'][task],batches['calibration'].families,.05))
            key=stream+'__'+policy;assert cutoff==frozen['thresholds'][key]
            result,*_=evaluate_task_scores(score,np.ones_like(score,dtype=bool),targets['test'][task],batch.families,batch.ids,batch.rates,cutoff,task,
                bootstrap_seed=protocol['data_seed']+100,bootstrap_samples=protocol['bootstrap_samples'])
            assert json.loads(json.dumps(result))==metrics[key],key
    if sources:
        old=json.loads((sources[0]/'frozen_plan.json').read_text(encoding='utf-8'))
        for original in (entry for entry in old['streams'] if entry['primary']):
            entry=dict(original);entry['window_index']=protocol['generator']['windows'].index(entry['window_days'])
            score,valid=score_selected_stream(batch,entry,sources[0]);task=entry['task']
            result,*_=evaluate_task_scores(score,valid,targets['test'][task],batch.families,batch.ids,batch.rates,entry['threshold'],task,
                bootstrap_seed=protocol['data_seed']+100,bootstrap_samples=protocol['bootstrap_samples'])
            assert json.loads(json.dumps(result))==metrics['old_primary_'+task]
    replays=0
    for split in ('train','test'):
        batch=batches[split]
        for family in np.unique(batch.families):
            index=np.flatnonzero(batch.families==family)[0]
            counts,events,truth,rate=simulate_selection_trajectory(tasks_saved[split][index])
            matrix,fits=build_selection_features(counts,events,protocol['generator']['windows'],protocol['generator']['prefix_days'],protocol['generator']['stage_days'])
            np.testing.assert_array_equal(counts,batch.counts[index]);np.testing.assert_array_equal(truth,batch.ratios[index]);assert rate==batch.rates[index]
            np.testing.assert_allclose(matrix,batch.matrix[index],rtol=1e-10,atol=1e-10);np.testing.assert_allclose(fits,batch.fits[index],rtol=1e-10,atol=1e-10);replays+=1
    print(f'Verified {len(x)} train-only trajectories, {sum(protocol["new_sizes"].values())*13} new streams, all {len(metrics)} detector rules, validation epochs, source integrity and {replays} exact replays.')


if __name__=='__main__':main()
