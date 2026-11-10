"""One preregistered extended neural experiment; old holdouts remain sealed."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .confident_calibration import calibrate_confident
from .dynamic_experiment import old_snapshot, reserved_seeds
from .selected_features import generate_selection_batch, load_selection_batch, save_selection_batch, selection_tasks
from .selected_training import score_selected_stream, selection_targets, sha, write_json
from .task_protocol import calibrate_task_threshold, evaluate_task_scores


def fused_features(batch, windows):
    indices = [list(windows).index(value) for value in (max(windows)//4, max(windows))]
    return np.concatenate([batch.matrix[:, i] for i in indices], axis=-1)


def build_model(inputs=128):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(inputs,64),torch.nn.ReLU(),torch.nn.Dropout(.1),
        torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,3))


def train_multitask(train_x, train_ratios, validation_x, validation_ratios, generator, seed, epochs):
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(seed)
    mean,scale=train_x.reshape(-1,train_x.shape[-1]).mean(0),train_x.reshape(-1,train_x.shape[-1]).std(0)
    scale[scale<1e-8]=1.
    model=build_model(train_x.shape[-1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    def tensor_data(x,ratios):
        target=selection_targets(SimpleNamespace(ratios=ratios),generator)
        return (torch.tensor((x-mean)/scale,dtype=torch.float32),
            {task:(torch.tensor(target[task].labels,dtype=torch.float32),torch.tensor(target[task].eligible)) for task in target},
            torch.tensor(ratios[:,2:],dtype=torch.float32))
    tx,ty,tn=tensor_data(train_x,train_ratios)
    vx,vy,vn=tensor_data(validation_x,validation_ratios)
    weights={task:torch.tensor(float((int(mask.sum())-labels[mask].sum())/labels[mask].sum())) for task,(labels,mask) in ty.items()}
    def loss(pred,target,truth):
        total=0.
        for index,task in enumerate(("state","warning")):
            labels,mask=target[task]
            total=total+.5*F.binary_cross_entropy_with_logits(pred[:,:,index][mask],labels[mask],pos_weight=weights[task])
        return total+4*F.mse_loss(torch.sigmoid(pred[:,:,2]),truth)
    best,best_loss,best_epoch=None,np.inf,None
    rows=[]
    for epoch in range(1,epochs+1):
        model.train(); optimizer.zero_grad()
        value=loss(model(tx),ty,tn); value.backward(); optimizer.step()
        val=None
        if epoch%10==0 or epoch==epochs:
            model.eval()
            with torch.no_grad(): val=float(loss(model(vx),vy,vn))
            if val<best_loss:
                best_loss,best_epoch,best=val,epoch,copy.deepcopy(model.state_dict())
        rows.append(dict(epoch=epoch,training_loss=float(value.detach()),validation_loss=val))
    return dict(state_dict=best,mean=mean,scale=scale,inputs=train_x.shape[-1],seed=seed,
        selected_epoch=best_epoch,validation_loss=best_loss,architecture="128-64-ReLU-Dropout0.1-32-ReLU-3",
        heads=["state_logit","warning_logit","current_n_logit"],labels_only_from_simulation=True),pd.DataFrame(rows)


def predict_multitask(x,path):
    import torch
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    if x.shape[-1]!=checkpoint["inputs"] or not np.isfinite(x).all():
        raise ValueError("features do not match checkpoint")
    model=build_model(checkpoint["inputs"]); model.load_state_dict(checkpoint["state_dict"]); model.eval()
    flat=((x-checkpoint["mean"])/checkpoint["scale"]).reshape(-1,x.shape[-1])
    with torch.no_grad():
        predictions=[torch.sigmoid(model(torch.tensor(flat[i:i+8192],dtype=torch.float32))).numpy() for i in range(0,len(flat),8192)]
    return np.concatenate(predictions).reshape(*x.shape[:2],3).astype(float)


def extended_scores(batch,windows,output,seeds):
    x=fused_features(batch,windows)
    members=[predict_multitask(x,Path(output)/"models"/f"multitask_s{seed}.pt") for seed in seeds]
    result={}
    for index,head in enumerate(("state","warning","aux_n_state")):
        for seed,prediction in zip(seeds,members): result[f"{head}__s{seed}"]=prediction[:,:,index]
        result[f"{head}__ensemble"]=np.mean([prediction[:,:,index] for prediction in members],axis=0)
    return result


def assemble_training(fresh,generator,sources):
    x,ratios,ids=[fused_features(fresh,generator["windows"])],[fresh.ratios],[fresh.ids]
    provenance={}
    for source in sources:
        protocol=json.loads((source/"protocol.json").read_text(encoding="utf-8"))
        source_generator=protocol.get("generator",protocol)
        for key in ("prefix_days","stage_days","stages","lookback","forecast_stages","critical_level","rate_min","rate_max"):
            if source_generator[key]!=generator[key]: raise ValueError(f"incompatible source training data: {source}/{key}")
        path=source/"data/train.npz"
        batch=load_selection_batch(path)
        x.append(fused_features(batch,source_generator["windows"]));ratios.append(batch.ratios);ids.append(batch.ids)
        provenance[str(path.resolve())]=sha(path)
    combined_ids=np.concatenate(ids)
    if len(combined_ids)!=len(set(combined_ids)): raise ValueError("training trajectories overlap")
    return np.concatenate(x),np.concatenate(ratios),combined_ids,provenance


def run_extended_training(out="results/extended_neural",new_train_per_family=150,validation_per_family=150,
        calibration_per_family=300,test_per_family=300,epochs=200,model_seeds=(101,113,127),
        data_seed=20261128,test_seed=20261129,workers=4,bootstrap_samples=500,
        prefix_days=730,stage_days=90,stages=12,rate_min=.2,rate_max=50.,sources=None):
    import torch
    torch.set_num_threads(min(4,torch.get_num_threads()))
    output=Path(out)
    sources=[Path(p) for p in (sources if sources is not None else
        ("results/selected_detectors","results/sensitivity_diagnostics","results/dynamic_estimation"))]
    if any(output.resolve()==p.resolve() for p in sources):raise ValueError("need a separate experiment directory")
    if min(new_train_per_family,validation_per_family,test_per_family,epochs,workers,bootstrap_samples)<1 or calibration_per_family<19:
        raise ValueError("invalid dataset sizes")
    generator=dict(prefix_days=prefix_days,stage_days=stage_days,stages=stages,lookback=3,forecast_stages=2,
        critical_level=.9,rate_min=rate_min,rate_max=rate_max,windows=[stage_days,2*stage_days,4*stage_days])
    snapshots=old_snapshot(sources)
    protected=3*(len(model_seeds)+1)
    protocol=dict(stage="9_one_extended_neural_attempt",generator=generator,new_sizes=dict(train=new_train_per_family,
        validation=validation_per_family,calibration=calibration_per_family,test=test_per_family),sources=[str(p.resolve()) for p in sources],
        data_seed=data_seed,test_seed=test_seed,model_seeds=list(model_seeds),epochs=epochs,workers=workers,
        bootstrap_samples=bootstrap_samples,protected_score_streams=protected,
        training="new train plus train-only partitions from R4/R5/R6; no old validation/calibration/holdout labels are used",
        model="causal 90+360 day features, 128-64-32-3; ReLU, dropout0.1, AdamW lr0.001 wd0.001; full batch",
        loss="0.5 weighted BCE state +0.5 weighted BCE warning +4 MSE current n; weights/normalization from train only",
        epoch_selection="minimum joint validation loss, every10 epochs and final epoch; ensemble of all fixed seeds",
        calibration="family confident alpha.05, failure.05 divided among all new heads/members/families; nominal family sensitivity also saved",
        primary="state__ensemble and warning__ensemble, declared before the new holdout; auxiliary n is secondary",
        purpose="one bounded attempt; positive detection quality not assumed; registered crime rows have no true criticality labels",
        old_snapshot_sha256=snapshots,code_sha256={name:sha(Path(__file__).parent/name) for name in
            ("extended_training.py","selected_features.py","simulation.py","fast_window_mle.py","background_models.py","task_protocol.py","confident_calibration.py")})
    output.mkdir(parents=True,exist_ok=True)
    for directory in ("data","models"): (output/directory).mkdir(exist_ok=True)
    if (output/"protocol.json").exists():
        if json.loads((output/"protocol.json").read_text(encoding="utf-8"))!=protocol:raise ValueError("protocol/code changed; use a new output directory")
    else:write_json(output/"protocol.json",protocol)
    if (output/"complete.json").exists():
        saved=json.loads((output/"complete.json").read_text(encoding="utf-8"))
        for name,digest in saved["artifact_sha256"].items():
            if sha(output/name)!=digest:raise ValueError(f"artifact changed: {name}")
        return json.loads((output/"metrics.json").read_text(encoding="utf-8"))
    reserved,batches,hashes=reserved_seeds(sources),{},{}
    for index,split in enumerate(("train","validation","calibration")):
        seed=int(np.random.SeedSequence([data_seed,index]).generate_state(1)[0])
        tasks=selection_tasks(seed,"extended_"+split,protocol["new_sizes"][split],generator,reserved)
        path=output/"data"/f"{split}.npz"
        if path.exists():batch=load_selection_batch(path)
        else:
            batch=generate_selection_batch(tasks,workers);save_selection_batch(batch,path)
        np.testing.assert_array_equal(batch.parameter_seeds,[t[2] for t in tasks]);np.testing.assert_array_equal(batch.simulator_seeds,[t[3] for t in tasks])
        batches[split]=batch;hashes[split]=sha(path);reserved.extend(batch.parameter_seeds.tolist()+batch.simulator_seeds.tolist())
    x,ratios,ids,provenance=assemble_training(batches["train"],generator,sources)
    np.savez_compressed(output/"data/training_features.npz",matrix=x,ratios=ratios,ids=ids)
    vx=fused_features(batches["validation"],generator["windows"])
    for seed in model_seeds:
        path=output/"models"/f"multitask_s{seed}.pt"
        if not path.exists():
            checkpoint,history=train_multitask(x,ratios,vx,batches["validation"].ratios,generator,seed,epochs)
            checkpoint.update(training_trajectories=len(x),source_train_sha256=provenance,
                new_training_sha256=hashes["train"],feature_schema="two_window_multitask_v1")
            torch.save(checkpoint,path);history.to_csv(path.with_suffix(".csv"),index=False)
            print(f"Extended MLP seed={seed}: selected epoch {checkpoint['selected_epoch']}, validation loss {checkpoint['validation_loss']:.4f}",flush=True)
    calibration=extended_scores(batches["calibration"],generator["windows"],output,model_seeds)
    targets=selection_targets(batches["calibration"],generator)
    thresholds,tables={},[]
    for stream,score in calibration.items():
        task="warning" if stream.startswith("warning__") else "state"
        for policy in ("family_confident","family_nominal_sensitivity"):
            cutoff,table=(calibrate_confident(score,targets[task],batches["calibration"].families,.05,.05/protected)
                if policy=="family_confident" else calibrate_task_threshold(score,targets[task],batches["calibration"].families,.05))
            thresholds[stream+"__"+policy]=cutoff;tables.append(table.assign(stream=stream,policy=policy))
    pd.concat(tables,ignore_index=True).to_csv(output/"calibration.csv",index=False)
    np.savez_compressed(output/"data/calibration_scores.npz",**calibration)
    frozen=dict(protocol_sha256=sha(output/"protocol.json"),new_dataset_sha256=hashes,source_train_sha256=provenance,
        training_trajectories=len(x),training_features_sha256=sha(output/"data/training_features.npz"),thresholds=thresholds,
        model_sha256={str(p.relative_to(output)).replace("\\","/"):sha(p) for p in (output/"models").glob("*.pt")},
        test_used_for_selection=False)
    if (output/"frozen_plan.json").exists() and json.loads((output/"frozen_plan.json").read_text(encoding="utf-8"))!=frozen:
        raise ValueError("cannot change frozen plan")
    if (output/"data/test.npz").exists() and not (output/"frozen_plan.json").exists():raise ValueError("existing test lacks original frozen plan")
    write_json(output/"frozen_plan.json",frozen);write_json(output/"test_started.json",dict(frozen_plan_sha256=sha(output/"frozen_plan.json")))
    tasks=selection_tasks(test_seed,"extended_test",test_per_family,generator,reserved)
    path=output/"data/test.npz"
    if path.exists():batch=load_selection_batch(path)
    else:batch=generate_selection_batch(tasks,workers);save_selection_batch(batch,path)
    np.testing.assert_array_equal(batch.parameter_seeds,[t[2] for t in tasks]);np.testing.assert_array_equal(batch.simulator_seeds,[t[3] for t in tasks])
    hashes["test"]=sha(path)
    scores=extended_scores(batch,generator["windows"],output,model_seeds)
    np.savez_compressed(output/"data/test_scores.npz",**scores)
    targets=selection_targets(batch,generator)
    metrics,rows,families,trajectories={ },[],[],[]
    for stream,score in scores.items():
        task="warning" if stream.startswith("warning__") else "state"
        for policy in ("family_confident","family_nominal_sensitivity"):
            name=stream+"__"+policy
            metric,family,trajectory,_=evaluate_task_scores(score,np.ones_like(score,dtype=bool),targets[task],batch.families,batch.ids,batch.rates,
                thresholds[name],task,bootstrap_seed=data_seed+100,bootstrap_samples=bootstrap_samples)
            metrics[name]=metric;rows.append(dict(stream=stream,policy=policy,**{k:v for k,v in metric.items() if not isinstance(v,(list,tuple))}))
            families.append(family.assign(stream=stream,policy=policy));trajectories.append(trajectory.assign(stream=stream,policy=policy))
    # Frozen old primary models evaluated as references on this new test, no new threshold fitting.
    if sources:
        source=sources[0];old_frozen=json.loads((source/"frozen_plan.json").read_text(encoding="utf-8"))
        for original in (entry for entry in old_frozen["streams"] if entry["primary"]):
            entry=dict(original);entry["window_index"]=generator["windows"].index(entry["window_days"])
            score,valid=score_selected_stream(batch,entry,source)
            metric,family,trajectory,_=evaluate_task_scores(score,valid,targets[entry["task"]],batch.families,batch.ids,batch.rates,entry["threshold"],entry["task"],
                bootstrap_seed=data_seed+100,bootstrap_samples=bootstrap_samples)
            stream="old_primary_"+entry["task"];metrics[stream]=metric
            rows.append(dict(stream=stream,policy="frozen_R4_reference",**{k:v for k,v in metric.items() if not isinstance(v,(list,tuple))}))
            families.append(family.assign(stream=stream,policy="frozen_R4_reference"));trajectories.append(trajectory.assign(stream=stream,policy="frozen_R4_reference"))
    write_json(output/"metrics.json",metrics)
    pd.DataFrame(rows).to_csv(output/"operating_metrics.csv",index=False)
    pd.concat(families,ignore_index=True).to_csv(output/"family_metrics.csv",index=False)
    pd.concat(trajectories,ignore_index=True).to_csv(output/"trajectory_metrics.csv",index=False)
    predicted=np.mean([predict_multitask(fused_features(batch,generator["windows"]),output/"models"/f"multitask_s{seed}.pt") for seed in model_seeds],axis=0)[:,:,2]
    error_rows=[]
    for family in np.unique(batch.families):
        use=batch.families==family
        error_rows.append(dict(family=family,trajectories=int(use.sum()),auxiliary_n_mae=float(np.abs(predicted[use]-batch.ratios[use,2:]).mean()),auxiliary_n_bias=float((predicted[use]-batch.ratios[use,2:]).mean())))
    pd.DataFrame(error_rows).to_csv(output/"auxiliary_n_errors.csv",index=False)
    if old_snapshot(sources)!=snapshots:raise RuntimeError("old experiments changed")
    write_extended_report(output)
    write_json(output/"complete.json",dict(dataset_sha256=hashes,old_experiments_unchanged=True,
        artifact_sha256={str(p.relative_to(output)).replace("\\","/"):sha(p) for p in output.rglob("*") if p.is_file() and p.name!="complete.json"}))
    return json.loads(json.dumps(metrics))


def write_extended_report(output):
    protocol=json.loads((output/"protocol.json").read_text(encoding="utf-8"))
    frozen=json.loads((output/"frozen_plan.json").read_text(encoding="utf-8"))
    table=pd.read_csv(output/"operating_metrics.csv")
    lines=["# R9. Единственный расширенный нейросетевой вариант","",
        f"Обучение: {frozen['training_trajectories']:,} целых потоков. Использованы только train-части прежних R4/R5/R6 и новые train-потоки. "
        "Прежние калибровки и финальные выборки в обучение не включены. Валидация, калибровка и финальная проверка новые и независимые.","",
        "Вход — объединение причинных признаков окон 90 и 360 суток, 128 чисел. Совместно обучаются два logit выхода состояния/предупреждения "
        "и вспомогательная оценка текущего n. Истинный n применяется только как обучающая цель. "
        "Архитектура 128→64→32→3, dropout0.1, AdamW; эпоха выбирается по валидационной потере. "
        "Три заранее заданных seed и их среднее, без выбора победителя по финальному тесту.","",
        "Новые реальные строки не размечаются выдуманным n и не используются для обучения критичности. "
        "Симуляционный диапазон остаётся 0.2–50 событий/сутки: добавление городских данных само по себе не устраняет сдвиг распределения.","",
        f"Основная калибровка: цель 5% хотя бы одной ложной тревоги на траекторию отдельно по семействам; совместное доверие 95% для {protocol['protected_score_streams']} новых оценок. "
        "Дополнительно сохранена заранее объявленная номинальная семейная калибровка. Старыe основные модели проверяются на том же новом тесте с прежними порогами как ориентиры.","",
        "| Оценка | Калибровка | Полнота состояния / чистое предупреждение | Макс. семейная ложная тревога | AUC |",
        "|---|---|---:|---:|---:|"]
    primary=table[table.stream.isin(["state__ensemble","warning__ensemble","old_primary_state","old_primary_warning"])]
    for row in primary.itertuples():
        utility=row.window_sensitivity if row.task=="state" else row.clean_event_detection_probability
        lines.append(f"| {row.stream} | {row.policy} | {utility:.2%} | {row.maximum_family_false_alarm_probability:.2%} | {row.auc_valid_only:.3f} |")
    lines += ["","Числа разных задач имеют разные знаменатели. Первое предупреждение должно предшествовать первому переходу, попадать в горизонт 180 суток "
        "и не иметь предыдущей ложной тревоги. Увеличение AUC или количества обучения не считается доказательством надёжного раннего предупреждения. "
        "Все отдельные seed, вспомогательная голова и интервалы сохранены, включая отрицательные результаты.","",
        "После этого опыта новый подбор по его финальной выборке не проводится. Прикладная проверка CODE/городских данных является применением "
        "без истинных меток критичности, поэтому там не вычисляются accuracy, TPR или ложные тревоги относительно неизвестного n.","",
        "```powershell",".\\.venv\\Scripts\\python.exe -m critical_processes.extended_training",".\\.venv\\Scripts\\python.exe scripts/verify_extended_training.py","```",""]
    (output/"analysis.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="results/extended_neural")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args();run_extended_training(**vars(args))


if __name__=="__main__":main()
