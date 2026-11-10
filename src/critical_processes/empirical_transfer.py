"""Frozen detector application to real records without invented criticality labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .extended_training import extended_scores, fused_features
from .selected_features import build_selection_features
from .selected_training import score_selected_stream, sha, write_json


def block_intervals(manifest):
    start,end=pd.Timestamp(manifest["start"]),pd.Timestamp(manifest["end_exclusive"])
    horizon=pd.Timedelta(days=730+12*90)
    if manifest["source"].startswith("CODE"):
        return [("archive",start,start+horizon)] if start+horizon<=end else []
    return [("early",start,start+horizon),("latest",end-horizon,end)]


def prepare_block(event_ns,start,end):
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    ns=np.asarray(event_ns,dtype=np.int64)
    selected=ns[(ns>=start.value)&(ns<end.value)]
    events=(selected-start.value)/86400e9
    counts=np.bincount(np.floor(events).astype(int),minlength=(end-start).days)
    if len(counts)!=(end-start).days:raise ValueError("invalid event bounds")
    return events,counts


def describe_block(manifest,events,counts,start,end,tag):
    return dict(source=manifest["source"],period=tag,start=str(start.date()),end_exclusive=str(end.date()),
        history_days=730,monitored_days=1080,events=len(events),mean_daily_rate=float(counts.mean()),
        unique_timestamps=int(len(np.unique(events))),tied_fraction=float(1-len(np.unique(events))/len(events)),
        unit=manifest["unit"],true_criticality_labels_available=False,
        note="observational application only; rate/precision/background/unit may differ from simulation; overlapping periods are not independent")


def run_transfer(panel="data/processed/expanded_20261110",old_model="results/selected_detectors",new_model="results/extended_neural",out="results/empirical_transfer"):
    panel,old_model,new_model,out=map(Path,(panel,old_model,new_model,out))
    if not (new_model/"complete.json").exists():raise ValueError("finish and freeze the new simulation experiment before empirical application")
    old=json.loads((old_model/"frozen_plan.json").read_text(encoding="utf-8"))
    new=json.loads((new_model/"frozen_plan.json").read_text(encoding="utf-8"))
    new_protocol=json.loads((new_model/"protocol.json").read_text(encoding="utf-8"))
    for model_root,plan in ((old_model,old),(new_model,new)):
        for name,digest in plan["model_sha256"].items():
            if sha(model_root/name)!=digest:raise ValueError(f"frozen model changed: {model_root/name}")
    manifests=json.loads((panel/"panel_manifest.json").read_text(encoding="utf-8"))["streams"]
    code_names=("empirical_transfer.py","selected_features.py","extended_training.py","background_models.py","fast_window_mle.py","window_features.py")
    protocol=dict(stage="7_frozen_real_application",panel_manifest_sha256=sha(panel/"panel_manifest.json"),
        old_plan_sha256=sha(old_model/"frozen_plan.json"),new_plan_sha256=sha(new_model/"frozen_plan.json"),
        windows=[90,180,360],prefix_days=730,stage_days=90,stages=12,lookback=3,forecast_stages=2,
        period_rule="CODE first complete1810 days, portals first and last complete1810 days; overlaps explicitly dependent",
        warning_rule="operational threshold crossing; summarize full future horizon decisions only, unknown first critical transition never invented",
        labels="none; no empirical accuracy/TPR/FPR claimed",thresholds="old R4 frozen primary; R9 frozen primary ensemble family_confident",
        code_sha256={name:sha(Path(__file__).parent/name) for name in code_names})
    out.mkdir(parents=True,exist_ok=True);(out/"data").mkdir(exist_ok=True)
    if (out/"protocol.json").exists():
        if json.loads((out/"protocol.json").read_text(encoding="utf-8"))!=protocol:raise ValueError("empirical protocol/code changed")
    else:write_json(out/"protocol.json",protocol)
    if (out/"complete.json").exists():
        done=json.loads((out/"complete.json").read_text(encoding="utf-8"))
        for name,digest in done["artifact_sha256"].items():
            if sha(out/name)!=digest:raise ValueError("saved empirical result changed")
        return pd.read_csv(out/"blocks.csv")
    with np.load(new_model/"data/training_features.npz") as data:
        training=data["matrix"].reshape(-1,128)
    lower,upper=np.quantile(training,[.005,.995],axis=0)
    write_json(out/"training_feature_reference.json",dict(quantiles=[.005,.995],lower=lower.tolist(),upper=upper.tolist(),
        meaning="descriptive central99% training feature range, not a calibrated OOD test",source_sha256=sha(new_model/"data/training_features.npz")))
    blocks,rows,skips,hashes=[],[],[],{}
    for manifest in manifests:
        source=manifest["source"]
        stem=("code_"+source.split(":",1)[1].lower().replace(" ","_")) if source.startswith("CODE") else "portal_"+source.split(":",1)[1]
        prepared=panel/(stem+".csv.npz")
        if sha(prepared)!=manifest["events_npz_sha256"]:raise ValueError(f"prepared source changed: {prepared}")
        hashes[str(prepared)]=sha(prepared)
        if not manifest["eligible_for_full_history"]:
            skips.append(dict(source=source,reason="missing full calendar months",months=manifest["missing_calendar_months"]));continue
        with np.load(prepared) as data:event_ns=data["event_ns"]
        for tag,start,end in block_intervals(manifest):
            key=stem+"_"+tag;cache=out/"data"/(key+".npz")
            events,counts=prepare_block(event_ns,start,end)
            if cache.exists():
                with np.load(cache) as saved:matrix,fits=saved["matrix"],saved["fits"]
            else:
                matrix,fits=build_selection_features(counts,events,[90,180,360],730,90,3,2)
                np.savez_compressed(cache,matrix=matrix,fits=fits,counts=counts,start=str(start),end=str(end))
            batch=SimpleNamespace(matrix=matrix[None],fits=fits[None])
            predicted=extended_scores(batch,[90,180,360],new_model,new_protocol["model_seeds"])
            new_x=fused_features(batch,[90,180,360])[0]
            outside=((new_x<lower)|(new_x>upper)).mean(axis=-1)
            meta=describe_block(manifest,events,counts,start,end,tag)
            meta["block_id"]=key;meta["mean_fraction_features_outside_training_central99pct"]=float(outside.mean())
            old_scores={}
            for original in (entry for entry in old["streams"] if entry["primary"]):
                entry=dict(original);entry["window_index"]=[90,180,360].index(entry["window_days"])
                score,valid=score_selected_stream(batch,entry,old_model)
                old_scores[entry["task"]]=(score[0],valid[0],entry["threshold"])
            for index in range(10):
                day=730+(index+3)*90
                item=dict(block_id=key,source=source,period=tag,decision_date=str((start+pd.Timedelta(days=day)).date()),
                    future180days_observed=day+180<=len(counts),observed_rate_last90=float(counts[day-90:day].mean()),
                    fraction_features_outside_training_central99pct=float(outside[index]),
                    calendar_n90=float(fits[0,index,1,0]),calendar_n90_valid=bool(fits[0,index,1,4]),
                    calendar_n360=float(fits[2,index,1,0]),calendar_n360_valid=bool(fits[2,index,1,4]),
                    auxiliary_n=float(predicted["aux_n_state__ensemble"][0,index]),true_n=None,true_transition=None)
                for task in ("state","warning"):
                    value=float(predicted[task+"__ensemble"][0,index]);threshold=new["thresholds"][task+"__ensemble__family_confident"]
                    item["new_"+task+"_score"]=value;item["new_"+task+"_alarm"]=value>=threshold
                    old_score,valid,old_threshold=old_scores[task]
                    item["old_"+task+"_score"]=float(old_score[index]);item["old_"+task+"_alarm"]=bool(valid[index] and old_score[index]>=old_threshold)
                rows.append(item)
            block_rows=pd.DataFrame(rows[-10:])
            for model in ("old","new"):
                meta[model+"_state_threshold_crossings"]=int(block_rows[model+"_state_alarm"].sum())
                meta[model+"_warning_threshold_crossings_full_horizon"]=int(block_rows.loc[block_rows.future180days_observed,model+"_warning_alarm"].sum())
            meta["valid_fraction_n90"]=float(block_rows.calendar_n90_valid.mean())
            blocks.append(meta)
            print(f"Empirical application {source}/{tag}: rate={meta['mean_daily_rate']:.1f}, new state alarms={meta['new_state_threshold_crossings']}, warning crossings={meta['new_warning_threshold_crossings_full_horizon']}",flush=True)
    frame=pd.DataFrame(blocks);decisions=pd.DataFrame(rows)
    frame.to_csv(out/"blocks.csv",index=False);decisions.to_csv(out/"decisions.csv",index=False)
    write_json(out/"skipped.json",skips)
    write_transfer_report(out,frame,decisions,skips)
    write_json(out/"complete.json",dict(prepared_source_sha256=hashes,
        model_sha256={str(p):sha(p) for root in (old_model,new_model) for p in (root/"models").glob("*.pt")},
        artifact_sha256={str(p.relative_to(out)).replace("\\","/"):sha(p) for p in out.rglob("*") if p.is_file() and p.name!="complete.json"}))
    return frame


def write_transfer_report(out,blocks,decisions,skips):
    lines=["# R7. Применение замороженных детекторов к CODE и свежим городским данным","",
        "CODE дополнен ежегодными core за 2018, 2019 и 2022; подготовлен период 2018–2022. "
        "Четыре городских портала выгружены за 2019–2025 и январь–сентябрь 2026. Это разные источники с разными единицами регистрации. "
        "2026 год неполный; его числа не сравниваются с полным годом без учёта экспозиции.","",
        "В каждом блоке 730 суток отдельной предыстории и 1080 суток мониторинга: 10 решений состояния и 8 решений с полностью наблюдённым будущим горизонтом 180 суток. "
        "Периоды назначены механически до просмотра scores. Ранний и поздний городские периоды перекрываются и не являются независимыми повторениями.","",
        "Для CODE анализируется первая полная последовательность 1810 суток с 2018 года, для порталов — первый и последний доступные блоки такой длины. "
        "Модель фоновой формы получает только префикс; будущие счётчики не участвуют в её обучении. "
        "Сохраняется минутная точность, равные метки не раздвигаются случайным шумом.","",
        "| Источник / период | Событий/сутки | Доля признаков вне центральных 99% обучения | Срабатывания состояния R9 из 10 | Срабатывания предупреждения R9 из 8 |",
        "|---|---:|---:|---:|---:|"]
    for r in blocks.itertuples():
        lines.append(f"| {r.source} / {r.period} | {r.mean_daily_rate:.1f} | {r.mean_fraction_features_outside_training_central99pct:.1%} | {r.new_state_threshold_crossings} | {r.new_warning_threshold_crossings_full_horizon} |")
    lines += ["","Срабатывания — превышения замороженного порога, а не подтверждённые критические события. "
        "Истинный n и время первого перехода неизвестны: accuracy, TPR, вероятность ложной тревоги и качество раннего предупреждения на этих данных не вычисляются. "
        "Отсутствие тревог также не доказывает отсутствие критичности. Scores сети не откалиброваны как вероятности.","",
        "Симуляционный диапазон 0.2–50 событий/сутки существенно меньше многих городских интенсивностей. "
        "Доля признаков вне центрального обучающего диапазона — описательный показатель сдвига распределения, не формальный тест применимости. "
        "Популяционная калибровочная гарантия симуляционного эксперимента не переносится автоматически на города.","",
        "У CODE единица — запись правонарушения с годовым uid; у порталов — дело/отчёт после дедупликации. "
        "Сравнение абсолютных частот между источниками не является сравнением риска преступности. "
        "Календарная дневная база может оставлять часовую сезонность и задержки регистрации; чувствительность к часу суток показана отдельно в E1.","",
        f"Проанализировано {len(blocks)} блоков; пропущено источников с недостаточным календарным покрытием: {len(skips)}. "
        "Причины перечислены в `skipped.json`, происхождение и годовые объёмы — в манифестах подготовленной панели. "
        "Неизвестные метки не заменялись оконными MLE для дообучения.","",
        "![Применение к свежим данным](fresh_city_scores.png)","",
        "Воспроизведение: `python -m critical_processes.empirical_transfer`; проверка: `python scripts/verify_empirical_transfer.py`. "
        "Повторный запуск проверяет хеши и возвращает сохранённую таблицу.",""]
    (out/"analysis.md").write_text("\n".join(lines),encoding="utf-8")
    latest=blocks[(blocks.period=="latest")&blocks.source.str.startswith("portal:")]
    fig,axes=plt.subplots(len(latest),2,figsize=(12,2.8*max(1,len(latest))),squeeze=False)
    for row,block in enumerate(latest.itertuples()):
        d=decisions[decisions.block_id==block.block_id]
        dates=pd.to_datetime(d.decision_date)
        axes[row,0].plot(dates,d.calendar_n90,label="MLE 90",marker=".")
        axes[row,0].plot(dates,d.calendar_n360,label="MLE 360",marker=".")
        axes[row,0].plot(dates,d.auxiliary_n,label="NN auxiliary n",marker=".")
        axes[row,0].axhline(.9,ls=":",color="gray")
        axes[row,0].set_title(block.source+": оценки, не истинный n")
        axes[row,1].plot(dates,d.new_state_score,label="Состояние",marker=".")
        axes[row,1].plot(dates,d.new_warning_score,label="Предупреждение",marker=".")
        axes[row,1].set_title("Неоткалиброванные scores R9")
        for ax in axes[row]: ax.set_ylim(-.02,1.02);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/"fresh_city_scores.png",dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",default="results/empirical_transfer")
    args=parser.parse_args();run_transfer(**vars(args))


if __name__=="__main__":main()
