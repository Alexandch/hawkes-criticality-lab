from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaln

from critical_processes.genealogy import sample_clusters
from critical_processes.negative_controls import wilson_interval
from critical_processes.selected_training import sha, write_json


def summarize(n, sizes, durations, depth):
    mean, theory = sizes.mean(), 1/(1-n)
    se = sizes.std(ddof=1)/np.sqrt(len(sizes))
    rows = dict(n=n, clusters=len(sizes), total_events=int(sizes.sum()), mean_size=float(mean),
        size_variance=float(sizes.var(ddof=1)), theoretical_mean_size=theory, theoretical_size_variance=n/(1-n)**3,
        mean_size_mc_se=float(se), mean_size_mc_ci95_lower=float(mean-1.96*se), mean_size_mc_ci95_upper=float(mean+1.96*se),
        median_size=float(np.median(sizes)), size_quantile95=float(np.quantile(sizes,.95)), largest_cluster=int(sizes.max()),
        mean_duration_days=float(durations.mean()), duration_quantile95_days=float(np.quantile(durations,.95)),
        mean_max_generation=float(depth.mean()))
    for cutoff in (10,100,1000):
        hits = int((sizes>=cutoff).sum())
        lo, hi = wilson_interval(hits, len(sizes))
        rows.update({f"probability_size_ge_{cutoff}": hits/len(sizes), f"size_ge_{cutoff}_ci95_lower": lo, f"size_ge_{cutoff}_ci95_upper": hi})
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path("results/genealogical_clusters"))
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    out=args.out
    if args.verify:
        protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
        complete=json.loads((out/"complete.json").read_text(encoding="utf-8"))
        for name,digest in complete["artifact_sha256"].items():
            assert sha(out/name)==digest
        for name,digest in protocol["code_sha256"].items():
            assert sha(name)==digest
        rows=[]
        with np.load(out/"clusters.npz") as saved:
            for index,n in enumerate(protocol["ratios"]):
                arrays=sample_clusters(n,protocol["decay"],protocol["clusters_per_level"],protocol["seeds"][index])
                for key,value in zip(("sizes","durations","generations"),arrays):
                    np.testing.assert_array_equal(value,saved[f"{index}_{key}"])
                rows.append(summarize(n,*arrays))
        pd.testing.assert_frame_equal(pd.DataFrame(rows),pd.read_csv(out/"summary.csv",float_precision="round_trip"),check_dtype=False,rtol=1e-12,atol=1e-12)
        print(f"Verified complete genealogies for {sum(r['clusters'] for r in rows):,} clusters, theoretical moment comparisons and all hashes.")
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError("choose a new directory or use --verify")
    out.mkdir(parents=True,exist_ok=True)
    ratios=[0.,.2,.4,.6,.75,.85,.92,.97]
    seeds=np.random.SeedSequence(20261125).generate_state(len(ratios)).astype(np.uint32).tolist()
    protocol=dict(ratios=ratios,seeds=seeds,clusters_per_level=100000,decay=1.,max_events=1000000,
        method="one immigrant at t=0; each node has Poisson(n) children with Exp(beta) independent delays; complete progeny, no horizon censoring",
        distinction="cluster-weighted complete trees, not event-weighted samples or observed daily concentration",
        code_sha256={name:sha(name) for name in ("scripts/complete_genealogy.py","src/critical_processes/genealogy.py")})
    write_json(out/"protocol.json",protocol)
    records,arrays=[],{}
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    for index,n in enumerate(ratios):
        sizes,durations,depth=sample_clusters(n,replicates=protocol["clusters_per_level"],seed=seeds[index])
        records.append(summarize(n,sizes,durations,depth))
        for key,value in zip(("sizes","durations","generations"),(sizes,durations,depth)):
            arrays[f"{index}_{key}"]=value
        if n in (.4,.75,.92,.97):
            grid=np.unique(np.geomspace(1,max(2,sizes.max()),150).astype(int))
            empirical=np.array([(sizes>=k).mean() for k in grid])
            axes[1].loglog(grid,empirical,label=f"n={n}")
        print(f"Genealogy n={n}: mean size={sizes.mean():.4f}, theory={1/(1-n):.4f}, max={sizes.max()}",flush=True)
    summary=pd.DataFrame(records)
    summary.to_csv(out/"summary.csv",index=False)
    np.savez_compressed(out/"clusters.npz",**arrays)
    axes[0].errorbar(summary.n,summary.mean_size,yerr=1.96*summary.mean_size_mc_se,fmt="o",capsize=3,label="100 000 полных кластеров")
    x=np.linspace(0,.97,200)
    axes[0].plot(x,1/(1-x),label="Теория 1/(1−n)")
    axes[0].set(xlabel="n",ylabel="Средний размер полного кластера")
    axes[1].set(xlabel="Размер s",ylabel="P(S ≥ s)")
    axes[2].plot(summary.n,summary.duration_quantile95_days,marker="o",label="95% квантиль длительности")
    axes[2].set(xlabel="n",ylabel="Длительность полного кластера, суток")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/"genealogy.png",dpi=160); plt.close(fig)
    lines=["# Полные генеалогические кластеры Хоукса","",
        "100 000 независимых полных деревьев на каждый из 8 уровней n, всего 800 000 кластеров. "
        "У каждого события число непосредственных потомков имеет распределение Poisson(n), задержки потомков — Exp(β), β=1/сутки. "
        "Родитель каждого события сохраняется в имитаторе; размер включает корневое событие и всех потомков. "
        "Дерево не обрезается годовым окном. При превышении лимита имитация останавливается с ошибкой, вместо молчаливого усечения.","",
        "Это отдельное исследование полного семейства одного иммигранта. Оно дополняет суточный прокси P1 и не означает восстановления генеалогии зарегистрированных преступлений. "
        "Частота иммигрантов μ в потоке могла бы задаваться как 2(1−n), сохраняя среднюю общую интенсивность; размер одного дерева от μ не зависит.","",
        "При Poisson(n) потомстве E[S]=1/(1−n), Var(S)=n/(1−n)^3. Эти формулы следуют из равенства S=1+Σ S_j: "
        "m=1+nm, v=nv+nm². Сравниваются полные размеры, а не число потомков только первого поколения.","",
        "| n | Средний размер | Теория | 95% квантиль размера | P(S≥100) | 95% квантиль длительности, сутки |",
        "|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.n:.2f} | {row.mean_size:.3f} | {row.theoretical_mean_size:.3f} | {row.size_quantile95:.0f} | {row.probability_size_ge_100:.3%} | {row.duration_quantile95_days:.3f} |")
    lines += ["","![Генеалогические кластеры](genealogy.png)","",
        "Полосы среднего — приближённые 1.96 Monte Carlo SE по независимым деревьям; при тяжёлом хвосте их точность ограничена. "
        "Для хвостовых долей сохранены Wilson-интервалы. Полная генеалогия известна только в имитации, поэтому её показатели не подаются детектору реальных событий.","",
        "Кластерное представление: [Hawkes и Oakes (1974)](https://doi.org/10.2307/3212693); "
        "[Bordenave и Torrisi (2007)](https://doi.org/10.1080/15326340701645959).","",
        "```powershell",".\\.venv\\Scripts\\python.exe scripts/complete_genealogy.py --verify","```",""]
    (out/"analysis.md").write_text("\n".join(lines),encoding="utf-8")
    write_json(out/"complete.json",{"artifact_sha256":{p.name:sha(p) for p in out.iterdir() if p.is_file()}})


if __name__=="__main__": main()
