"""Final Monte Carlo figure; no parameter fitting or detector selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from critical_processes.indicators import FEATURE_NAMES, summarize_counts
from critical_processes.selected_training import sha, write_json
from critical_processes.simulation import simulate_staged_hawkes_counts


def generate_counts(n, seed, horizon, rate, decay, burn_in_times):
    if n == 0:
        return np.random.default_rng(seed).poisson(rate, horizon)
    return simulate_staged_hawkes_counts(np.array([rate*(1-n)]), np.array([n]), horizon, decay, seed,
                                        burn_in=burn_in_times/(decay*(1-n)))


def summaries(raw, seed, bootstrap=2000):
    rng, rows = np.random.default_rng(seed), []
    for n, group in raw.groupby("true_n", sort=True):
        row = {"true_n": n, "replicates": len(group)}
        indices = rng.integers(0, len(group), size=(bootstrap, len(group)))
        for name in FEATURE_NAMES:
            values = group[name].to_numpy()
            means = values[indices].mean(axis=1)
            lower, upper = np.quantile(means, [.025, .975])
            row.update({name: float(values.mean()), name+"_ci95_lower": lower, name+"_ci95_upper": upper,
                        name+"_mc_se": float(values.std(ddof=1)/np.sqrt(len(values)))})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/indicators_final"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    out = args.out
    root = Path(__file__).resolve().parents[1]
    if args.verify:
        config = json.loads((out/"protocol.json").read_text(encoding="utf-8"))
        complete = json.loads((out/"complete.json").read_text(encoding="utf-8"))
        for path, digest in complete["sha256"].items():
            assert sha(out/path) == digest
        for path, digest in config["code_sha256"].items():
            assert sha(root/path) == digest
        raw = pd.read_csv(out/"replicates.csv", float_precision="round_trip")
        with np.load(out/"counts.npz") as data:
            counts, seeds = data["counts"], data["seeds"]
            assert len(seeds) == len(set(seeds.tolist()))
            for index, row in enumerate(raw.itertuples()):
                expected = generate_counts(row.true_n, int(seeds[index]), config["horizon"], config["rate"], config["decay"], config["burn_in_times"])
                np.testing.assert_array_equal(expected, counts[index])
                np.testing.assert_allclose(summarize_counts(expected, 1.), [getattr(row, name) for name in FEATURE_NAMES], rtol=1e-12, atol=1e-12)
        pd.testing.assert_frame_equal(summaries(raw, config["bootstrap_seed"], config["bootstrap_samples"]),
            pd.read_csv(out/"summary.csv", float_precision="round_trip"), check_dtype=False, rtol=1e-12, atol=1e-12)
        print(f"Verified all {len(raw)} independent Monte Carlo replays, statistics, bootstrap intervals and artifact hashes.")
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError("use --verify for completed results or choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    config = {"ratios": [0., .2, .4, .6, .75, .85, .92, .97], "replicates": 500, "horizon": 365,
        "rate": 2., "decay": 1., "burn_in_times": 8., "seed": 20261123, "bootstrap_seed": 20261124,
        "bootstrap_samples": 2000, "cluster_proxy": "total events / occupied daily bins; not genealogy",
        "code_sha256": {name: sha(root/name) for name in ("scripts/finalize_indicators.py", "src/critical_processes/indicators.py", "src/critical_processes/simulation.py")}}
    write_json(out/"protocol.json", config)
    reserved = set()
    for source in ("selected_detectors", "sensitivity_diagnostics", "dynamic_estimation"):
        for path in (root/"results"/source/"data").glob("*.npz"):
            with np.load(path) as data:
                for key in ("parameter_seeds", "simulator_seeds"):
                    if key in data:
                        reserved.update(data[key].tolist())
    rng, rows, counts, seeds = np.random.default_rng(config["seed"]), [], [], []
    for n in config["ratios"]:
        for replicate in range(config["replicates"]):
            seed = int(rng.integers(0, 2**32, dtype=np.uint64))
            while seed in reserved:
                seed = int(rng.integers(0, 2**32, dtype=np.uint64))
            reserved.add(seed)
            values = generate_counts(n, seed, config["horizon"], config["rate"], config["decay"], config["burn_in_times"])
            rows.append({"true_n": n, "replicate": replicate, "seed": seed,
                **dict(zip(FEATURE_NAMES, summarize_counts(values, 1.)))})
            counts.append(values)
            seeds.append(seed)
        print(f"Final indicators: n={n}, {config['replicates']} replicates", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out/"replicates.csv", index=False)
    np.savez_compressed(out/"counts.npz", counts=np.stack(counts), seeds=np.asarray(seeds, dtype=np.uint32))
    summary = summaries(raw, config["bootstrap_seed"], config["bootstrap_samples"])
    summary.to_csv(out/"summary.csv", index=False)
    titles = ("Средняя частота, событий/сутки", "Выборочная дисперсия суточных количеств", "Автокорреляция первого лага", "Событий на занятые сутки (прокси)")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, name, title in zip(axes.ravel(), FEATURE_NAMES, titles):
        values = summary[name].to_numpy()
        ax.errorbar(summary.true_n, values, yerr=np.array([values-summary[name+"_ci95_lower"], summary[name+"_ci95_upper"]-values]),
                    marker="o", capsize=3, label="500 потоков на уровень n")
        ax.axhline(summary.loc[summary.true_n == 0, name].iloc[0], color="gray", ls="--", label="Пуассон, n=0")
        ax.set(xlabel="Коэффициент ветвления n", ylabel=title)
        ax.grid(alpha=.2)
    axes[0, 0].axhline(config["rate"], color="black", ls=":", label="Ожидаемая частота 2/сутки")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"indicators.png", dpi=160)
    plt.close(fig)
    lines = ["# Итоговый Монте-Карло эксперимент показателей", "",
        "500 независимых повторов на уровень n; 8 уровней с однородным Пуассоном n=0, всего 4000 потоков. "
        "Горизонт 365 суток, β=1/сутки, μ=2(1−n), ожидаемая стационарная частота 2 события/сутки. Прогрев Хоукса — 8/[β(1−n)].", "",
        "Полосы — процентильные 95% интервалы для средних показателей по 500 потокам, бутстреп 2000 раз целых потоков. "
        "Они характеризуют ошибку Монте-Карло и не являются доверительным интервалом n отдельного потока. "
        "При n близком к 1 конечный год наблюдений не гарантирует устойчивую оценку теоретических моментов каждого потока.", "",
        "| n | Частота | Дисперсия | ACF(1) | Событий на занятые сутки |", "|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.true_n:.2f} | {row.mean_rate:.4f} | {row.variance:.4f} | {row.lag1_autocorrelation:.4f} | {row.mean_cluster_size:.4f} |")
    lines += ["", "![Показатели](indicators.png)", "",
        "Кластерная характеристика прежней реализации при суточных бинах и cluster_gap=0.5 равна числу событий, делённому на число непустых суток. "
        "Она показывает концентрацию событий во времени и не оценивает размер генеалогического каскада. "
        "Оценка MLE в этом опыте не повторяется: её точность исследована отдельно в R1.", "",
        "Пилот `simulation_v2` с 30 повторами сохранён. Новый график предназначен для основного текста отчёта.", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe scripts/finalize_indicators.py --verify", "```", ""]
    (out/"analysis.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(out/"complete.json", {"sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "complete.json"}})


if __name__ == "__main__":
    main()
