from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
from critical_processes.documentation_archive import read_artifact,verify_archive


def digest(path):
    return hashlib.sha256(read_artifact(path)).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    report = root / "report"
    test_result = json.loads((root / "reproducibility/test_result.json").read_text(encoding="utf-8"))
    assert test_result["successful"] and test_result["tests_run"] == 110
    assert test_result["failures"] == test_result["errors"] == 0
    documents = [root / "README.md"]
    assert verify_archive(root) == 37
    links = 0
    for path in documents:
        text = path.read_text(encoding="utf-8")
        assert text.count("```") % 2 == 0, f"unclosed code fence in {path.name}"
        if path.name == "README.md":
            assert "$" not in text and "```math" not in text, "README formulas must not depend on a browser TeX renderer"
            formulas = re.findall(r'<img src="(report/assets/equation_[^"]+\.png)" alt="[^"]+" width="(\d+)">', text)
            saved = json.loads((report / "assets/equations_manifest.json").read_text(encoding="utf-8"))
            assert digest(root / "scripts/render_readme_equations.py") == saved["renderer_sha256"]
            assert digest(report / "equations.json") == saved["source_sha256"]
            assert len(formulas) == len(saved["equations"]) == 7
            assert len(re.findall(r'(?ms)^<p align="center">\n\s*<img [^\n]+>\n</p>\n\n', text)) == 7, "formula HTML must be separated from Markdown"
            for (name, width), record in zip(formulas, saved["equations"]):
                assert name == record["file"] and int(width) == record["display_width"]
                content = (root / name).read_bytes()
                assert content.startswith(b"\x89PNG\r\n\x1a\n") and digest(root / name) == record["sha256"]
                assert [int.from_bytes(content[16:20], "big"), int.from_bytes(content[20:24], "big")] == record["pixel_size"]
            assert text.count("n<sup>*</sup>") == 2
            assert "итоговая редакция" in text.lower()
            figures = [int(v) for v in re.findall(r"(?m)^\*Рисунок (\d+)\.", text)]
            assert figures == list(range(1, 23)), "figure captions must be sequential"
        table_width = None
        in_code = False
        for number, line in enumerate(text.splitlines(), 1):
            if line.startswith("```"):
                in_code = not in_code
            if in_code:
                continue
            if line.startswith("|"):
                width = len(re.split(r"(?<!\\)\|", line))
                if table_width is not None:
                    assert width == table_width, f"table column mismatch: {path.name}:{number}"
                table_width = width
            else:
                table_width = None
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            link = link.strip("<>")
            if re.match(r"[a-z]+://", link) or link.startswith("#"):
                continue
            relative = unquote(link.split("#", 1)[0])
            target = (path.parent / relative).resolve()
            assert target.is_relative_to(root), f"link outside project: {link}"
            assert target.exists(), f"broken local link in {path.name}: {link}"
            links += 1
    manifest = json.loads((report / "assets/manifest.json").read_text(encoding="utf-8"))
    for item in manifest["assets"]:
        assert digest(root / item["source"]) == digest(root / item["copy"]) == item["sha256"]
    accuracy = pd.read_csv(root / "results/n_estimation_accuracy/summary.csv")
    for window, expected in ((90, .059143), (360, .017184)):
        cell = accuracy[(accuracy.true_n == .95) & (accuracy.target_rate == 20) & (accuracy.window_days == window)]
        assert len(cell) == 1
        np.testing.assert_allclose(cell.mae_valid.iloc[0], expected, atol=1e-6, rtol=0)
    background = pd.read_csv(root / "results/background_comparison/summary.csv")
    for scenario, method, expected in (("seasonal", "constant_joint", .909405),
        ("seasonal", "past_shape_joint", .023715), ("hawkes_n0p9", "past_shape_joint", .894580),
        ("hawkes_n0p9", "past_fixed", .170038)):
        cell = background[(background.scenario == scenario) & (background.method == method) &
                          (background.target_rate == 20) & (background.window_days == 360)]
        assert len(cell) == 1
        np.testing.assert_allclose(cell.mean_n_hat_valid.iloc[0], expected, atol=1e-6, rtol=0)
    final_directory = root / "results/selected_detectors"
    final = json.loads((final_directory / "metrics.json").read_text(encoding="utf-8"))
    assert final["warning__constant_mle__ensemble"]["detected_trajectories"] == 0
    assert final["warning__constant_mle__ensemble"]["positive_trajectories"] == 600
    np.testing.assert_allclose(final["state__calendar_mle__ensemble"]["window_sensitivity"], .02860530606195346)
    diagnostic = pd.read_csv(root / "results/sensitivity_diagnostics/operating_metrics.csv")
    for method, expected in (("paired_known", 187 / 600), ("paired_calendar", 1 / 600)):
        cell = diagnostic[(diagnostic.task == "warning") & (diagnostic.method == method) &
                          (diagnostic.policy == "family_confident_source_scope")]
        assert len(cell) == 1
        np.testing.assert_allclose(cell.clean_event_detection_probability.iloc[0], expected, atol=1e-12)
    dynamic = root / "results/dynamic_estimation"
    dynamic_complete = json.loads((dynamic / "complete.json").read_text(encoding="utf-8"))
    for name, expected_hash in dynamic_complete["artifact_sha256"].items():
        assert digest(dynamic / name) == expected_hash
    estimation = pd.read_csv(dynamic / "estimation_summary.csv").groupby("method").mae.mean()
    for method, expected in (("calendar90", .087492), ("dynamic_calendar", .081665)):
        np.testing.assert_allclose(estimation[method], expected, atol=1e-6, rtol=0)
    delay = pd.read_csv(dynamic / "abrupt_delay.csv").groupby("method").delay_or_censor_cap_days.mean()
    np.testing.assert_allclose(delay[["calendar90", "dynamic_calendar"]], [213.9, 272.4], atol=1e-10)
    dynamic_metrics = json.loads((dynamic / "metrics.json").read_text(encoding="utf-8"))
    warning = dynamic_metrics["warning__mlp_dynamic_calendar_ensemble__family_confident"]
    assert warning["detected_trajectories"] == 2 and warning["positive_trajectories"] == 600
    np.testing.assert_allclose(warning["clean_event_detection_probability"], 2/600)
    indicators = root / "results/indicators_final"
    indicator_complete = json.loads((indicators / "complete.json").read_text(encoding="utf-8"))
    for name, expected_hash in indicator_complete["sha256"].items():
        assert digest(indicators / name) == expected_hash
    summary = pd.read_csv(indicators / "summary.csv").set_index("true_n")
    assert (summary.replicates == 500).all() and summary.replicates.sum() == 4000
    np.testing.assert_allclose(summary.loc[[0., .97], "variance"], [1.9907, 27.5222], atol=1e-4, rtol=0)
    extension = root / "results/extended_neural"
    extended_complete = json.loads((extension / "complete.json").read_text(encoding="utf-8"))
    for name, expected_hash in extended_complete["artifact_sha256"].items():
        assert digest(extension / name) == expected_hash
    extended = json.loads((extension / "metrics.json").read_text(encoding="utf-8"))
    for policy, expected in (("family_confident", 5/600), ("family_nominal_sensitivity", 27/600)):
        np.testing.assert_allclose(extended["warning__ensemble__"+policy]["clean_event_detection_probability"], expected)
    auxiliary = pd.read_csv(extension / "auxiliary_n_errors.csv")
    np.testing.assert_allclose(auxiliary.auxiliary_n_mae.mean(), .042610474081631364)
    genealogy = root / "results/genealogical_clusters"
    gene_complete = json.loads((genealogy / "complete.json").read_text(encoding="utf-8"))
    for name, expected_hash in gene_complete["artifact_sha256"].items():
        assert digest(genealogy / name) == expected_hash
    gene = pd.read_csv(genealogy / "summary.csv").set_index("n")
    assert gene.clusters.sum() == 800000
    np.testing.assert_allclose(gene.loc[[.75,.97],"mean_size"],[3.97421,32.63428])
    np.testing.assert_allclose(gene.loc[[.75,.97],"probability_size_ge_100"],[.00021,.05566])
    empirical = root / "results/empirical_transfer"
    real_complete = json.loads((empirical / "complete.json").read_text(encoding="utf-8"))
    for name, expected_hash in real_complete["artifact_sha256"].items():
        assert digest(empirical / name) == expected_hash
    real = pd.read_csv(empirical / "blocks.csv")
    decisions = pd.read_csv(empirical / "decisions.csv")
    assert len(real) == 22 and len(decisions) == 220 and decisions.true_n.isna().all()
    assert real.filter(like="threshold_crossings").sum().sum() == 0
    completion = root / "results/project_completion"
    provenance = json.loads((completion / "provenance.json").read_text(encoding="utf-8"))
    for name, expected_hash in provenance["source_sha256"].items():
        assert digest(root / name) == expected_hash
    for name, expected_hash in provenance["artifact_sha256"].items():
        assert digest(completion / name) == expected_hash
    totals = json.loads((completion / "data_totals.json").read_text(encoding="utf-8"))
    assert totals["portal_raw_rows"] == 7395987 and totals["portal_prepared_cases"] == 7146403
    assert totals["code_prepared_offense_records"] == 9543104 and totals["code_eligible_cities"] == 14
    effects = pd.read_csv(completion / "paired_detector_differences.csv")
    assert effects.filter(like="bootstrap").notna().all().all()
    complete = json.loads((final_directory / "final_test_complete.json").read_text(encoding="utf-8"))
    assert digest(final_directory / "frozen_plan.json") == complete["frozen_plan_sha256"]
    assert digest(final_directory / "metrics.json") == complete["metrics_sha256"]
    spatial = root / "results/spatiotemporal_forecast"
    spatial_protocol = json.loads((spatial / "protocol.json").read_text(encoding="utf-8"))
    assert spatial_protocol["bin_hours"] == 6 and spatial_protocol["primary_information_lag_days"] == 7
    for name, value in spatial_protocol["code_sha256"].items():
        assert digest(root / "src/critical_processes" / name) == value
    spatial_complete = json.loads((spatial / "complete.json").read_text(encoding="utf-8"))
    for name, value in spatial_complete["artifact_sha256"].items():
        assert digest(spatial / name) == value
    spatial_metrics = pd.read_csv(spatial / "test_metrics.csv")
    assert len(spatial_metrics) == 40
    for city, expected in (("chicago", 1.8578675762290284), ("san_francisco", 1.4367698107119444)):
        cell = spatial_metrics[(spatial_metrics.city == city) & (spatial_metrics.information_lag_days == 7)
                               & (spatial_metrics.split == "test2026") & (spatial_metrics.method == "neural_ensemble")]
        assert len(cell) == 1 and cell.count_primary.iloc[0]
        np.testing.assert_allclose(cell.mae.iloc[0], expected, rtol=0, atol=1e-12)
    spikes = spatial_metrics[(spatial_metrics.information_lag_days == 7) & (spatial_metrics.split == "test2026")
                             & spatial_metrics.spike_primary].set_index("city")
    assert spikes.loc["chicago", "spike_true_positive"] == 22
    assert spikes.loc["san_francisco", "spike_true_positive"] == 0
    summary_dir = root / "results/spatiotemporal_summary"
    summary_provenance = json.loads((summary_dir / "provenance.json").read_text(encoding="utf-8"))
    for name, value in summary_provenance["source_sha256"].items():
        assert digest(root / name) == value
    for name, value in summary_provenance["artifact_sha256"].items():
        assert digest(summary_dir / name) == value
    forecasts = pd.read_csv(summary_dir / "next_interval_forecasts.csv")
    assert len(forecasts) == 32 and forecasts.observed_count.isna().all()
    assert (forecasts.forecast_start == "2026-10-01 00:00:00").all()
    offline = root / "results/offline_readiness"
    offline_protocol = json.loads((offline / "protocol.json").read_text(encoding="utf-8"))
    for name, value in offline_protocol["code_sha256"].items():
        assert digest(root / "src/critical_processes" / name) == value
    offline_complete = json.loads((offline / "complete.json").read_text(encoding="utf-8"))
    for name, value in offline_complete["artifact_sha256"].items():
        assert digest(offline / name) == value
    journal = json.loads((offline / "forecast_journal.json").read_text(encoding="utf-8"))
    assert len(journal["entries"]) == 136 and not journal["contains_outcomes"]
    readiness = json.loads((offline / "readiness.json").read_text(encoding="utf-8"))
    assert readiness["status"] == "research_prototype_offline_only" and not readiness["real_pilot_performed"]
    new_summary = root / "results/offline_summary"
    provenance = json.loads((new_summary / "provenance.json").read_text(encoding="utf-8"))
    for name, value in provenance["source_sha256"].items():
        assert digest(root / name) == value
    for name, value in provenance["artifact_sha256"].items():
        assert digest(new_summary / name) == value
    interpreted = pd.read_csv(new_summary / "interpretable_differences.csv")
    assert interpreted.loc[interpreted.blocks < 4, "mae_ci_low"].isna().all()
    expanded = root / 'results/expanded_district_detector'
    expanded_protocol = json.loads((expanded / 'protocol.json').read_text(encoding='utf-8'))
    for name, value in expanded_protocol['code_sha256'].items():
        assert digest(root / 'src/critical_processes' / name) == value
    assert digest(root / 'scripts/prepare_expanded_districts.py') == expanded_protocol['preparation_sha256']
    for name, value in json.loads((expanded / 'complete.json').read_text(encoding='utf-8'))['artifact_sha256'].items():
        assert digest(expanded / name) == value
    verification = json.loads((expanded / 'verification.json').read_text(encoding='utf-8'))
    assert verification['successful'] and verification['frozen_forecast_files'] == 8
    assert sum(verification['districts'].values()) == 114
    assert len(pd.read_csv(expanded / 'metrics.csv')) == 336
    assert len(pd.read_csv(expanded / 'headline.csv')) == 48
    expanded_assets = json.loads((report / 'assets/expanded_manifest.json').read_text(encoding='utf-8'))
    for name, value in expanded_assets['source_sha256'].items():assert digest(root / name) == value
    for item in expanded_assets['assets']:assert digest(root / item['file']) == item['sha256']
    data_accounting = pd.read_csv(expanded / 'data_summary.csv')
    assert data_accounting.common_scope_cases.sum() == 4036805
    assert data_accounting.train_cases.sum() == 2354861
    assert data_accounting.districts.sum() == 114
    pdf_build = json.loads((root / 'output/pdf/r12_build.json').read_text(encoding='utf-8'))
    assert pdf_build['total_pages'] == 53 and pdf_build['base_pages'] == 48
    assert digest(root / 'output/pdf/hawkes_criticality_report.pdf') == pdf_build['output_sha256']
    assert digest(root / 'scripts/append_expanded_pdf.py') == pdf_build['builder_sha256']
    for name, value in pdf_build['source_sha256'].items():assert digest(root / name) == value
    pdf_qa = json.loads((root / 'output/pdf/r12_qa.json').read_text(encoding='utf-8'))
    assert pdf_qa['successful'] and pdf_qa['sha256'] == pdf_build['output_sha256']
    print(f"Verified {len(documents)} Markdown documents, {links} local links, {len(manifest['assets'])+2} figure copies, P1/R1-R12 result cells and unchanged frozen final metrics.")


if __name__ == "__main__":
    main()
