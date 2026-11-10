from __future__ import annotations

import hashlib
import json
from pathlib import Path


ASSETS = {
    "pilot_indicators.png": "results/simulation_v2/indicators.png",
    "n_accuracy_mae.png": "results/n_estimation_accuracy/mae_map.png",
    "n_accuracy_auc.png": "results/n_estimation_accuracy/auc_map.png",
    "background_comparison.png": "results/background_comparison/n_background_comparison.png",
    "target_timeline.png": "results/state_warning_audit/target_timeline.png",
    "final_detector_comparison.png": "results/selected_detectors/final_comparison.png",
    "threshold_tradeoff.png": "results/sensitivity_diagnostics/threshold_tradeoff.png",
    "n_overlap.png": "results/sensitivity_diagnostics/n_overlap.png",
    "city_background_comparison.png": "results/intraday_comparison/intraday_comparison.png",
    "final_indicators.png": "results/indicators_final/indicators.png",
    "dynamic_estimation.png": "results/dynamic_estimation/estimation_comparison.png",
    "dynamic_examples.png": "results/dynamic_estimation/causal_examples.png",
    "genealogical_clusters.png": "results/genealogical_clusters/genealogy.png",
    "extended_detector_comparison.png": "results/project_completion/extended_detector_comparison.png",
    "fresh_city_scores.png": "results/project_completion/fresh_city_scores.png",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "report" / "assets"
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, relative_source in ASSETS.items():
        source = root / relative_source
        content = source.read_bytes()
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"expected PNG source: {relative_source}")
        destination = target / filename
        destination.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
        records.append({"source": relative_source, "copy": f"report/assets/{filename}", "sha256": digest})
    (target / "manifest.json").write_text(json.dumps({"assets": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(records)} report figures with matching SHA-256 and unchanged source images.")


if __name__ == "__main__":
    main()
