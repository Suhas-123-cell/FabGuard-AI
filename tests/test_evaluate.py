import pandas as pd

from fabguard.evaluate import run_evaluation


def test_offline_ablation_runs_frozen_cases_and_applies_adoption_rule(tmp_path):
    summary = run_evaluation(output_directory=tmp_path, repeats=3)

    assert summary["cases"] == 10
    assert summary["fixed_majority_passes"] == 10
    assert summary["adaptive_majority_passes"] == 10
    assert summary["adaptive_enabled_by_rule"] is False
    assert summary["default_variant"] == "fixed"
    assert (tmp_path / "case_runs.csv").exists()
    assert (tmp_path / "results.json").exists()


def test_offline_run_writes_per_variant_quality_metrics(tmp_path):
    run_evaluation(output_directory=tmp_path, repeats=1)

    metrics = pd.read_csv(tmp_path / "metrics.csv")

    assert list(metrics["variant"]) == ["adaptive", "fixed"]
    assert (metrics["model"] == "scripted-offline-fixture").all()
    assert (metrics["runs"] == 10).all()
    assert (metrics["pass_rate"] == 1.0).all()
    assert (metrics["schema_valid_rate"] == 1.0).all()
    # Four of the ten frozen cases expect abstention.
    assert (metrics["fail_closed_rate"] == 0.4).all()
    assert (metrics["unsafe_runs"] == 0).all()
    assert (metrics["p95_latency_ms"] >= metrics["p50_latency_ms"]).all()
