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
