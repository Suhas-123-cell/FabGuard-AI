from fabguard.config import ExperimentConfig, RuntimeSettings


def test_experiment_config_round_trip(tmp_path):
    config = ExperimentConfig()
    path = tmp_path / "config.json"
    config.write_json(path)
    assert ExperimentConfig.from_json(path) == config


def test_runtime_settings_keep_secrets_out_of_experiment_config(monkeypatch):
    monkeypatch.setenv("FABGUARD_LLM_API_KEY", "secret-value")
    settings = RuntimeSettings(_env_file=None)
    assert settings.llm_api_key == "secret-value"
    assert "secret-value" not in ExperimentConfig().model_dump_json()
