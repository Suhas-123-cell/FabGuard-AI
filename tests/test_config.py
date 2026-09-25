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


def test_runtime_settings_default_agents_to_groq(monkeypatch):
    monkeypatch.delenv("FABGUARD_LLM_PROVIDER", raising=False)
    settings = RuntimeSettings(_env_file=None)
    assert settings.llm_provider == "groq"
    assert settings.llm_model == "openai/gpt-oss-20b"
    assert settings.llm_base_url == "https://api.groq.com/openai/v1"
