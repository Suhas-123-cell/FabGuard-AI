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


def test_checkpoint_database_defaults_to_runtime_database():
    settings = RuntimeSettings(
        _env_file=None,
        database_url="postgresql+psycopg://user:password@localhost:5432/test",
        checkpoint_database_url=None,
    )
    assert settings.checkpoint_url == "postgresql://user:password@localhost:5432/test"


def test_runtime_api_reads_dotenv(tmp_path, monkeypatch):
    from fabguard.api import create_runtime_app

    monkeypatch.chdir(tmp_path)
    for role in ("PRODUCER", "ANALYST", "REVIEWER"):
        monkeypatch.delenv(f"FABGUARD_{role}_TOKEN", raising=False)
    monkeypatch.delenv("FABGUARD_DATABASE_URL", raising=False)
    (tmp_path / ".env").write_text(
        "FABGUARD_PRODUCER_TOKEN=producer-0123456789\n"
        "FABGUARD_ANALYST_TOKEN=analyst-0123456789\n"
        "FABGUARD_REVIEWER_TOKEN=reviewer-0123456789\n"
        f"FABGUARD_DATABASE_URL=sqlite:///{tmp_path / 'runtime.db'}\n"
        "FABGUARD_GRAPH_VERSION=dotenv-graph-v2\n"
    )
    app = create_runtime_app()
    assert app.state.database.engine.url.database == str(tmp_path / "runtime.db")
