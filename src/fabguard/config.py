"""Runtime and experiment configuration with safe local defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FeatureConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_rate_hz: int = 42_000
    window_seconds: float = 1.0
    hop_seconds: float = 0.5
    band_edges_hz: tuple[tuple[float, float], ...] = (
        (0.0, 500.0),
        (500.0, 2_000.0),
        (2_000.0, 5_000.0),
        (5_000.0, 10_000.0),
        (10_000.0, 21_000.0),
    )


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int = 17
    primary_bearings: tuple[int, ...] = (*range(1, 11), *range(16, 21))
    quarantined_bearings: tuple[int, ...] = tuple(range(11, 16))
    threshold_quantiles: tuple[float, ...] = (0.95, 0.975, 0.99)
    isolation_forest_estimators: tuple[int, ...] = (100, 250)
    isolation_forest_contamination: tuple[float, ...] = (0.01, 0.025, 0.05)
    random_forest_estimators: tuple[int, ...] = (150, 300)
    random_forest_max_depth: tuple[int, ...] = (4, 8)
    inner_folds: int = 4


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_version: str = "UORED-VAFCLS-v5"
    task: Literal["healthy_vs_abnormal"] = "healthy_vs_abnormal"
    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class RuntimeSettings(BaseSettings):
    """Environment-only secrets and service locations.

    Secrets are deliberately excluded from serialized run configuration.
    """

    model_config = SettingsConfigDict(
        env_prefix="FABGUARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://fabguard:fabguard@localhost:5432/fabguard"
    checkpoint_database_url: str = "postgresql://fabguard:fabguard@localhost:5432/fabguard"
    artifact_root: Path = Path("runs/replays")
    llm_provider: Literal["offline", "groq", "openai_compatible"] = "offline"
    llm_model: str = "openai/gpt-oss-20b"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str | None = None
    run_deadline_seconds: int = 120
    llm_timeout_seconds: int = 25
    audio_enabled: bool = True
    graph_version: str = "fabguard-graph-v1"
    prompt_version: str = "fabguard-report-v1"
    producer_token: str | None = None
    analyst_token: str | None = None
    reviewer_token: str | None = None


DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()
