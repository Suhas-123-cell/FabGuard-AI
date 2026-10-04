from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
from uuid import uuid4

import httpx
import numpy as np
import psycopg
import pytest
import uvicorn
from joblib import dump
from psycopg import sql

from fabguard.api import Principal, create_app
from fabguard.config import RuntimeSettings
from fabguard.initialize import initialize_runtime
from fabguard.replay import replay_recording
from fabguard.storage import Database
from fabguard.train import HealthyReferenceDetector


@pytest.fixture
def replay_fixture(tmp_path):
    """A synthetic paired recording; no claim of industrial ground truth."""
    sample_rate = 100
    seconds = np.arange(1000) / sample_rate
    matrix = np.column_stack(
        [
            3 * np.sin(2 * np.pi * 8 * seconds),
            np.sin(2 * np.pi * 6 * seconds),
            np.full(1000, 1750),
            np.full(1000, 400),
            np.full(1000, 25),
        ]
    )
    source = tmp_path / "recording.csv"
    np.savetxt(source, matrix, delimiter=",")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "recording_id": "fixture-recording",
                        "canonical_path": str(source),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "technically_usable": True,
                    }
                ]
            }
        )
    )
    model = tmp_path / "model.joblib"
    dump(
        {
            "bundle_version": 1,
            "model_version": "fixture-model-v1",
            "model_family": "healthy_reference",
            "feature_policy": "fusion",
            "feature_columns": ["vibration__rms"],
            "threshold": 2.0,
            "sample_rate_hz": sample_rate,
            "window_seconds": 1.0,
            "hop_seconds": 0.5,
            "frequency_bands_hz": ((0.0, 50.0),),
            "estimator": HealthyReferenceDetector().fit(np.array([[0.6], [0.7], [0.8]])),
        },
        model,
    )
    model.with_suffix(".joblib.sha256").write_text(hashlib.sha256(model.read_bytes()).hexdigest())
    return replay_recording(
        manifest_path=manifest,
        model_path=model,
        recording_id="fixture-recording",
        output_directory=tmp_path / "artifacts",
    )


@pytest.fixture
def postgres_runtime(tmp_path):
    url = os.getenv("FABGUARD_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("set FABGUARD_TEST_POSTGRES_URL for real PostgreSQL end-to-end coverage")
    schema = f"test_{uuid4().hex}"
    with psycopg.connect(url, autocommit=True) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    separator = "&" if "?" in url else "?"
    scoped_url = f"{url}{separator}options=-csearch_path%3D{schema},public"
    settings = RuntimeSettings(
        _env_file=None,
        database_url=scoped_url.replace("postgresql://", "postgresql+psycopg://"),
        checkpoint_database_url=scoped_url,
        artifact_root=tmp_path / "artifacts",
        llm_provider="offline",
        producer_token="producer-fixture-012345",
        analyst_token="analyst-fixture-012345",
        reviewer_token="reviewer-fixture-012345",
    )
    db = Database(settings.database_url, artifact_root=settings.artifact_root)
    try:
        assert initialize_runtime(settings) == 4
        # Setup is safe to repeat on the same schema.
        assert initialize_runtime(settings) == 4
        yield settings, db
    finally:
        db.engine.dispose()
        with psycopg.connect(url, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def live_api(postgres_runtime):
    settings, db = postgres_runtime
    app = create_app(
        db,
        api_keys={
            settings.producer_token: Principal("producer", frozenset({"producer"})),
            settings.analyst_token: Principal("analyst", frozenset({"analyst"})),
            settings.reviewer_token: Principal("reviewer", frozenset({"reviewer"})),
        },
        graph_version=settings.graph_version,
        prompt_version=settings.prompt_version,
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started
        assert httpx.get(f"{url}/health").status_code == 200
        yield url, settings, db
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
