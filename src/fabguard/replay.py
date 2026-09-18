"""Replay one opaque UORED recording through the frozen local detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from joblib import load

from fabguard.config import RuntimeSettings
from fabguard.data import CHANNEL_MAPPING, read_numeric_table
from fabguard.signals import assess_quality, extract_recording_features
from fabguard.storage import Database, file_sha256
from fabguard.train import anomaly_scores


def _manifest_entry(manifest_path: str | Path, recording_id: str) -> dict[str, Any]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = payload.get("entries", payload)
    matches = [entry for entry in entries if entry.get("recording_id") == recording_id]
    if len(matches) != 1:
        raise ValueError("recording_id was not found exactly once in the manifest")
    return matches[0]


def _artifact_id(recording_id: str, model_version: str, source_hash: str) -> str:
    digest = hashlib.sha256(f"{recording_id}:{model_version}:{source_hash}".encode()).hexdigest()
    return f"art_{digest[:32]}"


def replay_recording(
    *,
    manifest_path: str | Path,
    model_path: str | Path,
    recording_id: str,
    output_directory: str | Path = "runs/replays",
) -> dict[str, Any]:
    """Score all complete windows without exposing evaluator labels or source names."""

    entry = _manifest_entry(manifest_path, recording_id)
    if not entry.get("technically_usable", False):
        raise ValueError("recording is not technically usable under the frozen audit")
    model_file = Path(model_path)
    digest_file = model_file.with_suffix(model_file.suffix + ".sha256")
    if not digest_file.is_file():
        raise ValueError("model digest sidecar is missing")
    expected_digest = digest_file.read_text(encoding="ascii").strip()
    if file_sha256(model_file) != expected_digest:
        raise ValueError("model artifact failed integrity verification")
    bundle = load(model_file)
    if bundle.get("bundle_version") != 1:
        raise ValueError("unsupported model bundle version")
    matrix = read_numeric_table(entry["canonical_path"])
    vibration = matrix[:, CHANNEL_MAPPING["vibration"]]
    audio = matrix[:, CHANNEL_MAPPING["audio"]]
    sample_rate = float(bundle["sample_rate_hz"])
    vibration_quality = assess_quality(vibration, sample_rate_hz=sample_rate)
    audio_quality = assess_quality(audio, sample_rate_hz=sample_rate)
    policy = str(bundle["feature_policy"])
    if not vibration_quality.is_usable:
        raise ValueError("vibration evidence failed the frozen technical quality gate")
    if policy in {"audio", "fusion"} and not audio_quality.is_usable:
        raise ValueError(f"the deployed {policy} policy requires usable paired audio")

    arguments = {
        "sample_rate_hz": sample_rate,
        "window_seconds": float(bundle["window_seconds"]),
        "hop_seconds": float(bundle["hop_seconds"]),
        "frequency_bands_hz": tuple(tuple(band) for band in bundle["frequency_bands_hz"]),
    }
    vibration_rows = extract_recording_features(vibration, **arguments)
    audio_rows = extract_recording_features(audio, **arguments)
    if len(vibration_rows) != len(audio_rows):
        raise RuntimeError("paired modalities produced different complete-window counts")
    model_rows: list[list[float]] = []
    windows: list[dict[str, Any]] = []
    for vibration_row, audio_row in zip(vibration_rows, audio_rows, strict=True):
        merged = {
            **{f"vibration__{key}": value for key, value in vibration_row.items()},
            **{f"audio__{key}": value for key, value in audio_row.items()},
        }
        model_rows.append([float(merged[column]) for column in bundle["feature_columns"]])
        windows.append(
            {
                "window_index": int(vibration_row["window_index"]),
                "start_sample": int(vibration_row["start_sample"]),
                "end_sample": int(vibration_row["end_sample"]),
                "vibration_features": {
                    key: float(value)
                    for key, value in vibration_row.items()
                    if key not in {"window_index", "start_sample", "end_sample"}
                },
                "audio_features": {
                    key: float(value)
                    for key, value in audio_row.items()
                    if key not in {"window_index", "start_sample", "end_sample"}
                },
            }
        )
    scores = anomaly_scores(
        bundle["estimator"], bundle["model_family"], np.asarray(model_rows, dtype=float)
    )
    threshold = float(bundle["threshold"])
    for window, score in zip(windows, scores, strict=True):
        window["anomaly_score"] = float(score)
        window["threshold"] = threshold
        window["abnormal"] = bool(score >= threshold)
    selected_index = int(np.argmax(scores))
    selected = windows[selected_index]
    artifact_id = _artifact_id(recording_id, bundle["model_version"], entry["sha256"])
    visible = min(len(vibration), int(sample_rate * 2))
    stride = max(1, visible // 4_000)
    centered = vibration[:visible] - np.mean(vibration[:visible])
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(visible)))
    frequencies = np.fft.rfftfreq(visible, 1 / sample_rate)
    spectrum_stride = max(1, len(frequencies) // 4_000)
    result = {
        "artifact_id": artifact_id,
        "recording_id": recording_id,
        "recorded_data_replay": True,
        "model_version": bundle["model_version"],
        "model_family": bundle["model_family"],
        "feature_policy": policy,
        "preprocessing_version": "fabguard-features-v1",
        "evaluation_scope": "UORED v5 primary 15-bearing LOBO; window-level detector",
        "sample_rate_hz": sample_rate,
        "quality": {
            "vibration": vibration_quality.to_dict(),
            "audio": audio_quality.to_dict(),
        },
        "prediction": {
            "score": float(selected["anomaly_score"]),
            "threshold": threshold,
            "decision": "abnormal" if selected["abnormal"] else "healthy",
            "selected_window_index": selected_index,
        },
        "windows": windows,
        "preview": {
            "time_seconds": (np.arange(0, visible, stride) / sample_rate).tolist(),
            "vibration": vibration[:visible:stride].astype(float).tolist(),
            "frequency_hz": frequencies[::spectrum_stride].astype(float).tolist(),
            "spectrum_magnitude": spectrum[::spectrum_stride].astype(float).tolist(),
        },
    }
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    artifact_path = destination / f"{artifact_id}.json"
    artifact_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["artifact_path"] = str(artifact_path.resolve())
    return result


def register_and_submit(
    result: dict[str, Any],
    *,
    database_url: str,
    api_url: str,
    api_token: str,
) -> dict[str, Any]:
    """Register the opaque on-disk artifact, then send an idempotent API event."""

    path = Path(result["artifact_path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    database = Database(database_url, artifact_root=path.parent)
    registered = database.register_artifact(
        artifact_key=f"replay:{result['artifact_id']}", path=path, sha256=digest
    )
    prediction = result["prediction"]
    response = httpx.post(
        f"{api_url.rstrip('/')}/v1/events",
        headers={
            "Authorization": f"Bearer {api_token}",
            "Idempotency-Key": f"replay:{result['artifact_id']}",
        },
        json={
            "evidence_artifact_id": registered,
            "prediction": {
                "score": prediction["score"],
                "threshold": prediction["threshold"],
                "decision": prediction["decision"],
                "evaluation_scope": result["evaluation_scope"],
            },
            "evidence_version": result["preprocessing_version"],
            "model_version": result["model_version"],
            "graph_version": "fabguard-graph-v1",
            "prompt_version": "fabguard-report-v1",
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_id")
    parser.add_argument("--manifest", default="manifests/uored_vafcls_v5.json")
    parser.add_argument("--model", default="runs/uored-v5-seed17/model.joblib")
    parser.add_argument("--output-dir", default="runs/replays")
    parser.add_argument("--submit", action="store_true", help="submit using environment settings")
    args = parser.parse_args()
    result = replay_recording(
        manifest_path=args.manifest,
        model_path=args.model,
        recording_id=args.recording_id,
        output_directory=args.output_dir,
    )
    if args.submit:
        settings = RuntimeSettings()
        if not settings.producer_token:
            parser.error("FABGUARD_PRODUCER_TOKEN is required with --submit")
        result["submission"] = register_and_submit(
            result,
            database_url=settings.database_url,
            api_url=os.environ.get("FABGUARD_API_URL", "http://127.0.0.1:8000"),
            api_token=settings.producer_token,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
