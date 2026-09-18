"""Bearing-separated feature extraction, nested LOBO evaluation, and model export."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from joblib import dump
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from fabguard.config import ExperimentConfig
from fabguard.data import CHANNEL_MAPPING, RecordingManifestEntry, read_numeric_table
from fabguard.signals import FEATURE_COLUMNS, extract_features, extract_recording_features

IDENTITY_COLUMNS = (
    "recording_id",
    "bearing_id",
    "health_state",
    "fault_family",
    "manufacturer",
    "cohort",
    "binary_label",
    "window_index",
    "start_sample",
    "end_sample",
    "load_mean",
    "rpm_mean",
)


@dataclass(frozen=True)
class Candidate:
    family: str
    params: dict[str, Any]
    threshold_quantile: float

    @property
    def candidate_id(self) -> str:
        parts = [self.family, *(f"{key}={self.params[key]}" for key in sorted(self.params))]
        return ":".join(parts) + f":q={self.threshold_quantile:g}"


class HealthyReferenceDetector(BaseEstimator):
    """Robust distance from healthy training examples."""

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> HealthyReferenceDetector:
        values = np.asarray(x, dtype=float)
        if not len(values):
            raise ValueError("healthy reference requires training rows")
        self.center_ = np.median(values, axis=0)
        mad = np.median(np.abs(values - self.center_), axis=0)
        self.scale_ = np.where(mad > 1e-12, 1.4826 * mad, 1.0)
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        checked = np.asarray(x, dtype=float)
        robust_z = np.abs((checked - self.center_) / self.scale_)
        return np.sqrt(np.mean(np.square(robust_z), axis=1))


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("entries", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("manifest must contain an entries list")
    return entries


def _value(entry: RecordingManifestEntry | dict[str, Any], name: str) -> Any:
    value = getattr(entry, name) if not isinstance(entry, dict) else entry[name]
    return getattr(value, "value", value)


def build_feature_table(
    entries: Iterable[RecordingManifestEntry | dict[str, Any]],
    *,
    output_path: str | Path | None = None,
    config: ExperimentConfig | None = None,
) -> pd.DataFrame:
    """Build paired window features while keeping evaluator labels out of runtime evidence."""

    experiment = config or ExperimentConfig()
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if not bool(_value(entry, "technically_usable")):
            continue
        matrix = read_numeric_table(_value(entry, "canonical_path"))
        arguments = {
            "sample_rate_hz": experiment.feature.sample_rate_hz,
            "window_seconds": experiment.feature.window_seconds,
            "hop_seconds": experiment.feature.hop_seconds,
            "frequency_bands_hz": experiment.feature.band_edges_hz,
        }
        vibration = extract_recording_features(matrix[:, CHANNEL_MAPPING["vibration"]], **arguments)
        audio = extract_recording_features(matrix[:, CHANNEL_MAPPING["audio"]], **arguments)
        if len(vibration) != len(audio):
            raise ValueError(f"paired feature count mismatch for {_value(entry, 'recording_id')}")
        load = _value(entry, "load")
        speed = _value(entry, "motor_speed")
        load_mean = load.get("mean") if isinstance(load, dict) else load.mean
        speed_mean = speed.get("mean") if isinstance(speed, dict) else speed.mean
        for vibration_row, audio_row in zip(vibration, audio, strict=True):
            row: dict[str, Any] = {
                "recording_id": _value(entry, "recording_id"),
                "bearing_id": int(_value(entry, "bearing_id")),
                "health_state": str(_value(entry, "health_state")),
                "fault_family": str(_value(entry, "eventual_fault_family")),
                "manufacturer": str(_value(entry, "manufacturer")),
                "cohort": str(_value(entry, "cohort")),
                "binary_label": int(_value(entry, "binary_label")),
                "window_index": int(vibration_row["window_index"]),
                "start_sample": int(vibration_row["start_sample"]),
                "end_sample": int(vibration_row["end_sample"]),
                "load_mean": float(load_mean) if load_mean is not None else np.nan,
                "rpm_mean": float(speed_mean) if speed_mean is not None else np.nan,
            }
            for name in FEATURE_COLUMNS:
                row[f"vibration__{name}"] = float(vibration_row[name])
                row[f"audio__{name}"] = float(audio_row[name])
            rows.append(row)
    table = pd.DataFrame(rows)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(destination, index=False)
    return table


def _feature_columns(policy: str) -> list[str]:
    if policy == "vibration":
        return [f"vibration__{name}" for name in FEATURE_COLUMNS]
    if policy == "audio":
        return [f"audio__{name}" for name in FEATURE_COLUMNS]
    if policy == "fusion":
        return [f"vibration__{name}" for name in FEATURE_COLUMNS] + [
            f"audio__{name}" for name in FEATURE_COLUMNS
        ]
    if policy == "load_only":
        return ["load_mean"]
    raise ValueError(f"unknown feature policy {policy!r}")


def declared_candidates(config: ExperimentConfig | None = None) -> list[Candidate]:
    evaluation = (config or ExperimentConfig()).evaluation
    candidates = [Candidate("healthy_reference", {}, q) for q in evaluation.threshold_quantiles]
    for estimators in evaluation.isolation_forest_estimators:
        for contamination in evaluation.isolation_forest_contamination:
            candidates.append(
                Candidate(
                    "isolation_forest",
                    {"n_estimators": estimators, "contamination": contamination},
                    evaluation.threshold_quantiles[0],
                )
            )
    for estimators in evaluation.random_forest_estimators:
        for depth in evaluation.random_forest_max_depth:
            candidates.append(
                Candidate(
                    "random_forest",
                    {"n_estimators": estimators, "max_depth": depth},
                    evaluation.threshold_quantiles[0],
                )
            )
    return candidates


def _build_estimator(candidate: Candidate, seed: int) -> BaseEstimator:
    if candidate.family == "healthy_reference":
        return HealthyReferenceDetector()
    if candidate.family == "isolation_forest":
        model = IsolationForest(
            n_estimators=int(candidate.params["n_estimators"]),
            contamination=float(candidate.params["contamination"]),
            random_state=seed,
            n_jobs=1,
        )
    elif candidate.family == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(candidate.params["n_estimators"]),
            max_depth=int(candidate.params["max_depth"]),
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    else:
        raise ValueError(candidate.family)
    return Pipeline([("scale", RobustScaler()), ("model", model)])


def _fit(
    estimator: BaseEstimator, candidate: Candidate, x: np.ndarray, y: np.ndarray
) -> BaseEstimator:
    fitted = clone(estimator)
    if candidate.family in {"healthy_reference", "isolation_forest"}:
        healthy = y == 0
        if not np.any(healthy):
            raise ValueError("training fold has no healthy examples")
        fitted.fit(x[healthy])
    else:
        fitted.fit(x, y)
    return fitted


def anomaly_scores(estimator: BaseEstimator, family: str, x: np.ndarray) -> np.ndarray:
    if family == "healthy_reference":
        return np.asarray(estimator.score_samples(x), dtype=float)
    if family == "isolation_forest":
        return -np.asarray(estimator.decision_function(x), dtype=float)
    if family == "random_forest":
        return np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
    raise ValueError(family)


def _metrics(
    y: np.ndarray, scores: np.ndarray, threshold: float, states: Sequence[str]
) -> dict[str, float]:
    decisions = scores >= threshold
    healthy = y == 0
    developing = np.asarray([state == "developing" for state in states])
    faulty = np.asarray([state == "faulty" for state in states])
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y, decisions)),
        "healthy_false_positive_rate": float(np.mean(decisions[healthy]))
        if np.any(healthy)
        else np.nan,
        "developing_recall": float(np.mean(decisions[developing]))
        if np.any(developing)
        else np.nan,
        "faulty_recall": float(np.mean(decisions[faulty])) if np.any(faulty) else np.nan,
    }


def _inner_predictions(
    train: pd.DataFrame,
    feature_columns: list[str],
    candidate: Candidate,
    *,
    seed: int,
    inner_folds: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    groups = train["bearing_id"].to_numpy()
    splitter = GroupKFold(
        n_splits=min(inner_folds, len(np.unique(groups))), shuffle=True, random_state=seed
    )
    scores = np.full(len(train), np.nan)
    x_all = train[feature_columns].to_numpy(dtype=float)
    y_all = train["binary_label"].to_numpy(dtype=int)
    for inner_train, inner_valid in splitter.split(x_all, y_all, groups=groups):
        estimator = _fit(
            _build_estimator(candidate, seed), candidate, x_all[inner_train], y_all[inner_train]
        )
        scores[inner_valid] = anomaly_scores(estimator, candidate.family, x_all[inner_valid])
    if np.any(~np.isfinite(scores)):
        raise RuntimeError("inner grouped predictions are incomplete")
    return scores, y_all, train["health_state"].astype(str).tolist()


def choose_candidate(
    train: pd.DataFrame,
    feature_columns: list[str],
    candidates: Sequence[Candidate],
    *,
    seed: int,
    inner_folds: int,
) -> tuple[Candidate, float, list[dict[str, Any]]]:
    outcomes = []
    for candidate in candidates:
        scores, y, states = _inner_predictions(
            train, feature_columns, candidate, seed=seed, inner_folds=inner_folds
        )
        threshold = float(np.quantile(scores[y == 0], candidate.threshold_quantile))
        outcomes.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate": asdict(candidate),
                "threshold": threshold,
                "metrics": _metrics(y, scores, threshold, states),
            }
        )
    best = max(
        outcomes,
        key=lambda item: (
            item["metrics"]["balanced_accuracy"],
            item["metrics"]["average_precision"],
            -item["metrics"]["healthy_false_positive_rate"],
            item["candidate_id"],
        ),
    )
    return Candidate(**best["candidate"]), float(best["threshold"]), outcomes


def evaluate_lobo(
    table: pd.DataFrame,
    *,
    policies: Sequence[str] = ("vibration", "audio", "fusion"),
    config: ExperimentConfig | None = None,
    cohort: str = "primary",
    candidates: Sequence[Candidate] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    experiment = config or ExperimentConfig()
    eligible = table.copy() if cohort == "all" else table.loc[table["cohort"] == cohort].copy()
    if eligible.empty:
        raise ValueError(f"no rows in cohort {cohort!r}")
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    grid = list(candidates or declared_candidates(experiment))
    families = sorted({candidate.family for candidate in grid})
    for policy in policies:
        columns = _feature_columns(policy)
        for family in families:
            family_grid = [candidate for candidate in grid if candidate.family == family]
            for bearing in sorted(eligible["bearing_id"].unique()):
                outer_train = eligible.loc[eligible["bearing_id"] != bearing].reset_index(drop=True)
                outer_test = eligible.loc[eligible["bearing_id"] == bearing].reset_index(drop=True)
                if set(outer_test["binary_label"].unique()) != {0, 1}:
                    raise ValueError(f"outer bearing {bearing} lacks both health labels")
                selected, threshold, inner = choose_candidate(
                    outer_train,
                    columns,
                    family_grid,
                    seed=experiment.evaluation.seed + int(bearing),
                    inner_folds=experiment.evaluation.inner_folds,
                )
                x_train = outer_train[columns].to_numpy(dtype=float)
                y_train = outer_train["binary_label"].to_numpy(dtype=int)
                estimator = _fit(
                    _build_estimator(selected, experiment.evaluation.seed),
                    selected,
                    x_train,
                    y_train,
                )
                scores = anomaly_scores(
                    estimator, selected.family, outer_test[columns].to_numpy(dtype=float)
                )
                metrics = _metrics(
                    outer_test["binary_label"].to_numpy(dtype=int),
                    scores,
                    threshold,
                    outer_test["health_state"].astype(str).tolist(),
                )
                output = outer_test[list(IDENTITY_COLUMNS)].copy()
                output["feature_policy"] = policy
                output["model_family"] = selected.family
                output["candidate_id"] = selected.candidate_id
                output["anomaly_score"] = scores
                output["threshold"] = threshold
                output["decision"] = scores >= threshold
                predictions.append(output)
                folds.append(
                    {
                        "fold": f"bearing-{int(bearing):02d}",
                        "test_bearing_id": int(bearing),
                        "feature_policy": policy,
                        "selected_candidate": asdict(selected),
                        "threshold": threshold,
                        "metrics": metrics,
                        "inner_selection": inner,
                    }
                )
    return pd.concat(predictions, ignore_index=True), folds


def summarize_folds(folds: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "feature_policy": fold["feature_policy"],
            "model_family": fold["selected_candidate"]["family"],
            **fold["metrics"],
        }
        for fold in folds
    ]
    frame = pd.DataFrame(rows)
    summaries = []
    metrics = (
        "auroc",
        "average_precision",
        "balanced_accuracy",
        "healthy_false_positive_rate",
        "developing_recall",
        "faulty_recall",
    )
    for (policy, family), group in frame.groupby(["feature_policy", "model_family"], sort=True):
        result: dict[str, Any] = {
            "feature_policy": policy,
            "model_family": family,
            "bearing_folds": int(len(group)),
        }
        for metric in metrics:
            result[f"{metric}_mean"] = float(group[metric].mean())
            result[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summaries.append(result)
    return summaries


def benchmark_estimator(
    estimator: BaseEstimator,
    family: str,
    sample: np.ndarray,
    *,
    warmup: int = 20,
    repetitions: int = 200,
) -> dict[str, Any]:
    for _ in range(warmup):
        anomaly_scores(estimator, family, sample)
    timings_ms = []
    process = psutil.Process()
    before = process.memory_info().rss
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        anomaly_scores(estimator, family, sample)
        timings_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    return {
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "threads": 1,
        "batch_size": int(len(sample)),
        "warmup": warmup,
        "repetitions": repetitions,
        "p50_model_ms": float(np.percentile(timings_ms, 50)),
        "p95_model_ms": float(np.percentile(timings_ms, 95)),
        "rss_delta_bytes": int(max(0, process.memory_info().rss - before)),
    }


def benchmark_native_pipeline(
    bundle: dict[str, Any],
    matrix: np.ndarray,
    *,
    warmup: int = 10,
    repetitions: int = 100,
) -> dict[str, Any]:
    """Measure one in-memory one-second feature extraction plus model score."""

    sample_count = int(bundle["sample_rate_hz"] * bundle["window_seconds"])
    vibration = matrix[:sample_count, CHANNEL_MAPPING["vibration"]]
    audio = matrix[:sample_count, CHANNEL_MAPPING["audio"]]

    def score_once() -> None:
        arguments = {
            "sample_rate_hz": bundle["sample_rate_hz"],
            "frequency_bands_hz": bundle["frequency_bands_hz"],
        }
        vibration_features = extract_features(vibration, **arguments)
        audio_features = extract_features(audio, **arguments)
        merged = {
            **{f"vibration__{name}": value for name, value in vibration_features.items()},
            **{f"audio__{name}": value for name, value in audio_features.items()},
        }
        sample = np.asarray([[merged[column] for column in bundle["feature_columns"]]], dtype=float)
        anomaly_scores(bundle["estimator"], bundle["model_family"], sample)

    for _ in range(warmup):
        score_once()
    process = psutil.Process()
    before = process.memory_info().rss
    timings_ms = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        score_once()
        timings_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    return {
        "window_seconds": bundle["window_seconds"],
        "warmup": warmup,
        "repetitions": repetitions,
        "p50_preprocessing_plus_model_ms": float(np.percentile(timings_ms, 50)),
        "p95_preprocessing_plus_model_ms": float(np.percentile(timings_ms, 95)),
        "rss_delta_bytes": int(max(0, process.memory_info().rss - before)),
    }


def fit_deployment_model(
    table: pd.DataFrame,
    *,
    config: ExperimentConfig,
    policies: Sequence[str] = ("vibration", "audio", "fusion"),
) -> dict[str, Any]:
    """Select a deployable pipeline using grouped predictions on all primary bearings."""

    primary = table.loc[table["cohort"] == "primary"].reset_index(drop=True)
    outcomes: list[dict[str, Any]] = []
    for policy in policies:
        columns = _feature_columns(policy)
        for family in sorted({candidate.family for candidate in declared_candidates(config)}):
            candidates = [
                candidate for candidate in declared_candidates(config) if candidate.family == family
            ]
            selected, threshold, inner = choose_candidate(
                primary,
                columns,
                candidates,
                seed=config.evaluation.seed,
                inner_folds=config.evaluation.inner_folds,
            )
            selected_result = next(
                result for result in inner if result["candidate_id"] == selected.candidate_id
            )
            outcomes.append(
                {
                    "feature_policy": policy,
                    "feature_columns": columns,
                    "candidate": selected,
                    "threshold": threshold,
                    "selection_metrics": selected_result["metrics"],
                }
            )
    best = max(
        outcomes,
        key=lambda item: (
            item["selection_metrics"]["balanced_accuracy"],
            item["selection_metrics"]["average_precision"],
            -item["selection_metrics"]["healthy_false_positive_rate"],
            item["feature_policy"],
            item["candidate"].candidate_id,
        ),
    )
    x = primary[best["feature_columns"]].to_numpy(dtype=float)
    y = primary["binary_label"].to_numpy(dtype=int)
    estimator = _fit(
        _build_estimator(best["candidate"], config.evaluation.seed), best["candidate"], x, y
    )
    version_payload = {
        "dataset": config.dataset_version,
        "policy": best["feature_policy"],
        "candidate": asdict(best["candidate"]),
        "seed": config.evaluation.seed,
    }
    version = hashlib.sha256(
        json.dumps(version_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "bundle_version": 1,
        "model_version": f"fabguard-{version}",
        "estimator": estimator,
        "model_family": best["candidate"].family,
        "feature_policy": best["feature_policy"],
        "feature_columns": best["feature_columns"],
        "threshold": float(best["threshold"]),
        "candidate": asdict(best["candidate"]),
        "selection_metrics": best["selection_metrics"],
        "sample_rate_hz": config.feature.sample_rate_hz,
        "window_seconds": config.feature.window_seconds,
        "hop_seconds": config.feature.hop_seconds,
        "frequency_bands_hz": config.feature.band_edges_hz,
        "training_bearings": sorted(int(value) for value in primary["bearing_id"].unique()),
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_evaluation_report(
    destination: str | Path,
    *,
    summaries: Sequence[dict[str, Any]],
    folds: Sequence[dict[str, Any]],
    confound_summary: Sequence[dict[str, Any]],
    bundle: dict[str, Any],
    benchmark: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output / "primary_results.csv", index=False)
    pd.DataFrame(confound_summary).to_csv(output / "confound_results.csv", index=False)
    fold_rows = [
        {
            "fold": fold["fold"],
            "test_bearing_id": fold["test_bearing_id"],
            "feature_policy": fold["feature_policy"],
            "model_family": fold["selected_candidate"]["family"],
            **fold["metrics"],
        }
        for fold in folds
    ]
    pd.DataFrame(fold_rows).to_csv(output / "per_bearing_results.csv", index=False)

    labels = [f"{row.feature_policy}\n{row.model_family}" for row in summary_frame.itertuples()]
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    axis.bar(
        labels,
        summary_frame["balanced_accuracy_mean"],
        yerr=summary_frame["balanced_accuracy_std"],
        capsize=3,
    )
    axis.set(
        ylabel="Balanced accuracy (mean ± bearing-fold SD)",
        ylim=(0, 1.05),
        title="Primary 15-bearing leave-one-bearing-out comparison",
    )
    axis.tick_params(axis="x", rotation=35)
    figure.savefig(output / "primary_balanced_accuracy.png", dpi=160)
    plt.close(figure)

    selected = next(
        row
        for row in summaries
        if row["feature_policy"] == bundle["feature_policy"]
        and row["model_family"] == bundle["model_family"]
    )
    native = benchmark["native_pipeline"]
    report = f"""# FabGuard model card

## Delivered detector

The deployable model is a `{bundle["model_family"]}` using `{bundle["feature_policy"]}` features,
selected with grouped inner predictions on the 15 eligible training bearings. It emits a
window-level anomaly score and threshold decision; it does not diagnose a fault family or root
cause. Model version: `{bundle["model_version"]}`.

## Primary bearing-separated result

Across 15 outer leave-one-bearing-out folds, the matching policy/family achieved balanced
accuracy {selected["balanced_accuracy_mean"]:.3f} ± {selected["balanced_accuracy_std"]:.3f},
AUROC {selected["auroc_mean"]:.3f} ± {selected["auroc_std"]:.3f}, average precision
{selected["average_precision_mean"]:.3f} ± {selected["average_precision_std"]:.3f}, healthy
false-positive rate {selected["healthy_false_positive_rate_mean"]:.3f} ±
{selected["healthy_false_positive_rate_std"]:.3f}, developing recall
{selected["developing_recall_mean"]:.3f} ± {selected["developing_recall_std"]:.3f}, and faulty
recall {selected["faulty_recall_mean"]:.3f} ± {selected["faulty_recall_std"]:.3f}. Standard
deviations describe bearing-fold variability; they are not confidence intervals.

All audio channels passed the frozen technical audit, and paired audio features contribute to the
selected fusion model. Vibration-only and audio-only results remain in `primary_results.csv`.

## Confound audit

The separately labeled 20-bearing load-only Random Forest reached mean balanced accuracy
{confound_summary[0]["balanced_accuracy_mean"]:.3f} ±
{confound_summary[0]["balanced_accuracy_std"]:.3f}. This is not combined with the primary result;
it demonstrates that operating condition contains label information for the ball-fault bearings.

## Runtime measurement

On `{benchmark["processor"]}` with Python {benchmark["python"]}, one thread, batch size one,
{benchmark["warmup"]} warm-up calls, and {benchmark["repetitions"]} measured calls, model-only
latency was P50 {benchmark["p50_model_ms"]:.3f} ms and P95 {benchmark["p95_model_ms"]:.3f} ms.
For a one-second in-memory window, preprocessing plus model latency was P50
{native["p50_preprocessing_plus_model_ms"]:.3f} ms and P95
{native["p95_preprocessing_plus_model_ms"]:.3f} ms over {native["repetitions"]} repetitions. The
one-second acquisition window still dominates alert delay. These are laptop measurements, not
physical edge-device benchmarks.

## Limits

UORED contains short, selected laboratory recordings, accelerated degradation, only 20
independent bearings, and load/manufacturer/fault-family confounding. Cross-validation does not
create equipment diversity. Transfer to other machines is unverified. Snapshot replay does not
demonstrate remaining useful life, continuous deterioration, or field false alarms per hour.
"""
    (output / "MODEL_CARD.md").write_text(report, encoding="utf-8")


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run_training(
    manifest_path: str | Path,
    run_directory: str | Path,
    *,
    config_path: str | Path = "configs/experiment.json",
    report_directory: str | Path | None = "reports/model-evaluation",
) -> dict[str, Any]:
    experiment = ExperimentConfig.from_json(config_path)
    run_dir = Path(run_directory)
    run_dir.mkdir(parents=True, exist_ok=True)
    table = build_feature_table(
        load_manifest(manifest_path), output_path=run_dir / "features.parquet", config=experiment
    )
    predictions, folds = evaluate_lobo(table, config=experiment)
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    summaries = summarize_folds(folds)
    (run_dir / "metrics.json").write_text(
        json.dumps({"folds": folds, "summary": summaries}, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(summaries).to_csv(run_dir / "results.csv", index=False)
    confound_candidates = [
        candidate
        for candidate in declared_candidates(experiment)
        if candidate.family == "random_forest"
    ]
    confound_predictions, confound_folds = evaluate_lobo(
        table,
        policies=("load_only",),
        config=experiment,
        cohort="all",
        candidates=confound_candidates,
    )
    confound_predictions.to_csv(run_dir / "confound_predictions.csv", index=False)
    confound_summary = summarize_folds(confound_folds)
    (run_dir / "confound_metrics.json").write_text(
        json.dumps({"folds": confound_folds, "summary": confound_summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    bundle = fit_deployment_model(table, config=experiment)
    dump(bundle, run_dir / "model.joblib")
    model_sha256 = _sha256(run_dir / "model.joblib")
    (run_dir / "model.joblib.sha256").write_text(model_sha256 + "\n", encoding="ascii")
    sample = table.loc[table["cohort"] == "primary", bundle["feature_columns"]].iloc[:1]
    benchmark = benchmark_estimator(
        bundle["estimator"], bundle["model_family"], sample.to_numpy(dtype=float)
    )
    first_entry = next(
        entry
        for entry in load_manifest(manifest_path)
        if entry["cohort"] == "primary" and entry["technically_usable"]
    )
    benchmark["native_pipeline"] = benchmark_native_pipeline(
        bundle, read_numeric_table(first_entry["canonical_path"])
    )
    (run_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2) + "\n", encoding="utf-8"
    )
    split = {
        fold["fold"]: sorted(
            predictions.loc[predictions["bearing_id"] == fold["test_bearing_id"], "recording_id"]
            .unique()
            .tolist()
        )
        for fold in folds
    }
    (run_dir / "split.json").write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if report_directory is not None:
        write_evaluation_report(
            report_directory,
            summaries=summaries,
            folds=folds,
            confound_summary=confound_summary,
            bundle=bundle,
            benchmark=benchmark,
        )
    experiment.write_json(run_dir / "config.json")
    metadata = {
        "git_revision": _git_revision(),
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "config_sha256": _sha256(config_path),
        "split_sha256": _sha256(run_dir / "split.json"),
        "rows": int(len(table)),
        "recordings": int(table["recording_id"].nunique()),
        "bearings": int(table["bearing_id"].nunique()),
        "model_version": bundle["model_version"],
        "model_sha256": model_sha256,
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "run_directory": str(run_dir),
        "summary": summaries,
        "confound_summary": confound_summary,
        **metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/uored-v5.json")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--run-dir", default="runs/latest")
    parser.add_argument("--report-dir", default="reports/model-evaluation")
    args = parser.parse_args()
    print(
        json.dumps(
            run_training(
                args.manifest,
                args.run_dir,
                config_path=args.config,
                report_directory=args.report_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
