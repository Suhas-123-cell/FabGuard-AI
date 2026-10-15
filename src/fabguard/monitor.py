"""Monitor one live or recorded channel against its own commissioning baseline.

Input is a WAV file or raw 16-bit little-endian mono PCM on stdin, so any sensor or microphone that
a capture tool can stream works (see the README). Audio is resampled to 42 kHz and cut into the
product's one-second windows at a half-second hop. The first ``--baseline-seconds`` are taken as
this machine's healthy state and fit a healthy reference; every later window prints one JSON line
with a causal 3-of-5 alarm. Nothing checks that the baseline really is healthy, and a baseline fit
on a machine that is already damaged will learn that damage as normal.

Each alarm onset writes an evidence artifact in the replay schema. ``--submit`` sends it to the API
for an investigation; the investigation reads vibration evidence, so submission needs
``--modality vibration``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from collections.abc import Iterable, Iterator
from math import gcd
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import soundfile
from scipy.signal import resample_poly

from fabguard.alerting import DEFAULT_ALERT_RULE
from fabguard.calibration import MIN_BASELINE_WINDOWS, InsufficientBaseline
from fabguard.config import RuntimeSettings
from fabguard.signals import (
    DEFAULT_HOP_SECONDS,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_WINDOW_SECONDS,
    FEATURE_COLUMNS,
    assess_quality,
    extract_features,
)
from fabguard.train import HealthyReferenceDetector

SAMPLE_RATE_HZ = int(DEFAULT_SAMPLE_RATE_HZ)
WINDOW = int(DEFAULT_WINDOW_SECONDS * SAMPLE_RATE_HZ)
HOP = int(DEFAULT_HOP_SECONDS * SAMPLE_RATE_HZ)
THRESHOLD_QUANTILE = 0.99
CONTEXT_WINDOWS = 20  # about ten seconds of evidence before an alarm
NOT_RECORDED = {"is_usable": False, "issues": ["not_recorded"]}


def baseline_windows(seconds: float) -> int:
    return int((seconds - DEFAULT_WINDOW_SECONDS) / DEFAULT_HOP_SECONDS) + 1


def resample(samples: np.ndarray, rate: int) -> np.ndarray:
    divisor = gcd(SAMPLE_RATE_HZ, rate)
    return resample_poly(samples, SAMPLE_RATE_HZ // divisor, rate // divisor)


def read_wav(
    path: str | Path, channel: int = 0, chunk_seconds: float = 1.0
) -> Iterator[np.ndarray]:
    """Yield chunks at 42 kHz, so a file is processed exactly as a live stream would be."""
    data, rate = soundfile.read(path, dtype="float64", always_2d=True)
    if not 0 <= channel < data.shape[1]:
        raise ValueError(f"channel {channel} not in a {data.shape[1]}-channel file")
    signal = data[:, channel] if rate == SAMPLE_RATE_HZ else resample(data[:, channel], rate)
    step = int(chunk_seconds * SAMPLE_RATE_HZ)
    for start in range(0, len(signal), step):
        yield signal[start : start + step]


def read_pcm(stream: BinaryIO, rate: int = SAMPLE_RATE_HZ) -> Iterator[np.ndarray]:
    """Yield chunks of raw s16le mono PCM, resampled per chunk when ``rate`` differs."""
    leftover = b""
    while chunk := stream.read(rate * 2 // 4):  # about a quarter second
        chunk, leftover = leftover + chunk, b""
        if len(chunk) % 2:
            chunk, leftover = chunk[:-1], chunk[-1:]
        samples = np.frombuffer(chunk, dtype="<i2").astype(np.float64) / 32768.0
        yield samples if rate == SAMPLE_RATE_HZ else resample(samples, rate)


class Monitor:
    """Commission on the first windows, then score each new window causally."""

    def __init__(self, *, baseline_seconds: float, modality: str, source: str) -> None:
        self.needed = baseline_windows(baseline_seconds)
        if self.needed < MIN_BASELINE_WINDOWS:
            raise InsufficientBaseline(
                f"need {MIN_BASELINE_WINDOWS} baseline windows, {baseline_seconds} s gives "
                f"{self.needed}"
            )
        self.modality, self.source = modality, source
        self.buffer = np.empty(0)
        self.consumed = 0  # samples dropped from the front of the buffer
        self.next_start = 0
        self.baseline: list[np.ndarray] = []
        self.model: HealthyReferenceDetector | None = None
        self.threshold = np.nan
        self.model_version = ""
        self.recent: deque[bool] = deque(maxlen=DEFAULT_ALERT_RULE.window_count)
        self.history: deque[dict[str, Any]] = deque(maxlen=CONTEXT_WINDOWS)
        self.samples: deque[np.ndarray] = deque(maxlen=CONTEXT_WINDOWS)
        self.alarm = False

    def feed(self, chunk: np.ndarray) -> Iterator[dict[str, Any]]:
        self.buffer = np.concatenate([self.buffer, chunk])
        while self.next_start + WINDOW <= self.consumed + len(self.buffer):
            offset = self.next_start - self.consumed
            window = self.buffer[offset : offset + WINDOW]
            start, self.next_start = self.next_start, self.next_start + HOP
            yield from self._window(start, window)
        drop = self.next_start - self.consumed
        self.buffer, self.consumed = self.buffer[drop:], self.next_start

    def _window(self, start: int, window: np.ndarray) -> Iterator[dict[str, Any]]:
        features = extract_features(window, sample_rate_hz=SAMPLE_RATE_HZ)
        vector = np.array([features[name] for name in FEATURE_COLUMNS])
        seconds = start / SAMPLE_RATE_HZ
        if not np.all(np.isfinite(vector)) or not assess_quality(window).is_usable:
            self.recent.append(False)
            yield {"event": "unusable", "t_seconds": seconds}
            return
        if self.model is None:
            self.baseline.append(vector)
            if len(self.baseline) == self.needed:
                yield self._commission(seconds)
            return
        score = float(self.model.score_samples(vector[None])[0])
        abnormal = score >= self.threshold
        self.recent.append(abnormal)
        alarm = sum(self.recent) >= DEFAULT_ALERT_RULE.min_abnormal
        self.history.append(
            {
                "start_sample": start,
                "end_sample": start + WINDOW,
                f"{self.modality}_features": features,
                "anomaly_score": score,
                "threshold": self.threshold,
                "abnormal": bool(abnormal),
            }
        )
        self.samples.append(window.copy())
        event = {
            "event": "window",
            "t_seconds": seconds,
            "score": round(score, 4),
            "threshold": round(self.threshold, 4),
            "abnormal": bool(abnormal),
            "alarm": bool(alarm),
        }
        if alarm and not self.alarm:
            event["event"] = "alarm"
            event["artifact"] = self.evidence()
        self.alarm = alarm
        yield event

    def _commission(self, seconds: float) -> dict[str, Any]:
        baseline = np.vstack(self.baseline)
        self.model = HealthyReferenceDetector().fit(baseline)
        fitted = self.model.score_samples(baseline)
        self.threshold = float(np.quantile(fitted, THRESHOLD_QUANTILE))
        digest = hashlib.sha256(baseline.tobytes()).hexdigest()
        self.model_version = f"commissioned-{digest[:16]}"
        return {
            "event": "commissioned",
            "t_seconds": seconds,
            "baseline_windows": len(self.baseline),
            "threshold": round(self.threshold, 4),
            "model_version": self.model_version,
        }

    def evidence(self) -> dict[str, Any]:
        """The last ``CONTEXT_WINDOWS`` windows, with the alarm window selected."""
        first = self.history[0]["start_sample"]
        other = "audio" if self.modality == "vibration" else "vibration"
        windows = [
            {
                **item,
                f"{other}_features": {},
                "window_index": index,
                "start_sample": item["start_sample"] - first,
                "end_sample": item["end_sample"] - first,
            }
            for index, item in enumerate(self.history)
        ]
        selected = windows[-1]
        # Windows overlap by half; one hop of each plus the whole last window is the raw span.
        samples = np.concatenate([w[:HOP] for w in list(self.samples)[:-1]] + [self.samples[-1]])
        digest = hashlib.sha256(f"{self.source}:{self.model_version}:{first}".encode()).hexdigest()
        quality = assess_quality(samples, sample_rate_hz=SAMPLE_RATE_HZ).to_dict()
        return {
            "artifact_id": f"art_{digest[:32]}",
            "recording_id": self.source,
            "recorded_data_replay": False,
            "stream_offset_seconds": first / SAMPLE_RATE_HZ,
            "model_version": self.model_version,
            "model_family": "healthy_reference",
            "feature_policy": self.modality,
            "preprocessing_version": "fabguard-features-v1",
            "evaluation_scope": "per-machine commissioning baseline; not validated on this machine",
            "sample_rate_hz": float(SAMPLE_RATE_HZ),
            "quality": {self.modality: quality, other: NOT_RECORDED},
            "prediction": {
                "score": selected["anomaly_score"],
                "threshold": selected["threshold"],
                "decision": "abnormal",
                "alert_rule": DEFAULT_ALERT_RULE.to_dict(),
                "threshold_source": "machine_baseline",
                "selected_window_index": len(windows) - 1,
            },
            "windows": windows,
        }


def run(
    chunks: Iterable[np.ndarray],
    monitor: Monitor,
    *,
    output_directory: str | Path,
    submit: Any = None,
) -> Iterator[dict[str, Any]]:
    for chunk in chunks:
        for event in monitor.feed(chunk):
            if "artifact" in event:
                artifact = event.pop("artifact")
                destination = Path(output_directory)
                destination.mkdir(parents=True, exist_ok=True)
                path = destination / f"{artifact['artifact_id']}.json"
                path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
                artifact["artifact_path"] = str(path.resolve())
                event["artifact_path"] = artifact["artifact_path"]
                if submit is not None:
                    event["submission"] = submit(artifact)
            yield event


def main() -> None:
    settings = RuntimeSettings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="a WAV file, or - for raw s16le mono PCM on stdin")
    parser.add_argument("--rate", type=int, default=SAMPLE_RATE_HZ, help="stdin sample rate")
    parser.add_argument("--channel", type=int, default=0, help="WAV channel to monitor")
    parser.add_argument("--modality", choices=["vibration", "audio"], default="audio")
    parser.add_argument("--baseline-seconds", type=float, default=120.0)
    parser.add_argument("--machine", default="machine-1", help="opaque label for evidence")
    parser.add_argument("--output-dir", default=settings.artifact_root)
    parser.add_argument("--submit", action="store_true", help="submit alarms using env settings")
    args = parser.parse_args()
    submit = None
    if args.submit:
        if args.modality != "vibration":
            parser.error("--submit needs --modality vibration; investigations read vibration")
        if not settings.producer_token:
            parser.error("FABGUARD_PRODUCER_TOKEN is required with --submit")
        from fabguard.replay import register_and_submit

        def submit(artifact: dict[str, Any]) -> dict[str, Any]:
            return register_and_submit(
                artifact,
                database_url=settings.database_url,
                api_url=settings.api_url,
                api_token=settings.producer_token,
                artifact_root=settings.artifact_root,
                graph_version=settings.graph_version,
                prompt_version=settings.prompt_version,
            )

    chunks = (
        read_pcm(sys.stdin.buffer, args.rate)
        if args.source == "-"
        else read_wav(args.source, args.channel)
    )
    monitor = Monitor(
        baseline_seconds=args.baseline_seconds, modality=args.modality, source=args.machine
    )
    try:
        for event in run(chunks, monitor, output_directory=args.output_dir, submit=submit):
            print(json.dumps(event), flush=True)
    except KeyboardInterrupt:
        pass
    if monitor.model is None:
        sys.exit(f"stream ended after {len(monitor.baseline)} of {monitor.needed} baseline windows")


if __name__ == "__main__":
    main()
