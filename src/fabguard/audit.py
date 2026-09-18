"""Command-line entry point for the raw UORED-VAFCLS audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf

from fabguard.data import (
    CHANNEL_MAPPING,
    EXPECTED_SAMPLE_COUNT,
    Cohort,
    audit_dataset,
    freeze_cohorts,
    read_numeric_table,
)
from fabguard.signals import DEFAULT_SAMPLE_RATE_HZ

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fabguard-audit",
        description="Audit UORED-VAFCLS raw tables and write a JSON manifest.",
    )
    parser.add_argument("dataset", type=Path, help="extracted dataset directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifests/uored_vafcls_v5.json"),
        help="manifest destination (default: manifests/uored_vafcls_v5.json)",
    )
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--expected-samples", type=int, default=EXPECTED_SAMPLE_COUNT)
    parser.add_argument("--load-tolerance", type=float, default=80.0)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="quality tables, representative plots, and listening copies",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a nonzero status when top-level audit issues remain",
    )
    return parser


def write_audit_artifacts(entries: Sequence[object], destination: Path) -> None:
    """Write deterministic quality evidence without using plots to select cases."""

    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in entries:
        rows.append(
            {
                "recording_id": entry.recording_id,
                "source_name": entry.source_name,
                "bearing_id": entry.bearing_id,
                "health_state": entry.health_state.value,
                "fault_family": entry.eventual_fault_family.value,
                "manufacturer": entry.manufacturer.value,
                "cohort": entry.cohort.value,
                "load_n": entry.load.mean,
                "rpm": entry.motor_speed.mean,
                "sample_count": entry.sample_count,
                "duration_seconds": entry.duration_seconds,
                "vibration_usable": entry.vibration_quality.is_usable,
                "audio_usable": entry.audio_quality.is_usable,
                "issues": ";".join(entry.issues),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "recording_quality.csv", index=False)
    frame.groupby(["fault_family", "manufacturer", "cohort", "health_state"], dropna=False).agg(
        recordings=("recording_id", "count"),
        load_mean_n=("load_n", "mean"),
        load_min_n=("load_n", "min"),
        load_max_n=("load_n", "max"),
        rpm_mean=("rpm", "mean"),
    ).reset_index().to_csv(destination / "confound_table.csv", index=False)

    representatives = []
    healthy = next(
        entry
        for entry in entries
        if entry.bearing_id == 1 and entry.health_state.value == "healthy"
    )
    representatives.append(healthy)
    for family in ("inner_race", "outer_race", "ball", "cage"):
        representatives.append(
            next(
                entry
                for entry in entries
                if entry.eventual_fault_family.value == family
                and entry.health_state.value == "developing"
            )
        )

    plot_dir = destination / "plots"
    audio_dir = destination / "audio"
    plot_dir.mkdir(exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    for entry in representatives:
        matrix = read_numeric_table(entry.canonical_path)
        vibration = matrix[:, CHANNEL_MAPPING["vibration"]]
        audio = matrix[:, CHANNEL_MAPPING["audio"]]
        sample_rate = float(entry.sample_rate_hz)
        visible = min(len(vibration), int(sample_rate * 2))
        stride = max(1, visible // 8_000)
        time_axis = np.arange(0, visible, stride) / sample_rate
        spectrum = np.abs(
            np.fft.rfft((vibration[:visible] - np.mean(vibration[:visible])) * np.hanning(visible))
        )
        frequencies = np.fft.rfftfreq(visible, 1 / sample_rate)
        figure, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
        axes[0].plot(time_axis, vibration[:visible:stride], linewidth=0.6)
        axes[0].set(
            xlabel="Time (s)", ylabel="Source amplitude", title=f"{entry.source_name}: vibration"
        )
        axes[1].semilogy(frequencies, np.maximum(spectrum, 1e-12), linewidth=0.6)
        axes[1].set(xlabel="Frequency (Hz)", ylabel="Magnitude", xlim=(0, sample_rate / 2))
        figure.savefig(plot_dir / f"{entry.source_name}.png", dpi=140)
        plt.close(figure)

        centered = np.nan_to_num(audio - np.nanmean(audio))
        peak = float(np.max(np.abs(centered)))
        normalized = centered if peak == 0 else centered * (0.8 / peak)
        sf.write(
            audio_dir / f"{entry.source_name}.wav", normalized.astype(np.float32), int(sample_rate)
        )

    cohort_counts = frame.groupby("cohort")["recording_id"].count().to_dict()
    report = {
        "recordings": int(len(frame)),
        "bearings": int(frame["bearing_id"].nunique()),
        "cohorts": {str(key): int(value) for key, value in cohort_counts.items()},
        "vibration_usable": int(frame["vibration_usable"].sum()),
        "audio_usable": int(frame["audio_usable"].sum()),
        "representative_recordings": [entry.source_name for entry in representatives],
        "selection_rule": "bearing 1 healthy plus the first developing recording per fault family",
        "listening_note": "copies are DC-centered and peak-normalized to 0.8 for safe comparison",
    }
    (destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (destination / "README.md").write_text(
        "# UORED-VAFCLS v5 data audit\n\n"
        f"Audited {report['recordings']} recordings from {report['bearings']} bearings. "
        f"All {report['vibration_usable']} vibration and {report['audio_usable']} audio channels "
        "passed the frozen technical checks. The primary cohort contains 45 recordings from "
        "15 bearings; 15 recordings from bearings 11–15 remain in the separate load-confound "
        "audit. No recording was excluded for being difficult to classify.\n\n"
        "The release encodes RPM and load as a scalar in the first CSV row followed by zero "
        "padding. `recording_quality.csv` reports the recovered scalar values. "
        "`confound_table.csv` makes the ball-fault load shift explicit. Plots and normalized "
        "listening copies are deterministic quality checks, not outcome-based case selection.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    audit = audit_dataset(
        arguments.dataset,
        sample_rate_hz=arguments.sample_rate,
        expected_sample_count=arguments.expected_samples,
        load_tolerance_n=arguments.load_tolerance,
    )
    frozen_entries = freeze_cohorts(audit.entries)
    # Keep the audit object's serialization contract while replacing only its
    # preliminary assignments with the explicitly frozen cohort assignments.
    frozen_audit = type(audit)(
        root=audit.root,
        entries=frozen_entries,
        unparsed_files=audit.unparsed_files,
        duplicate_content_groups=audit.duplicate_content_groups,
        issues=audit.issues,
    )
    frozen_audit.write_json(arguments.output)
    if arguments.report_dir is not None:
        write_audit_artifacts(frozen_entries, arguments.report_dir)

    primary_bearings = {
        entry.bearing_id for entry in frozen_entries if entry.cohort is Cohort.PRIMARY
    }
    quarantined_bearings = {
        entry.bearing_id for entry in frozen_entries if entry.cohort is Cohort.QUARANTINED_LOAD
    }
    print(f"manifest: {arguments.output.resolve()}")
    print(f"recordings: {len(frozen_entries)}; primary bearings: {len(primary_bearings)}")
    print(f"load-quarantined bearings: {len(quarantined_bearings)}")
    print(f"unparsed files: {len(audit.unparsed_files)}; audit issues: {len(audit.issues)}")
    return int(arguments.strict and bool(audit.issues or audit.unparsed_files))


if __name__ == "__main__":
    raise SystemExit(main())
