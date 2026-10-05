"""Second-dataset check on the Paderborn (KAt) bearing data, vibration only.

The UORED fusion model cannot be applied here: there is no audio channel and the sensor and
sample rate differ. This replicates the *method* instead: the same features, model grid, nested
grouped evaluation, and 3-of-5 alert rule on a different rig with real accelerated-life damage.
Each bearing is either healthy or damaged, so every outer fold holds out one of the six healthy
bearings plus a sixth of the damaged ones.

Raw archives are processed one bearing at a time and deleted after feature extraction, because
the extracted recordings take about 22 GB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .alerting import DEFAULT_ALERT_RULE, AlertRule, persistent_score, rule_sensitivity
from .calibration import InsufficientBaseline, calibrated_threshold
from .config import ExperimentConfig
from .signals import extract_recording_features
from .train import FEATURE_COLUMNS, evaluate_lobo, summarize_folds

OFFICIAL_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/{code}.rar"
ZENODO_URL = "https://zenodo.org/records/15845309/files/{code}.rar?download=1"
SAMPLE_RATE_HZ = 64_000

# Lessmeier et al., PHME 2016, tables 4-6. KA = outer race, KI = inner race, KB = both.
HEALTHY = ("K001", "K002", "K003", "K004", "K005", "K006")
ARTIFICIAL = (
    "KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09",
    "KI01", "KI03", "KI05", "KI07", "KI08",
)  # fmt: skip
REAL = (
    "KA04", "KA15", "KA16", "KA22", "KA30", "KB23", "KB24", "KB27",
    "KI04", "KI14", "KI16", "KI17", "KI18", "KI21",
)  # fmt: skip
BEARINGS = HEALTHY + ARTIFICIAL + REAL
FAULT_FAMILY = {"K0": "none", "KA": "outer_race", "KI": "inner_race", "KB": "combined"}

# Zenodo record 15845309 checksums; K001 from the official host matched. KA08 is not on the
# mirror, so it is checked only by extracting its 80 recordings.
MD5 = {
    "K001": "af6dc58283d356cd438e7738a2d525b8", "K002": "88631bc0081a6470846d18189d46381d",
    "K003": "08b61792e1058dc47dd21071666bca2c", "K004": "5b9fde1c6d754806539f9a890ee6bbdd",
    "K005": "28d91412ad14583fbdfda958b64a8267", "K006": "b36389294fc5a21539d9fcfacfa3604a",
    "KA01": "5627ec9320199078205db5325c1c2f84", "KA03": "0bba6927ac1732d23934a909af4c1fc6",
    "KA04": "f87503ed69a624ba6e1d9e573e6c09fb", "KA05": "22b6ffd9ff083f084918c34f6ddcc538",
    "KA06": "be58a820fee4851b2c16a97a88c282cb", "KA07": "185e3de688149ebabf20c51d80fd7dd8",
    "KA09": "de9392fdbf878171a429c2451d2696fd", "KA15": "db518f97ba5bcaedb09c648a317431e0",
    "KA16": "ef848f5621f9c10523c313131d1a4c21", "KA22": "7a16a5bc88d5fe1d83d169f88f0709e8",
    "KA30": "715c0cf7862a7ba83e7c6a6fb5c91a61", "KB23": "bb87c72669baa9dbbefd10b5294ffcc8",
    "KB24": "c60ce8867c824583540794ae00cce333", "KB27": "965dd4af26391d7def4638c8b31a87e7",
    "KI01": "03429a4ab5e3768b7395d1654e55bead", "KI03": "26f0452f00ab698a4e657e27b7a249b3",
    "KI04": "a5a634fcdcb33a8501e7411d30441c6d", "KI05": "c777459ddc7a1d38df7155f09e0ce782",
    "KI07": "c26144f53ebcc34562bdc2e3a0ef1087", "KI08": "f546b3481ffdcd80f0d14a797db48deb",
    "KI14": "13603834975435ca10eb08de6d1840dc", "KI16": "db14eae04a213afd0d1ca99d3d850342",
    "KI17": "d1055fe5cc44948e5d3a39f2e0db9dbe", "KI18": "db5da511e9240dbc56ec1e850fd91699",
    "KI21": "911ce3e37a86d258818e3e93274752a2",
}  # fmt: skip

RECORDING = re.compile(
    r"^N(?P<rpm>\d{2})_M(?P<torque>\d{2})_F(?P<force>\d{2})_(?P<code>K[0-9AIB]\d{2,3})_(?P<index>\d+)$"
)
MAX_UNREADABLE_PER_BEARING = 2  # a corrupt recording is excluded and reported, not guessed
BASELINE_MEASUREMENTS = range(1, 11)  # calibration uses measurements 1-10, tests on 11-20


def bearing_number(code: str) -> int:
    """Readable integer IDs: K001 -> 1, KA01 -> 101, KI21 -> 221, KB23 -> 323."""
    return {"K0": 0, "KA": 100, "KI": 200, "KB": 300}[code[:2]] + int(code[2:])


def parse_recording(stem: str) -> dict[str, Any]:
    match = RECORDING.match(stem)
    if not match:
        raise ValueError(f"unexpected Paderborn recording name {stem!r}")
    return {
        "code": match["code"],
        "rpm": int(match["rpm"]) * 100,
        "torque_nm": int(match["torque"]) / 10,
        "radial_force_n": int(match["force"]) * 100,
        "measurement": int(match["index"]),
        "condition": stem.rsplit("_", 2)[0],
    }


def read_vibration(path: str | Path) -> np.ndarray:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    record = next(value for key, value in data.items() if not key.startswith("__"))
    for channel in np.atleast_1d(record.Y):
        if channel.Name == "vibration_1":
            return np.asarray(channel.Data, dtype=float)
    raise ValueError(f"{path} has no vibration_1 channel")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(code: str, destination: Path, *, source: str = "official", retries: int = 4) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{code}.rar"
    expected = MD5.get(code)
    if target.exists() and (expected is None or _md5(target) == expected):
        return target
    url = (ZENODO_URL if source == "zenodo" and code in MD5 else OFFICIAL_URL).format(code=code)
    partial = target.with_suffix(".part")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "FabGuard-AI/0.1"})
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                partial.open("wb") as sink,
            ):
                shutil.copyfileobj(response, sink, 1 << 20)
            if expected is not None and _md5(partial) != expected:
                raise OSError(f"{code}.rar failed its pinned MD5 check")
            partial.replace(target)
            return target
        except OSError:
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def extract(archive: Path, destination: Path) -> list[Path]:
    for tool, arguments in (
        ("bsdtar", ["-xf", str(archive), "-C", str(destination)]),
        ("unrar", ["x", "-o+", "-idq", str(archive), f"{destination}/"]),
        ("7z", ["x", "-y", f"-o{destination}", str(archive)]),
    ):
        if shutil.which(tool):
            subprocess.run([tool, *arguments], check=True)
            return sorted(destination.rglob("*.mat"))
    raise RuntimeError("install bsdtar (libarchive-tools), unrar, or 7z to unpack RAR archives")


def bearing_features(
    code: str, mat_files: list[Path], config: ExperimentConfig
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Return window features plus the recordings excluded because they could not be read."""
    if len(mat_files) != 80:
        raise ValueError(f"{code} unpacked {len(mat_files)} recordings, expected 80")
    healthy = code in HEALTHY
    origin = "none" if healthy else ("artificial" if code in ARTIFICIAL else "real")
    rows = []
    excluded: list[dict[str, str]] = []
    for path in mat_files:
        meta = parse_recording(path.stem)
        if meta["code"] != code:
            raise ValueError(f"{path.name} does not belong to {code}")
        try:
            vibration = read_vibration(path)
        except (TypeError, ValueError) as error:  # scipy raises both for malformed structs
            reason = f"{type(error).__name__}: {error}"
            excluded.append({"recording_id": path.stem, "reason": reason})
            if len(excluded) > MAX_UNREADABLE_PER_BEARING:
                raise ValueError(f"{code} has {len(excluded)} unreadable recordings") from error
            continue
        windows = extract_recording_features(
            vibration,
            sample_rate_hz=SAMPLE_RATE_HZ,
            window_seconds=config.feature.window_seconds,
            hop_seconds=config.feature.hop_seconds,
            frequency_bands_hz=config.feature.band_edges_hz,
        )
        for window in windows:
            row = {
                "recording_id": path.stem,
                "bearing_id": bearing_number(code),
                "bearing_code": code,
                "health_state": "healthy" if healthy else "faulty",
                "damage_origin": origin,
                "fault_family": FAULT_FAMILY[code[:2]],
                "manufacturer": "unknown",
                "cohort": "primary",
                "binary_label": 0 if healthy else 1,
                "window_index": int(window["window_index"]),
                "start_sample": int(window["start_sample"]),
                "end_sample": int(window["end_sample"]),
                "load_mean": float(meta["radial_force_n"]),
                "rpm_mean": float(meta["rpm"]),
                "condition": meta["condition"],
                "measurement": meta["measurement"],
            }
            row.update({f"vibration__{name}": float(window[name]) for name in FEATURE_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows), excluded


def build_feature_table(
    root: Path, config: ExperimentConfig, *, source: str = "official", keep_raw: bool = False
) -> pd.DataFrame:
    """Download, unpack, featurize, and clean up one bearing at a time; resumable."""
    cache = root / "features"
    cache.mkdir(parents=True, exist_ok=True)
    for number, code in enumerate(BEARINGS, start=1):
        cached = cache / f"{code}.parquet"
        if cached.exists():
            continue
        started = time.monotonic()
        archive = download(code, root / "rar", source=source)
        with tempfile.TemporaryDirectory(dir=root) as scratch:
            table, excluded = bearing_features(code, extract(archive, Path(scratch)), config)
        (cache / f"{code}.excluded.json").write_text(json.dumps(excluded, indent=2) + "\n")
        table.to_parquet(cached, index=False)
        if not keep_raw:
            archive.unlink()
        elapsed = time.monotonic() - started
        note = f", excluded {len(excluded)} unreadable" if excluded else ""
        progress = f"[{number:02d}/{len(BEARINGS)}] {code}"
        print(f"{progress}: {len(table)} windows in {elapsed:.0f}s{note}")
    return pd.concat(
        [pd.read_parquet(cache / f"{code}.parquet") for code in BEARINGS], ignore_index=True
    )


def outer_folds() -> list[tuple[int, ...]]:
    """Six folds: one healthy bearing each, damaged bearings dealt round-robin."""
    damaged = sorted(ARTIFICIAL + REAL)
    return [
        (bearing_number(healthy), *(bearing_number(code) for code in damaged[index::6]))
        for index, healthy in enumerate(HEALTHY)
    ]


def _recording_alarms(windows: pd.DataFrame, threshold: float) -> pd.Series:
    scores = windows.sort_values("window_index").groupby("recording_id").anomaly_score
    return scores.apply(
        lambda s: persistent_score(s.to_numpy(), DEFAULT_ALERT_RULE)[0] >= threshold
    )


def calibration_check(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per fold: does the held-out healthy bearing's own baseline cut false alarms, at what recall?

    Baseline = that bearing's measurements 1-10 across its four operating conditions; test =
    measurements 11-20 of the healthy bearing and of the damaged bearings in the same fold. The
    damaged bearings stand in for "the same machine later degraded"; that is a proxy, not a
    longitudinal recording.
    """
    frame = predictions.copy()
    frame["measurement"] = frame.recording_id.map(lambda r: parse_recording(r)["measurement"])
    rows = []
    for fold, group in frame.groupby("fold"):
        healthy = group[group.binary_label == 0]
        baseline = healthy[healthy.measurement.isin(BASELINE_MEASUREMENTS)].anomaly_score
        test = group[~group.measurement.isin(BASELINE_MEASUREMENTS)]
        labels = test.groupby("recording_id").agg(
            label=("binary_label", "first"), origin=("damage_origin", "first")
        )
        fleet = float(group.threshold.iloc[0])
        try:
            machine = calibrated_threshold(baseline, fleet)
        except InsufficientBaseline as error:
            rows.append({"fold": fold, "error": str(error)})
            continue
        row: dict[str, Any] = {
            "fold": fold,
            "healthy_bearing": healthy.bearing_code.iloc[0],
            "baseline_windows": len(baseline),
            "fleet_threshold": fleet,
            "machine_threshold": machine,
        }
        for name, threshold in (("fleet", fleet), ("machine", machine)):
            alarms = _recording_alarms(test, threshold).reindex(labels.index)
            row[f"{name}_healthy_alarmed"] = float(alarms[labels.label == 0].mean())
            for origin in ("artificial", "real"):
                selected = alarms[labels.origin == origin]
                row[f"{name}_{origin}_recall"] = float(selected.mean()) if len(selected) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run(root: Path, output: Path, *, source: str = "official", keep_raw: bool = False) -> dict:
    config = ExperimentConfig()
    table = build_feature_table(root, config, source=source, keep_raw=keep_raw)
    output.mkdir(parents=True, exist_ok=True)
    folds = outer_folds()
    predictions, fold_results = evaluate_lobo(
        table, policies=("vibration",), config=config, outer_folds=folds
    )
    fold_of = {bearing: f"bearings-{held[0]:03d}" for held in folds for bearing in held}
    extra = ["recording_id", "window_index", "bearing_code", "damage_origin", "condition"]
    predictions = predictions.merge(table[extra], on=["recording_id", "window_index"])
    predictions["fold"] = predictions.bearing_id.map(fold_of)
    predictions.to_csv(output / "predictions.csv.gz", index=False)
    (output / "folds.json").write_text(json.dumps(fold_results, indent=2, default=str) + "\n")
    summary = pd.DataFrame(summarize_folds(fold_results))
    summary.to_csv(output / "summary.csv", index=False)
    rules = [AlertRule(1, 1), AlertRule(2, 2), AlertRule(3, 5), AlertRule(5, 7)]
    sensitivity, calibration = [], []
    for family, group in predictions.groupby("model_family"):
        sensitivity.append(rule_sensitivity(group, rules).assign(model_family=family))
        calibration.append(calibration_check(group).assign(model_family=family))
    pd.concat(sensitivity).to_csv(output / "alert_rule_sensitivity.csv", index=False)
    pd.concat(calibration).to_csv(output / "calibration.csv", index=False)
    result = {
        "dataset": "Paderborn KAt bearing data (CC BY 4.0), vibration_1 at 64 kHz",
        "bearings": len(BEARINGS),
        "windows": len(table),
        "outer_folds": [list(held) for held in folds],
        "alert_rule": DEFAULT_ALERT_RULE.to_dict(),
        "excluded_recordings": [
            item
            for path in sorted((root / "features").glob("*.excluded.json"))
            for item in json.loads(path.read_text())
        ],
        "summary": summary.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/paderborn"))
    parser.add_argument("--output", type=Path, default=Path("reports/paderborn-evaluation"))
    parser.add_argument("--source", choices=("official", "zenodo"), default="official")
    parser.add_argument("--keep-raw", action="store_true", help="keep downloaded archives")
    args = parser.parse_args()
    result = run(args.root, args.output, source=args.source, keep_raw=args.keep_raw)
    print(json.dumps(result["summary"], indent=2, default=float))


if __name__ == "__main__":
    main()
