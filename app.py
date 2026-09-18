"""FabGuard's local recorded-data replay and incident review demo."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

from fabguard.replay import replay_recording

ROOT = Path(__file__).resolve().parent
MANIFEST = Path(os.getenv("FABGUARD_MANIFEST", ROOT / "manifests/uored_vafcls_v5.json"))
MODEL = Path(os.getenv("FABGUARD_MODEL", ROOT / "runs/uored-v5-seed17/model.joblib"))
API_URL = os.getenv("FABGUARD_API_URL", "http://127.0.0.1:8000")
API_TOKEN = os.getenv("FABGUARD_REVIEWER_TOKEN", "")

st.set_page_config(page_title="FabGuard AI", page_icon="⚙️", layout="wide")
st.title("FabGuard AI")
st.caption("Recorded laboratory-data replay — not live machine monitoring or a causal diagnosis")


def api_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


view = st.sidebar.radio("View", ("Replay", "Incident", "Results"))

if view == "Replay":
    st.subheader("Replay")
    st.write("Run the frozen local detector on one audited recording, addressed only by opaque ID.")
    if not MANIFEST.exists() or not MODEL.exists():
        st.warning(
            "Run the data audit and training commands first; the manifest or model is missing."
        )
    else:
        manifest = load_json(MANIFEST)
        recording_ids = [
            entry["recording_id"] for entry in manifest["entries"] if entry["technically_usable"]
        ]
        selected = st.selectbox("Opaque recording ID", recording_ids)
        if st.button("Run recorded-data replay", type="primary"):
            with st.spinner("Validating channels, extracting windows, and scoring…"):
                replay = replay_recording(
                    manifest_path=MANIFEST,
                    model_path=MODEL,
                    recording_id=selected,
                    output_directory=ROOT / "runs/replays",
                )
            st.session_state["last_replay"] = replay
        replay = st.session_state.get("last_replay")
        if replay:
            prediction = replay["prediction"]
            first, second, third = st.columns(3)
            first.metric("Decision", prediction["decision"])
            second.metric("Anomaly score", f"{prediction['score']:.4f}")
            third.metric("Threshold", f"{prediction['threshold']:.4f}")
            st.caption(
                f"Model {replay['model_version']} · {replay['model_family']} · "
                f"{replay['feature_policy']} · {replay['evaluation_scope']}"
            )
            preview = replay["preview"]
            wave = pd.DataFrame(
                {
                    "time_seconds": preview["time_seconds"],
                    "vibration": preview["vibration"],
                }
            ).set_index("time_seconds")
            spectrum = pd.DataFrame(
                {
                    "frequency_hz": preview["frequency_hz"],
                    "magnitude": preview["spectrum_magnitude"],
                }
            ).set_index("frequency_hz")
            left, right = st.columns(2)
            left.markdown("**Vibration waveform (first two seconds)**")
            left.line_chart(wave)
            right.markdown("**Vibration spectrum (quality view)**")
            right.line_chart(spectrum)
            with st.expander("Provenance and quality"):
                st.json(
                    {
                        key: replay[key]
                        for key in (
                            "artifact_id",
                            "recording_id",
                            "recorded_data_replay",
                            "model_version",
                            "preprocessing_version",
                            "quality",
                        )
                    }
                )
            with st.expander("Window scores"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "window": item["window_index"],
                                "start_sample": item["start_sample"],
                                "end_sample": item["end_sample"],
                                "score": item["anomaly_score"],
                                "threshold": item["threshold"],
                                "abnormal": item["abnormal"],
                            }
                            for item in replay["windows"]
                        ]
                    ),
                    use_container_width=True,
                )

elif view == "Incident":
    st.subheader("Incident")
    st.write("This view reads durable state through FastAPI. It never starts work on a page rerun.")
    incident_id = st.text_input("Incident UUID")
    if st.button("Load incident", disabled=not incident_id):
        try:
            response = httpx.get(
                f"{API_URL.rstrip('/')}/v1/incidents/{incident_id}",
                headers=api_headers(),
                timeout=10,
            )
            response.raise_for_status()
            st.session_state["incident"] = response.json()
        except httpx.HTTPError as error:
            st.error(f"API request failed: {error}")
    incident = st.session_state.get("incident")
    if incident:
        st.caption(
            f"Evidence {incident['evidence_version']} · model {incident['model_version']} · "
            f"graph {incident['graph_version']} · prompt {incident['prompt_version']}"
        )
        st.json(
            {"id": incident["id"], "state": incident["state"], "prediction": incident["prediction"]}
        )
        for report in incident.get("reports", []):
            st.markdown(f"### Report revision {report['revision']}")
            content = report["content"]
            st.write("Tool trace", content.get("trace", []))
            st.write("Observations", content.get("observations", []))
            st.write("Predictions", content.get("predictions", []))
            st.write("Hypotheses", content.get("hypotheses", []))
            st.write("Guidance and citations", content.get("guidance", []))
            st.write("Ticket preview", report.get("ticket_draft"))
            if report.get("ticket_draft") and st.button(
                "Approve concrete ticket draft", key=f"approve-{report['revision']}"
            ):
                try:
                    approval = httpx.post(
                        f"{API_URL.rstrip('/')}/v1/incidents/{incident['id']}/approve",
                        headers={
                            **api_headers(),
                            "Idempotency-Key": f"ui:{incident['id']}:{report['revision']}",
                        },
                        json={
                            "report_revision": report["revision"],
                            "action": "create_internal_ticket",
                        },
                        timeout=10,
                    )
                    approval.raise_for_status()
                    st.success("Ticket transaction completed idempotently.")
                    st.json(approval.json())
                except httpx.HTTPError as error:
                    st.error(f"Approval failed: {error}")

else:
    st.subheader("Results")
    st.write("Primary bearing-separated results are shown before architecture details.")
    result_path = ROOT / "runs/uored-v5-seed17/results.csv"
    confound_path = ROOT / "runs/uored-v5-seed17/confound_metrics.json"
    ablation_path = ROOT / "runs/investigation-eval/results.json"
    if result_path.exists():
        results = pd.read_csv(result_path)
        st.markdown("### Primary 15-bearing LOBO")
        st.dataframe(results, use_container_width=True)
    else:
        st.info("Primary training results are not present yet.")
    if confound_path.exists():
        st.markdown("### Separate 20-bearing load-confound audit")
        st.json(load_json(confound_path)["summary"])
    if ablation_path.exists():
        st.markdown("### Fixed versus adaptive investigation")
        st.json(load_json(ablation_path)["summary"])
    st.warning(
        "Short selected laboratory recordings, accelerated degradation, load/manufacturer "
        "confounding, and only 20 independent bearings limit transfer claims. This does not "
        "demonstrate remaining useful life, continuous deterioration, or field false alarms."
    )
