from pathlib import Path

from streamlit.testing.v1 import AppTest

from fabguard.graph import InvestigationGraph, build_langgraph, postgres_saver_factory
from fabguard.llm import LocalTemplateLLM
from fabguard.retrieval import postgres_retriever
from fabguard.worker import InvestigationWorker, RuntimeGraphRunner


def click(app, label):
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception


def test_streamlit_replay_incident_results_and_revision(
    live_api, replay_fixture, tmp_path, monkeypatch
):
    url, settings, db = live_api
    environment = {
        "FABGUARD_DATABASE_URL": settings.database_url,
        "FABGUARD_CHECKPOINT_DATABASE_URL": settings.checkpoint_url,
        "FABGUARD_ARTIFACT_ROOT": str(settings.artifact_root),
        "FABGUARD_API_URL": url,
        "FABGUARD_MANIFEST": str(tmp_path / "manifest.json"),
        "FABGUARD_MODEL": str(tmp_path / "model.joblib"),
        "FABGUARD_PRODUCER_TOKEN": settings.producer_token,
        "FABGUARD_ANALYST_TOKEN": settings.analyst_token,
        "FABGUARD_REVIEWER_TOKEN": settings.reviewer_token,
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=20).run()
    assert not app.exception
    click(app, "Run recorded-data replay")
    assert len(app.metric) == 3
    click(app, "Submit replay for investigation")
    incident_id = app.session_state["incident"]["id"]
    with postgres_saver_factory(settings.checkpoint_url) as saver:
        graph = build_langgraph(
            InvestigationGraph(
                postgres_retriever(settings.database_url), LocalTemplateLLM(), adaptive=False
            ),
            checkpointer=saver,
        )
        worker = InvestigationWorker(db, RuntimeGraphRunner(db, graph))
        assert worker.run_once().outcome == "completed"
        app.sidebar.radio[0].set_value("Incident").run()
        click(app, "Load / refresh incident")
        assert any("Report revision 1" in item.value for item in app.markdown)
        click(app, "Approve concrete ticket draft")
        click(app, "Load / refresh incident")
        assert len(db.get_incident(incident_id)["tickets"]) == 1
        # A page rerun must never launch another investigation.
        app.run()
        assert len(db.get_incident(incident_id)["investigations"]) == 1
        click(app, "Request investigation revision")
        assert len(db.get_incident(incident_id)["investigations"]) == 2
        assert worker.run_once().outcome == "completed"
        click(app, "Load / refresh incident")
        assert any("Report revision 2" in item.value for item in app.markdown)
        buttons = [
            button for button in app.button if button.label == "Approve concrete ticket draft"
        ]
        assert buttons[0].disabled and not buttons[1].disabled
    app.sidebar.radio[0].set_value("Results").run()
    assert not app.exception
    assert len(app.dataframe) >= 1


def test_streamlit_missing_artifacts_shows_setup_message(tmp_path, monkeypatch):
    monkeypatch.setenv("FABGUARD_MANIFEST", str(tmp_path / "missing.json"))
    monkeypatch.setenv("FABGUARD_MODEL", str(tmp_path / "missing.joblib"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    assert not app.exception
    assert "manifest or model is missing" in app.warning[0].value
