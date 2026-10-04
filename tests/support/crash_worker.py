"""Subprocess fixture that exits at real PostgreSQL recovery boundaries."""

import json
import os
from dataclasses import asdict
from pathlib import Path

from langgraph.checkpoint.postgres import PostgresSaver

from fabguard.config import RuntimeSettings
from fabguard.graph import InvestigationGraph, build_langgraph
from fabguard.llm import LocalTemplateLLM
from fabguard.retrieval import postgres_retriever
from fabguard.storage import Database
from fabguard.worker import InvestigationWorker, RuntimeGraphRunner

settings = RuntimeSettings()
crash_point = os.environ["FABGUARD_CRASH_POINT"]
log = Path(os.environ["FABGUARD_TEST_CALL_LOG"])


def record(call):
    with log.open("a", encoding="utf-8") as output:
        output.write(json.dumps({"call": call}) + "\n")


class CrashSaver(PostgresSaver):
    def put(self, config, checkpoint, metadata, new_versions):
        saved = super().put(config, checkpoint, metadata, new_versions)
        trace = checkpoint["channel_values"].get("trace", [])
        if trace and trace[-1] == crash_point:
            os._exit(17)
        return saved


class CountingLLM(LocalTemplateLLM):
    def complete(self, **kwargs):
        record("llm")
        result = super().complete(**kwargs)
        if crash_point == "inflight":
            os._exit(17)
        return result


class CountingRetriever:
    def search(self, *args, **kwargs):
        record("retrieval")
        return postgres_retriever(settings.database_url).search(*args, **kwargs)


with CrashSaver.from_conn_string(settings.checkpoint_url) as saver:
    database = Database(settings.database_url, artifact_root=settings.artifact_root)
    graph = build_langgraph(
        InvestigationGraph(CountingRetriever(), CountingLLM(), adaptive=False),
        checkpointer=saver,
    )
    result = InvestigationWorker(database, RuntimeGraphRunner(database, graph)).run_once()
    print(json.dumps(asdict(result)))
