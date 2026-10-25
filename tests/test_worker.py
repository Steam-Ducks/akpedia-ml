"""Integration tests of the indexing worker: a Redis stream and a simulated akpedia-server.

Redis is ``fakeredis`` by default, so the suite needs no infrastructure. Setting
``REDIS_TEST_URL`` (e.g. ``redis://localhost:6379/15``) runs the same tests against a
real Redis; each test then uses its own stream and deletes it afterwards.

The server is a real HTTP server on a local port, not a mocked transport: the worker
goes through httpx, headers and JSON exactly as it does in production.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import fakeredis
import pytest
import redis

from app.embeddings import Embedder
from app.worker import (
    BLOCK_MS,
    MESSAGE_ID_FIELD,
    TOKEN_HEADER,
    IndexingWorker,
    JsonFormatter,
    ServerClient,
    connect_redis,
)
from tests.pdf_builder import build_pdf
from tests.test_process_document import FakeEmbedder

TOKEN = "segredo-de-teste"
GROUP = "ml-workers"


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    def json(self):
        return json.loads(self.body)


@dataclass
class SimulatedServer:
    """akpedia-server's ``/internal`` routes, answering whatever each test sets up."""

    files: dict[str, tuple[int, bytes, dict[str, str]]] = field(default_factory=dict)
    statuses: dict[str, int] = field(default_factory=dict)
    requests: list[Recorded] = field(default_factory=list)
    url: str = ""

    def serve_pdf(self, document_id: str, content: bytes, filename: str | None = None) -> None:
        headers = {"Content-Type": "application/pdf"}
        if filename:
            # What Spring 6.2's ContentDisposition.filename(name, UTF_8) writes, even for ASCII names.
            headers["Content-Disposition"] = f"inline; filename=\"=?UTF-8?Q?{filename}?=\"; filename*=UTF-8''{filename}"
        self.files[document_id] = (200, content, headers)

    def calls(self, suffix: str) -> list[Recorded]:
        return [request for request in self.requests if request.path.endswith(suffix)]


@pytest.fixture
def server() -> Iterator[SimulatedServer]:
    state = SimulatedServer()

    class Handler(BaseHTTPRequestHandler):
        def _record(self) -> Recorded:
            length = int(self.headers.get("Content-Length") or 0)
            request = Recorded(self.command, self.path, dict(self.headers), self.rfile.read(length))
            state.requests.append(request)
            return request

        def _answer(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
            request = self._record()
            if request.headers.get(TOKEN_HEADER) != TOKEN:
                return self._answer(401)
            document_id = request.path.split("/")[3]
            status, body, headers = state.files.get(document_id, (404, b"", {}))
            self._answer(status, body, headers)

        def do_POST(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
            request = self._record()
            if request.headers.get(TOKEN_HEADER) != TOKEN:
                return self._answer(401)
            route = request.path.rsplit("/", 1)[-1]
            self._answer(state.statuses.get(route, 204))

        def log_message(self, *args) -> None:
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{http.server_address[1]}"
    thread = threading.Thread(target=http.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    yield state
    http.shutdown()
    http.server_close()


@pytest.fixture
def redis_client() -> Iterator[redis.Redis]:
    url = os.getenv("REDIS_TEST_URL")
    client = redis.Redis.from_url(url) if url else fakeredis.FakeRedis()
    yield client
    client.close()


@pytest.fixture
def stream(redis_client: redis.Redis) -> Iterator[str]:
    name = f"documents:processing:test:{uuid.uuid4().hex}"
    yield name
    redis_client.delete(name)


@pytest.fixture
def worker(redis_client: redis.Redis, server: SimulatedServer, stream: str) -> Iterator[IndexingWorker]:
    client = ServerClient(server.url, TOKEN, timeout=5)
    yield IndexingWorker(
        redis_client=redis_client,
        server=client,
        embedder=FakeEmbedder(),
        stream=stream,
        group=GROUP,
        consumer="worker-1",
        block_ms=100,
    )
    client.close()


def approve(redis_client: redis.Redis, stream: str, document_id: str = "42") -> str:
    """What akpedia-server does on approval: one message with the document id."""
    message_id = redis_client.xadd(stream, {"documentId": document_id})
    return message_id.decode() if isinstance(message_id, bytes) else message_id


def pending(redis_client: redis.Redis, stream: str) -> int:
    return redis_client.xpending(stream, GROUP)["pending"]


# --- success ------------------------------------------------------------------


def test_approved_document_is_indexed_and_sent_in_the_process_format(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    server.serve_pdf("42", build_pdf(["Primeira frase do manual. Segunda frase do manual."]), "manual.pdf")
    # Approved before the worker created its group: starting the group at 0 still delivers it.
    message_id = approve(redis_client, stream)
    worker.ensure_group()

    assert worker.poll() == 1

    download = server.calls("/internal/documents/42/file")
    assert len(download) == 1 and download[0].headers[TOKEN_HEADER] == TOKEN

    [sent] = server.calls("/internal/documents/42/embeddings")
    assert sent.headers[TOKEN_HEADER] == TOKEN
    body = sent.json()
    assert body[MESSAGE_ID_FIELD] == message_id
    assert body["filename"] == "manual.pdf"
    assert body["model"] == {"name": "fake-embedder", "dimensions": 3, "normalized": True}
    assert body["chunk_count"] == len(body["chunks"]) >= 1
    assert body["chunks"][0]["text"].startswith("Primeira frase")
    assert len(body["chunks"][0]["embedding"]) == 3

    assert server.calls("processing-failure") == []


def test_worker_does_not_acknowledge_the_message(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    server.serve_pdf("42", build_pdf(["Documento aprovado."]))
    worker.ensure_group()
    approve(redis_client, stream)

    worker.poll()

    # XACK is the server's job, after it stores the result: the entry stays pending here.
    assert pending(redis_client, stream) == 1


def test_messages_are_taken_one_at_a_time(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    for document_id in ("1", "2"):
        server.serve_pdf(document_id, build_pdf([f"Documento {document_id}."]))
        approve(redis_client, stream, document_id)
    worker.ensure_group()

    assert worker.poll() == 1
    assert worker.poll() == 1
    assert worker.poll() == 0
    assert [r.path for r in server.calls("/embeddings")] == [
        "/internal/documents/1/embeddings",
        "/internal/documents/2/embeddings",
    ]


def test_filename_falls_back_when_the_server_sends_none(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    server.serve_pdf("42", build_pdf(["Documento sem nome."]), filename=None)
    worker.ensure_group()
    approve(redis_client, stream)

    worker.poll()

    assert server.calls("/embeddings")[0].json()["filename"] == "document-42.pdf"


@pytest.mark.parametrize(
    ("disposition", "expected"),
    [
        # Copied from Spring 6.2's output for "manual técnico.pdf".
        (
            "inline; filename=\"=?UTF-8?Q?manual_t=C3=A9cnico.pdf?=\"; filename*=UTF-8''manual%20t%C3%A9cnico.pdf",
            "manual técnico.pdf",
        ),
        ('inline; filename="=?UTF-8?Q?manual_t=C3=A9cnico.pdf?="', "manual técnico.pdf"),
        ('attachment; filename="relatorio.pdf"', "relatorio.pdf"),
        ("inline; filename=simples.pdf", "simples.pdf"),
    ],
    ids=["spring", "encoded-word-only", "quoted", "token"],
)
def test_filename_is_decoded_from_the_content_disposition(
    worker: IndexingWorker,
    server: SimulatedServer,
    redis_client: redis.Redis,
    stream: str,
    disposition: str,
    expected: str,
):
    headers = {"Content-Type": "application/pdf", "Content-Disposition": disposition}
    server.files["42"] = (200, build_pdf(["Documento com nome."]), headers)
    worker.ensure_group()
    approve(redis_client, stream)

    worker.poll()

    assert server.calls("/embeddings")[0].json()["filename"] == expected


def test_each_step_is_logged_with_the_document_id_and_duration(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str, caplog
):
    server.serve_pdf("42", build_pdf(["Documento aprovado."]))
    worker.ensure_group()
    approve(redis_client, stream)

    with caplog.at_level(logging.INFO, logger="app.worker"):
        worker.poll()

    steps = [r.fields for r in caplog.records if r.getMessage() == "step_finished"]
    assert [s["step"] for s in steps] == ["download", "extract", "chunk", "embed", "send"]
    assert all(s["documentId"] == "42" and s["durationMs"] >= 0 for s in steps)
    [done] = [r.fields for r in caplog.records if r.getMessage() == "document_indexed"]
    assert done["documentId"] == "42" and "durationMs" in done


# --- failure ------------------------------------------------------------------


def test_pdf_without_text_is_reported_as_a_failure(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    server.serve_pdf("42", build_pdf([""]), "digitalizado.pdf")
    worker.ensure_group()
    message_id = approve(redis_client, stream)

    assert worker.poll() == 1

    assert server.calls("/embeddings") == []
    [failure] = server.calls("/internal/documents/42/processing-failure")
    assert failure.headers[TOKEN_HEADER] == TOKEN
    assert failure.json()[MESSAGE_ID_FIELD] == message_id
    assert failure.json()["code"] == "no_extractable_text"
    assert "OCR" in failure.json()["message"]
    # A failure is acknowledged by the server too, once it records it.
    assert pending(redis_client, stream) == 1


def test_download_error_is_reported_as_a_failure(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    server.files["42"] = (500, b"boom", {})
    worker.ensure_group()
    approve(redis_client, stream)

    worker.poll()

    [failure] = server.calls("processing-failure")
    assert failure.json()["code"] == "server_call_failed"
    assert "500" in failure.json()["message"]


def test_corrupt_file_is_reported_as_unreadable(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    server.serve_pdf("42", b"isto nao e um pdf")
    worker.ensure_group()
    approve(redis_client, stream)

    worker.poll()

    assert server.calls("processing-failure")[0].json()["code"] == "unreadable_document"


def test_rejected_result_is_reported_as_a_failure(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    server.serve_pdf("42", build_pdf(["Documento aprovado."]))
    server.statuses["embeddings"] = 500
    worker.ensure_group()
    approve(redis_client, stream)

    worker.poll()

    assert len(server.calls("/embeddings")) == 1
    assert server.calls("processing-failure")[0].json()["code"] == "server_call_failed"


def test_worker_survives_a_failure_report_that_also_fails(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str, caplog
):
    server.serve_pdf("42", build_pdf([""]))
    server.statuses["processing-failure"] = 503
    worker.ensure_group()
    approve(redis_client, stream)

    with caplog.at_level(logging.INFO, logger="app.worker"):
        assert worker.poll() == 1

    assert "failure_report_failed" in [r.getMessage() for r in caplog.records]


def test_file_past_the_size_limit_is_refused(redis_client: redis.Redis, server: SimulatedServer, stream: str):
    server.serve_pdf("42", build_pdf(["Documento grande demais para o limite."]))
    client = ServerClient(server.url, TOKEN, max_file_bytes=64)
    small = IndexingWorker(
        redis_client=redis_client,
        server=client,
        embedder=FakeEmbedder(),
        stream=stream,
        group=GROUP,
        consumer="worker-1",
        block_ms=100,
    )
    small.ensure_group()
    approve(redis_client, stream)

    small.poll()
    client.close()

    assert server.calls("processing-failure")[0].json()["code"] == "file_too_large"


def test_message_without_a_valid_document_id_calls_nothing(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    worker.ensure_group()
    redis_client.xadd(stream, {"documentId": "../admin"})
    redis_client.xadd(stream, {"outroCampo": "1"})

    assert worker.poll() == 1
    assert worker.poll() == 1
    assert server.requests == []


# --- loop and setup -----------------------------------------------------------


def test_redis_replies_may_take_longer_than_one_block():
    client = connect_redis("redis://localhost:6379/0")
    assert client.connection_pool.connection_kwargs["socket_timeout"] > BLOCK_MS / 1000
    client.close()


@pytest.mark.skipif(not os.getenv("REDIS_TEST_URL"), reason="fakeredis never times out on a blocking read")
def test_waiting_on_an_empty_stream_is_not_taken_for_an_outage(server: SimulatedServer, stream: str):
    redis_client = connect_redis(os.environ["REDIS_TEST_URL"])
    client = ServerClient(server.url, TOKEN, timeout=5)
    worker = IndexingWorker(
        redis_client=redis_client,
        server=client,
        embedder=FakeEmbedder(),
        stream=stream,
        group=GROUP,
        consumer="worker-1",
        block_ms=BLOCK_MS,
    )
    worker.ensure_group()

    assert worker.poll() == 0
    assert worker.poll() == 0

    redis_client.close()
    client.close()


def test_ensure_group_is_idempotent(worker: IndexingWorker):
    worker.ensure_group()
    worker.ensure_group()


@pytest.mark.skipif(
    not os.getenv("REDIS_TEST_URL"),
    reason="fakeredis answers XREADGROUP on a deleted stream with another error than Redis' NOGROUP",
)
def test_group_is_recreated_when_the_stream_disappears(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    worker.ensure_group()
    redis_client.delete(stream)

    assert worker.poll() == 0

    server.serve_pdf("42", build_pdf(["Documento aprovado depois."]))
    approve(redis_client, stream)
    assert worker.poll() == 1


@pytest.mark.parametrize(
    "error",
    [redis.ConnectionError("connection refused"), redis.TimeoutError("timed out"), redis.BusyLoadingError("loading")],
    ids=["connection", "timeout", "loading"],
)
def test_run_keeps_going_while_redis_is_down_and_stops_on_request(server: SimulatedServer, error: Exception):
    stop = threading.Event()
    attempts = []

    class DownRedis:
        def xgroup_create(self, *args, **kwargs):
            attempts.append(1)
            if len(attempts) >= 2:
                stop.set()
            raise error

    client = ServerClient(server.url, TOKEN)
    IndexingWorker(redis_client=DownRedis(), server=client, embedder=FakeEmbedder()).run(stop)
    client.close()

    assert len(attempts) == 2


def test_failed_step_is_logged_with_its_duration(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str, caplog
):
    server.serve_pdf("42", b"isto nao e um pdf")
    worker.ensure_group()
    approve(redis_client, stream)

    with caplog.at_level(logging.INFO, logger="app.worker"):
        worker.poll()

    [failed] = [r.fields for r in caplog.records if r.getMessage() == "step_failed"]
    assert failed["step"] == "extract" and failed["documentId"] == "42" and failed["durationMs"] >= 0
    [done] = [r.fields for r in caplog.records if r.getMessage() == "document_failed"]
    assert done["step"] == "extract" and done["code"] == "unreadable_document"


def test_message_with_bytes_that_are_not_utf8_does_not_stop_the_worker(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    worker.ensure_group()
    redis_client.xadd(stream, {b"documentId": b"\xff\xfe"})

    assert worker.poll() == 1
    assert server.requests == []


def test_log_lines_are_json_with_the_event_fields():
    record = logging.LogRecord("app.worker", logging.INFO, __file__, 1, "step_finished", None, None)
    record.fields = {"documentId": "42", "step": "embed", "durationMs": 12.5}

    line = json.loads(JsonFormatter().format(record))

    assert line["event"] == "step_finished"
    assert line["level"] == "INFO"
    assert line["documentId"] == "42" and line["step"] == "embed" and line["durationMs"] == 12.5
    assert line["ts"].endswith("+00:00")


# --- pending entries after a restart ------------------------------------------


def deliver_without_processing(redis_client: redis.Redis, stream: str, consumer: str = "worker-1") -> None:
    """What a worker that died mid-document leaves behind: a delivered, unacknowledged entry."""
    redis_client.xreadgroup(GROUP, consumer, {stream: ">"}, count=1)


def test_restarted_worker_resumes_its_own_pending_message_first(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    worker.ensure_group()
    server.serve_pdf("1", build_pdf(["Documento interrompido no meio."]))
    server.serve_pdf("2", build_pdf(["Documento novo."]))
    approve(redis_client, stream, "1")
    deliver_without_processing(redis_client, stream)
    approve(redis_client, stream, "2")

    assert worker.poll() == 1
    assert worker.poll() == 1

    assert [r.path for r in server.calls("/embeddings")] == [
        "/internal/documents/1/embeddings",
        "/internal/documents/2/embeddings",
    ]


def test_pending_list_is_read_once_and_does_not_loop(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    # The worker never acknowledges, so the entries stay pending after being resumed:
    # reading the pending list from the start again would return them forever.
    worker.ensure_group()
    for document_id in ("1", "2"):
        server.serve_pdf(document_id, build_pdf([f"Documento {document_id}."]))
        approve(redis_client, stream, document_id)
        deliver_without_processing(redis_client, stream)

    handled = [worker.poll() for _ in range(4)]

    assert handled == [1, 1, 0, 0]
    assert len(server.calls("/embeddings")) == 2


def test_pending_messages_of_other_consumers_are_left_alone(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    worker.ensure_group()
    approve(redis_client, stream, "1")
    deliver_without_processing(redis_client, stream, consumer="worker-2")

    assert worker.poll() == 0
    assert server.requests == []


def test_pending_entry_deleted_from_the_stream_is_skipped(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str, caplog
):
    # Redis answers a deleted entry with no fields (nil; fakeredis gives an empty dict).
    worker.ensure_group()
    message_id = approve(redis_client, stream, "1")
    deliver_without_processing(redis_client, stream)
    redis_client.xdel(stream, message_id)

    with caplog.at_level(logging.INFO, logger="app.worker"):
        worker.poll()

    assert server.requests == []
    assert "message_deleted" in [r.getMessage() for r in caplog.records]


def test_malformed_encoded_filename_falls_back_instead_of_failing(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    headers = {"Content-Type": "application/pdf", "Content-Disposition": 'inline; filename="=?x-unknown?Q?a?="'}
    server.files["42"] = (200, build_pdf(["Documento com nome estranho."]), headers)
    worker.ensure_group()
    approve(redis_client, stream)

    worker.poll()

    assert server.calls("processing-failure") == []
    assert server.calls("/embeddings")[0].json()["filename"] == "document-42.pdf"


# --- documents that keep killing the worker ------------------------------------


class _WorkerKilled(BaseException):
    """Stands in for the process dying mid-document: nothing in the worker catches it."""


class KillingEmbedder(FakeEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise _WorkerKilled()


def restarted_worker(redis_client: redis.Redis, server: SimulatedServer, stream: str, embedder: Embedder):
    """A fresh worker process with the same consumer name, as after a container restart."""
    return IndexingWorker(
        redis_client=redis_client,
        server=ServerClient(server.url, TOKEN, timeout=5),
        embedder=embedder,
        stream=stream,
        group=GROUP,
        consumer="worker-1",
        block_ms=100,
    )


def deliveries(redis_client: redis.Redis, stream: str, message_id: str) -> int:
    [entry] = redis_client.xpending_range(stream, GROUP, min=message_id, max=message_id, count=1)
    return entry["times_delivered"]


def test_document_that_keeps_killing_the_worker_is_given_up_after_three_deliveries(
    server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    server.serve_pdf("1", build_pdf(["Documento que derruba o worker."]))
    server.serve_pdf("2", build_pdf(["Documento que chegou depois."]))
    restarted_worker(redis_client, server, stream, FakeEmbedder()).ensure_group()
    poison = approve(redis_client, stream, "1")

    # Three runs die in the middle of the same document: the first takes it as a new
    # message, the next two resume it from the pending list.
    for _ in range(3):
        with pytest.raises(_WorkerKilled):
            restarted_worker(redis_client, server, stream, KillingEmbedder()).poll()
    assert deliveries(redis_client, stream, poison) == 3

    approve(redis_client, stream, "2")
    healthy = restarted_worker(redis_client, server, stream, FakeEmbedder())

    assert healthy.poll() == 1  # gives the poisoned document up instead of a fourth try
    assert healthy.poll() == 1  # and the queue moves on

    [failure] = server.calls("/internal/documents/1/processing-failure")
    assert failure.json()["code"] == "retry_limit_exceeded"
    assert "3 times" in failure.json()["message"]
    assert [r.path for r in server.calls("/embeddings")] == ["/internal/documents/2/embeddings"]


def test_resuming_a_pending_message_counts_as_a_delivery(
    worker: IndexingWorker, server: SimulatedServer, redis_client: redis.Redis, stream: str
):
    worker.ensure_group()
    server.serve_pdf("1", build_pdf(["Documento interrompido."]))
    message_id = approve(redis_client, stream, "1")
    deliver_without_processing(redis_client, stream)

    worker.poll()

    assert deliveries(redis_client, stream, message_id) == 2
    assert len(server.calls("/internal/documents/1/embeddings")) == 1
