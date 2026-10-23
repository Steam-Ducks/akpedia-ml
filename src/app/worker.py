"""Background indexing of approved documents: ``python -m app.worker``.

akpedia-server adds one message per approved document to a Redis stream. This worker
reads them one at a time through a consumer group, downloads the document's PDF from
the server, runs the same pipeline as ``POST /api/v1/documents/process`` (extraction,
chunking, embedding) and posts the result back in the ``/process`` response format.
When any step fails, the server is told so instead.

The worker never acknowledges (``XACK``) a message: the server does, once it has
stored the result or the failure. Every call back carries the message id in a
``messageId`` field of its body so the server knows which entry to acknowledge.

Each document produces structured (JSON) log lines with its ``documentId`` and the
duration of every step, so a slow or failing document can be found in the logs.
"""

from __future__ import annotations

import json
import logging
import re
import signal
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from email.errors import HeaderParseError
from email.header import decode_header, make_header
from typing import Any
from urllib.parse import unquote

import httpx
import redis

from app import config
from app.api.documents import build_process_response, no_text_error
from app.documents import (
    DocumentReadError,
    FileTooLargeError,
    NoExtractableTextError,
    UnsupportedDocumentFormatError,
    extract_text,
    split_text,
)
from app.embeddings import Embedder, get_embedder

logger = logging.getLogger("app.worker")

#: Field of each stream message holding the document id.
DOCUMENT_ID_FIELD = "documentId"

#: Header carrying the shared secret on the server's ``/internal`` routes.
TOKEN_HEADER = "X-Internal-Token"

#: Field of every call-back body telling the server which stream entry to acknowledge.
MESSAGE_ID_FIELD = "messageId"

#: How long one ``XREADGROUP`` waits for a message, in milliseconds.
BLOCK_MS = 5000

#: How long a Redis reply may take, in seconds. It must outlast ``BLOCK_MS``: redis-py 8
#: defaults to 5 s, the same as the block, so every wait on an empty stream would time
#: out and be taken for an outage, pausing the worker up to ``_MAX_BACKOFF_SECONDS``.
SOCKET_TIMEOUT_SECONDS = BLOCK_MS / 1000 + 5

#: Longest pause between attempts while Redis is unreachable, in seconds.
_MAX_BACKOFF_SECONDS = 30.0

#: Deliveries after which a pending message is given up instead of retried: a document
#: that keeps killing the worker would otherwise be retried first on every restart and
#: block the queue. Same limit as the server's recovery job.
MAX_DELIVERIES = 3


class RetryLimitExceededError(Exception):
    """A pending message was delivered :data:`MAX_DELIVERIES` times without finishing."""


_DOCUMENT_ID = re.compile(r"^\d+$")
_FILENAME_EXTENDED = re.compile(r"filename\*\s*=\s*([\w-]+)'[^']*'([^;\s]+)", re.IGNORECASE)
_FILENAME = re.compile(r'filename\s*=\s*"?([^";]+)"?', re.IGNORECASE)

#: ``code`` sent to the server per failure, matching the API's error codes where one exists.
_FAILURE_CODES: dict[type[Exception], str] = {
    UnsupportedDocumentFormatError: "unsupported_format",
    DocumentReadError: "unreadable_document",
    NoExtractableTextError: "no_extractable_text",
    FileTooLargeError: "file_too_large",
    httpx.HTTPError: "server_call_failed",
    RetryLimitExceededError: "retry_limit_exceeded",
}


@dataclass(frozen=True)
class DownloadedFile:
    """A document's stored file, as the server sent it."""

    content: bytes
    filename: str
    media_type: str | None


class ServerClient:
    """Calls to akpedia-server's ``/internal/documents`` routes."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: float = 30.0,
        max_file_bytes: int = config.MAX_UPLOAD_BYTES,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url, headers={TOKEN_HEADER: token}, timeout=timeout, transport=transport
        )
        self._max_file_bytes = max_file_bytes

    def fetch_file(self, document_id: str) -> DownloadedFile:
        """Download the document's PDF, refusing anything past the upload limit."""
        with self._client.stream("GET", f"/internal/documents/{document_id}/file") as response:
            response.raise_for_status()
            content = bytearray()
            for part in response.iter_bytes():
                content += part
                if len(content) > self._max_file_bytes:
                    raise FileTooLargeError(
                        f"The file of document {document_id} is larger than {self._max_file_bytes} bytes."
                    )
            return DownloadedFile(
                content=bytes(content),
                filename=_filename_from(response.headers.get("content-disposition", ""))
                or f"document-{document_id}.pdf",
                media_type=response.headers.get("content-type"),
            )

    def send_embeddings(self, document_id: str, body: dict[str, Any], message_id: str) -> None:
        """Post the chunks and vectors: the ``/process`` response format plus ``messageId``."""
        self._client.post(
            f"/internal/documents/{document_id}/embeddings", json={**body, MESSAGE_ID_FIELD: message_id}
        ).raise_for_status()

    def report_failure(self, document_id: str, code: str, message: str, message_id: str) -> None:
        """Tell the server the document could not be indexed, and why."""
        self._client.post(
            f"/internal/documents/{document_id}/processing-failure",
            json={"code": code, "message": message, MESSAGE_ID_FIELD: message_id},
        ).raise_for_status()

    def close(self) -> None:
        self._client.close()


class IndexingWorker:
    """Consumes the approved-documents stream and indexes one document per message."""

    def __init__(
        self,
        *,
        redis_client: redis.Redis,
        server: ServerClient,
        embedder: Embedder,
        stream: str = config.DOCUMENTS_STREAM,
        group: str = config.WORKER_GROUP,
        consumer: str = config.WORKER_CONSUMER,
        block_ms: int = BLOCK_MS,
        max_deliveries: int = MAX_DELIVERIES,
    ) -> None:
        self._redis = redis_client
        self._server = server
        self._embedder = embedder
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._block_ms = block_ms
        self._max_deliveries = max_deliveries
        #: Last pending entry taken from this consumer's list (``"-"`` before the first);
        #: ``None`` once the list is drained and only new messages are read.
        self._pending_cursor: str | None = "-"

    def ensure_group(self) -> None:
        """Create the consumer group (and the stream) if they do not exist yet.

        The group starts at ``0`` rather than ``$`` so documents approved before the
        first worker ever ran are still indexed.
        """
        try:
            self._redis.xgroup_create(self._stream, self._group, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def poll(self) -> int:
        """Handle one message: first this consumer's pending ones, then new ones.

        Returns how many were handled. New messages are waited for up to ``block_ms``.
        """
        try:
            if self._pending_cursor is not None:
                resumed = self._resume_pending()
                if resumed is not None:
                    return resumed
            entries = self._redis.xreadgroup(
                self._group, self._consumer, {self._stream: ">"}, count=1, block=self._block_ms
            )
        except redis.ResponseError as exc:
            if "NOGROUP" not in str(exc):
                raise
            # The stream was deleted (and the group with it) while the worker was running.
            self.ensure_group()
            self._pending_cursor = None
            return 0

        messages = [message for _stream, batch in entries or [] for message in batch]
        for raw_id, fields in messages:
            self._handle_entry(_text(raw_id), fields)
        return len(messages)

    def _resume_pending(self) -> int | None:
        """Take the next message this consumer received but the server never acknowledged.

        That happens when the worker stopped in the middle of a document, or could not
        even report its failure. Pending messages are taken once, at startup, before any
        new message. Returns ``None`` once there are none left.

        The list is walked with a cursor: the worker never acknowledges, so whatever it
        resumes stays pending, and starting from the top again would return the same
        entry forever. Each one is taken with ``XCLAIM``, which counts the delivery, and
        a message already delivered :data:`MAX_DELIVERIES` times is reported as failed
        instead of tried again: a document that kills the worker (out of memory, say)
        would otherwise be resumed first on every restart and block the whole queue.
        Entries left by other consumers are not taken here; the server's recovery job
        claims those.
        """
        pending = self._redis.xpending_range(
            self._stream, self._group, min=self._pending_cursor, max="+", count=2, consumername=self._consumer
        )
        entry = next((item for item in pending if _text(item["message_id"]) != self._pending_cursor), None)
        if entry is None:
            _log(logging.INFO, "pending_drained", consumer=self._consumer)
            self._pending_cursor = None
            return None

        message_id = _text(entry["message_id"])
        self._pending_cursor = message_id
        deliveries = int(entry["times_delivered"])
        if deliveries >= self._max_deliveries:
            self._give_up(message_id, deliveries)
            return 1

        claimed = self._redis.xclaim(
            self._stream, self._group, self._consumer, min_idle_time=0, message_ids=[message_id]
        )
        _log(logging.INFO, "pending_resumed", messageId=message_id, delivery=deliveries + 1)
        self._handle_entry(message_id, claimed[0][1] if claimed else None)
        return 1

    def _give_up(self, message_id: str, deliveries: int) -> None:
        """Report a message that keeps failing as failed, without processing it again."""
        rows = self._redis.xrange(self._stream, min=message_id, max=message_id, count=1)
        fields = {_text(key): _text(value) for key, value in (rows[0][1] or {}).items()} if rows else {}
        document_id = fields.get(DOCUMENT_ID_FIELD, "")
        if not _DOCUMENT_ID.match(document_id):
            _log(logging.ERROR, "message_invalid", messageId=message_id, fields=fields)
            return
        error = RetryLimitExceededError(f"Retry limit exceeded: delivered {deliveries} times without finishing.")
        context = {"documentId": document_id, "messageId": message_id}
        self._report_failure(context, "resume", error, time.perf_counter())

    def _handle_entry(self, message_id: str, fields: dict | None) -> None:
        if not fields:
            # The entry was deleted from the stream after it was delivered: there is
            # nothing left to read, and no document id to report on.
            _log(logging.WARNING, "message_deleted", messageId=message_id)
            return
        self.handle(message_id, {_text(key): _text(value) for key, value in fields.items()})

    def handle(self, message_id: str, fields: dict[str, str]) -> bool:
        """Index the document named in one message. Returns whether it was indexed.

        Never raises: a failure is reported to the server and logged, so one bad
        document cannot stop the worker.
        """
        document_id = fields.get(DOCUMENT_ID_FIELD, "")
        if not _DOCUMENT_ID.match(document_id):
            # Without a valid id there is no document to report the failure on.
            _log(logging.ERROR, "message_invalid", messageId=message_id, fields=fields)
            return False

        context = {"documentId": document_id, "messageId": message_id}
        _log(logging.INFO, "document_received", **context)
        started = time.perf_counter()
        step = "download"
        try:
            with _timed(context, "download"):
                file = self._server.fetch_file(document_id)
            step = "extract"
            with _timed(context, "extract"):
                text = extract_text(file.content, filename=file.filename, media_type=file.media_type)
            step = "chunk"
            with _timed(context, "chunk"):
                chunks = split_text(text)
            if not chunks:
                raise no_text_error(file.filename)
            step = "embed"
            with _timed(context, "embed", chunkCount=len(chunks)):
                vectors = self._embedder.embed(chunks)
            step = "send"
            with _timed(context, "send"):
                body = build_process_response(file.filename, self._embedder, chunks, vectors)
                self._server.send_embeddings(document_id, body.model_dump(mode="json"), message_id)
        except Exception as exc:  # the worker's boundary: every failure is reported
            self._report_failure(context, step, exc, started)
            return False

        _log(logging.INFO, "document_indexed", **context, chunkCount=len(chunks), durationMs=_elapsed_ms(started))
        return True

    def run(self, stop: threading.Event) -> None:
        """Poll until ``stop`` is set, riding out Redis outages with a growing pause.

        Any Redis error is retried, not only a lost connection: a timeout, or a Redis
        still loading its data after a restart, would otherwise end the process.
        """
        backoff = 1.0
        group_ready = False
        while not stop.is_set():
            try:
                if not group_ready:
                    self.ensure_group()
                    group_ready = True
                self.poll()
                backoff = 1.0
            except redis.RedisError as exc:
                _log(logging.WARNING, "redis_unavailable", error=str(exc), retryInSeconds=backoff)
                stop.wait(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)

    def _report_failure(self, context: dict[str, str], step: str, exc: Exception, started: float) -> None:
        code = next((code for kind, code in _FAILURE_CODES.items() if isinstance(exc, kind)), "processing_error")
        message = str(exc) or type(exc).__name__
        _log(
            logging.ERROR,
            "document_failed",
            **context,
            step=step,
            code=code,
            error=message,
            durationMs=_elapsed_ms(started),
        )
        try:
            self._server.report_failure(context["documentId"], code, message, context["messageId"])
        except Exception as report_exc:  # nothing left to tell; the log is the record
            # The message stays pending: the server never heard of it, so it never acknowledges it.
            _log(logging.ERROR, "failure_report_failed", **context, error=str(report_exc))


class JsonFormatter(logging.Formatter):
    """One JSON object per line: timestamp, level, event and the event's fields."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    # One line per HTTP call would repeat what the step logs already say.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def connect_redis(url: str) -> redis.Redis:
    """A Redis client whose replies may take longer than one ``XREADGROUP`` block."""
    return redis.Redis.from_url(url, socket_timeout=SOCKET_TIMEOUT_SECONDS)


def main() -> int:
    configure_logging()
    if not config.INTERNAL_API_TOKEN:
        _log(logging.ERROR, "worker_misconfigured", error="INTERNAL_API_TOKEN is not set.")
        return 1

    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())

    _log(
        logging.INFO,
        "worker_starting",
        stream=config.DOCUMENTS_STREAM,
        group=config.WORKER_GROUP,
        consumer=config.WORKER_CONSUMER,
        server=config.SERVER_BASE_URL,
    )
    started = time.perf_counter()
    # Loaded before the first message, so the first document is not charged the model load.
    embedder = get_embedder()
    _log(logging.INFO, "model_loaded", model=embedder.model_name, durationMs=_elapsed_ms(started))

    server = ServerClient(config.SERVER_BASE_URL, config.INTERNAL_API_TOKEN, timeout=config.SERVER_TIMEOUT_SECONDS)
    worker = IndexingWorker(redis_client=connect_redis(config.REDIS_URL), server=server, embedder=embedder)
    try:
        worker.run(stop)
    finally:
        server.close()
    _log(logging.INFO, "worker_stopped")
    return 0


@contextmanager
def _timed(context: dict[str, str], step: str, **fields: Any) -> Iterator[None]:
    """Log how long a step took, whether it finished or failed."""
    started = time.perf_counter()
    try:
        yield
    except Exception:
        _log(logging.WARNING, "step_failed", **context, step=step, durationMs=_elapsed_ms(started), **fields)
        raise
    _log(logging.INFO, "step_finished", **context, step=step, durationMs=_elapsed_ms(started), **fields)


def _log(level: int, event: str, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": fields})


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _filename_from(disposition: str) -> str | None:
    """Read the file name out of a ``Content-Disposition`` header.

    Spring writes ``filename="=?UTF-8?Q?...?="; filename*=UTF-8''...`` for every name,
    accented or not, so the RFC 5987 ``filename*`` is preferred and an RFC 2047 encoded
    word in the plain ``filename`` is decoded rather than passed through as is.
    """
    extended = _FILENAME_EXTENDED.search(disposition)
    if extended:
        charset, encoded = extended.groups()
        try:
            return unquote(encoded, encoding=charset, errors="strict")
        except (LookupError, UnicodeDecodeError):
            pass
    plain = _FILENAME.search(disposition)
    if not plain:
        return None
    name = plain.group(1).strip()
    if name.startswith("=?"):
        try:
            name = str(make_header(decode_header(name)))
        except (LookupError, UnicodeDecodeError, HeaderParseError, ValueError):
            # A malformed encoded word or an unknown charset says nothing usable about
            # the name; the caller falls back to document-{id}.pdf. The file itself
            # downloaded fine, so this must not fail the document.
            return None
    return name or None


def _text(value: bytes | str) -> str:
    # A message holding bytes that are not UTF-8 must not stop the worker: it is logged
    # as invalid like any other malformed message.
    return value.decode(errors="replace") if isinstance(value, bytes) else value


if __name__ == "__main__":
    raise SystemExit(main())
