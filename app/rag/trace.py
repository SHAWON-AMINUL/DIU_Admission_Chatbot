"""The per-chat cycle log: one plain-text file per chat, one block per stage.

A chat's whole cycle -- rewrite, classify, retrieve, generate -- belongs in
one file, in order, so it can be read as a cycle instead of reconstructed by
grepping a shared stream for a chat id. Plain text rather than JSON because
the file is meant to be opened and read: the question, the chunks it matched
and the answer it produced, in the order they happened, with timings.

Tracing is opt-in per call: no chat_id means no file. The chain is also run
from scripts and tests, and those must not leave files nobody can attribute.
"""

import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from app.config import settings

LOG_DIR = Path(settings.CHAT_LOG_DIR)

# chat_id arrives verbatim in an unauthenticated request body, so it is
# untrusted input, not a name: joined naively "../../etc/x" writes outside
# LOG_DIR. Everything outside the allowed set becomes "_", which keeps the
# file inside the directory no matter what was sent.
UNSAFE = re.compile(r"[^A-Za-z0-9-]")

_INDENT = "    "


def _block(stage: str, fields: dict) -> str:
    """One readable stage block.

    Every value line is indented, including the continuation lines of a
    multi-line answer -- otherwise a Bangla answer with a line break would
    break the block structure and read as if it were the next stage.
    """
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"[{stamp}] {stage}"]
    for key, value in fields.items():
        text = str(value)
        first, *rest = text.split("\n")
        lines.append(f"{_INDENT}{key}: {first}")
        lines.extend(f"{_INDENT}{line}" for line in rest)
    return "\n".join(lines) + "\n\n"


def log_path(chat_id: str) -> Path:
    """Where this chat's log lives.

    The single place a chat_id becomes a filename, used by the writer and by
    the endpoint that reads it back -- so the sanitising above cannot be
    applied on one side and forgotten on the other.
    """
    return LOG_DIR / f"{UNSAFE.sub('', chat_id)[:100]}.log"


def record(chat_id: str | None, stage: str, **fields) -> None:
    """Append one stage block to this chat's log. No chat_id, no log."""
    if not chat_id:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = log_path(chat_id)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_block(stage, fields))


@contextmanager
def stage(chat_id: str | None, name: str, **fields):
    """Time a stage and log it on the way out, raised exception included.

    A cycle that died halfway is the one worth reading, so the block is
    written whether the block succeeded or not.
    """
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:
        record(
            chat_id,
            name,
            ms=round((time.perf_counter() - started) * 1000, 1),
            error=f"{type(exc)._name_}: {exc}",
            **fields,
        )
        raise
    record(
        chat_id,
        name,
        ms=round((time.perf_counter() - started) * 1000, 1),
        **fields,
    )