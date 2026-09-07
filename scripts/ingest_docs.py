"""
Command-line entry point for building the document index.

    python -m scripts.ingest_docs [--directory PATH]

THIS IS THE ONE-SHOT INDEXING STEP YOU RUN BEFORE STARTING THE SERVER. The
running app only ever READS from the `documents` collection — nothing in the
WebSocket path ingests anything — so an un-ingested corpus produces an agent
that connects, listens, answers, and never once cites a document. Retrieval
returns zero sources and the model, correctly, says it does not know.

Kept as a separate process rather than a startup hook in `app/main.py` because
the two jobs have opposite shapes. Ingestion is slow, bursty and expensive: it
reads every file, splits it, and calls the embedding API once per batch. Doing
that at boot would delay every server start by the size of the corpus, repeat
the cost on every restart and every dev-server reload, and make a transient
embedding-API failure into a server that will not start. Indexing is something
you do when the documents change; serving is something you do all the time.

A thin wrapper on purpose. Everything real lives in `app/rag/ingest.py`, which
is importable and testable without a terminal; this file owns only the three
things a CLI owns — parsing arguments, starting an event loop, and turning a
result into readable output and an exit code.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app import config
from app.logging_config import setup_logging
from app.rag import ingest, store

logger = logging.getLogger(__name__)


EXIT_OK = 0

EXIT_FAILURE = 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    One optional argument. The default is `config.DOCUMENTS_DIR`, so the
    common case is a bare `python -m scripts.ingest_docs`, and `--directory`
    exists for the case that actually comes up in practice: indexing a folder
    of documents that lives somewhere else, without first copying it into the
    project.
    """
    parser = argparse.ArgumentParser(
        prog="python -m scripts.ingest_docs",
        description=(
            "Chunk, embed and index every supported document in a directory into "
            "MongoDB Atlas, so the voice agent can retrieve from them. Run this "
            "before starting the server, and again whenever the documents change."
        ),
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=config.DOCUMENTS_DIR,
        help=(
            "Directory to ingest. Not recursive. "
            f"Defaults to {config.DOCUMENTS_DIR} relative to the project root."
        ),
    )
    return parser.parse_args(argv)


def _print_summary(directory: Path, summary: dict) -> None:
    """Report what the run did, in the terms the operator asked the question in.

    Printed rather than logged, and printed even though the same numbers appear
    in the log. The log is a record of how the run went; this is the answer to
    "did it work", and it has to be readable at a glance underneath however
    many INFO lines the embedding batches produced.

    The skipped files are listed by name, not just counted. A count tells you
    something was left out; the names tell you whether it was the `.DS_Store`
    you expected or the `.docx` you did not realise was unsupported.
    """
    print()
    print(f"Ingestion complete for {directory}")
    print(f"  files processed : {summary.get('files_processed', 0)}")
    print(f"  chunks written  : {summary.get('chunks_written', 0)}")
    print(f"  files skipped   : {summary.get('files_skipped', 0)}")

    skipped = summary.get("skipped") or []
    for name in skipped:
        print(f"      skipped: {name}")

    if not summary.get("chunks_written"):
        print()
        print(
            "  No chunks were written, so the agent has nothing to retrieve. Check that the "
            "directory contains supported files."
        )


async def _run(directory: Path) -> dict:
    """Ingest the directory, then close the Mongo client.

    The `finally` is the reason this is a function rather than a bare
    `asyncio.run(ingest.ingest_directory(...))`. `store` holds a module-level
    `AsyncMongoClient` whose connection pool and topology-monitor task are
    bound to the loop `asyncio.run` created; letting that loop close underneath
    them is what produces "Task was destroyed but it is pending" and a
    non-zero-feeling exit from a run that actually succeeded. Closing inside
    the loop, on both the success and the failure path, avoids it.
    """
    try:
        return await ingest.ingest_directory(directory)
    finally:
        await store.close_client()


def main(argv: list[str] | None = None) -> int:
    """Run one ingestion and return the process exit code.

    Every failure is caught and turned into a message plus `EXIT_FAILURE`
    rather than a traceback. This is a command a person runs from a shell and,
    more importantly, one a Makefile or a deploy script runs unattended: a
    non-zero exit is the only part of the output the caller reliably reads, so
    it must be correct even when the cause is a missing `.env` rather than a
    bug. The traceback is not lost — it goes to the log at ERROR — but it is
    not what the operator is shown first.

    `ingest_directory` propagates rather than collecting failures into the
    summary, deliberately (see its docstring): a corpus you were told was built
    but silently is not is worse than a run that stopped and said why. This
    function preserves that by exiting non-zero on anything it raises.
    """
    args = _parse_args(argv)
    setup_logging()

    directory = Path(args.directory).expanduser()

    logger.info("Ingesting documents from %s", directory)

    try:
        summary = asyncio.run(_run(directory))
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        print("Interrupted. Files already written are indexed; re-run to finish.", file=sys.stderr)
        return EXIT_FAILURE
    except Exception as error:
        logger.error("Ingestion failed: %s", error, exc_info=True)
        print(f"Error: ingestion failed ({type(error).__name__}: {error})", file=sys.stderr)
        return EXIT_FAILURE

    _print_summary(directory, summary)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
