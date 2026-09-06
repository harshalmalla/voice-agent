"""
Document ingestion: files on disk -> chunks -> vectors -> MongoDB Atlas.

This is the write half of the `documents` side of the RAG loop, and the only
module in the package that touches the filesystem. It reads whatever the user
has dropped into `data/documents/` (`config.DOCUMENTS_DIR`), splits each file
into overlapping passages, embeds them in batches via
`app/rag/embeddings.py`, and writes them to `config.DOCUMENTS_COLLECTION`
through `app/rag/store.py`. It is driven from a CLI
(`python -m scripts.ingest_docs`), not from the live voice path — nothing here
runs while a user is talking.

--- Why chunk at all -------------------------------------------------------
Embedding models compress a whole passage into one fixed-width vector, so the
longer the passage the more diluted that vector becomes: a 40-page handbook
embedded whole produces a vector that is "about employment, vaguely" and is
near-equidistant from every question you could ask it. Retrieval quality comes
from the chunks being small enough that each one is *about* one thing. The
other half of the argument is the prompt: the retrieved text is pasted into a
Gemini call, and shipping whole documents burns context and latency on
material the question never touches.

--- Why the chunks OVERLAP -------------------------------------------------
A hard split at 800 characters lands wherever it lands, and it will sometimes
land in the middle of the one sentence that answers the question — the setup
in chunk N, the payoff in chunk N+1, and neither chunk on its own is a good
match for the query. Repeating the last `CHUNK_OVERLAP` characters of each
chunk at the start of the next means any span shorter than the overlap
survives intact in at least one chunk. The cost is that the corpus grows by
roughly `overlap / chunk_size` (~12% at the defaults), which is cheap
insurance.

--- Idempotency ------------------------------------------------------------
Re-running ingestion must not double the corpus. Duplicate chunks are worse
than wasted storage: they crowd out genuinely different material in the top-k,
so the model receives the same paragraph four times instead of four
perspectives. Every chunk records the filename it came from in `source`, and
each file's existing chunks are deleted (`store.delete_chunks_by_source`)
immediately before its new ones are inserted. Delete-then-insert is chosen
over upsert-by-key because re-chunking a file with a different `CHUNK_SIZE`
renumbers everything — a keyed upsert would leave the tail of the previous,
longer chunk list orphaned in the collection.

--- pypdf API, confirmed against the installed package (not from memory) ----
`pypdf` 6.17.0, source at `.venv/lib/python3.12/site-packages/pypdf/`.

  * `pypdf.PdfReader(stream, strict=False, password=None, *,
    root_object_recovery_limit=10000)` — `stream` accepts a `str`, a
    file object, or a `pathlib.Path` directly, so no manual `open()` is
    needed. Verified with `inspect.signature`.
  * `reader.pages` is a lazy sequence of `PageObject`; iterating it is the
    supported way to walk a document.
  * `PageObject.extract_text(*args, orientations=(0, 90, 180, 270),
    space_width=200.0, extraction_mode="plain", ...) -> str`. It returns a
    plain `str` and — importantly — returns an EMPTY string rather than
    raising for a page that carries no text layer. That is the normal outcome
    for a scanned or image-only PDF, which is why `_load_pdf` counts empty
    pages and warns instead of treating them as an error: without OCR (out of
    scope here) there is genuinely nothing to extract, and one such page in a
    mostly-textual document should not abort the run.
  * Failure modes live in `pypdf.errors`: `PdfReadError` (malformed file),
    `EmptyFileError`, `FileNotDecryptedError` / `WrongPasswordError`
    (encrypted). These are wrapped in `DocumentLoadError` below so a caller
    sees which file failed, not just a stack trace from inside the parser.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app import config
from app.rag import embeddings, store

logger = logging.getLogger(__name__)


class DocumentLoadError(RuntimeError):
    """Raised when a file exists and is a supported type but cannot be read.

    Distinct from "unsupported extension", which is a skip, not a failure: a
    `.zip` sitting in `data/documents/` is a user putting a file in a folder,
    whereas a `.pdf` that will not parse is something the user needs to know
    about by name. Carrying its own type — the same convention as
    `store.VectorStoreError` — lets a CLI catch load failures and report the
    offending filename without also swallowing embedding or database errors.
    """


SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt", ".md")

EMBED_BATCH_SIZE = 100

_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", " ")


def _load_text_file(path: Path) -> str:
    """Read a `.txt` or `.md` file as UTF-8.

    Markdown is read as-is rather than rendered to plain text. Stripping the
    syntax would cost more than it gains: `#` headings and `-` bullets are
    weak but real signals of structure to the embedding model, and the
    paragraph-first splitter below already keeps sections from being cut in
    half. Leaving the source untouched also means a retrieved chunk can be
    shown back to the user exactly as it appears in the file.

    Raises:
        DocumentLoadError: if the file is not valid UTF-8. Decoding with
            `errors="replace"` instead would silently embed replacement
            characters, quietly degrading retrieval on the affected chunks;
            failing by name lets the user re-save the file in UTF-8.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(
            f"{path.name} is not valid UTF-8 text ({exc}). Re-save it as UTF-8."
        ) from exc


def _load_pdf(path: Path) -> str:
    """Extract the text layer of a PDF, page by page.

    Pages are joined with a blank line so that the chunker's highest-priority
    separator (`\\n\\n`) treats a page boundary as a preferred split point.

    A page whose `extract_text()` comes back empty is counted and skipped, not
    treated as an error — see the pypdf note in the module docstring. If EVERY
    page is empty the file is almost certainly a scan, and this returns an
    empty string; the caller records it as skipped and logs a warning naming
    the file, which is far more actionable than silently ingesting nothing.

    Raises:
        DocumentLoadError: if pypdf cannot parse or decrypt the file.
    """
    try:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
    except PyPdfError as exc:
        raise DocumentLoadError(f"Could not read PDF {path.name}: {exc}") from exc

    populated = [text for text in pages if text.strip()]
    empty_count = len(pages) - len(populated)

    if empty_count:
        logger.warning(
            "%d of %d pages in %s have no extractable text layer (scanned or "
            "image-only pages are skipped; OCR is out of scope).",
            empty_count,
            len(pages),
            path.name,
        )

    return "\n\n".join(populated)


_LOADERS = {
    ".pdf": _load_pdf,
    ".txt": _load_text_file,
    ".md": _load_text_file,
}


def load_document(path: Path) -> str:
    """Dispatch to the loader for `path`'s extension and return its raw text.

    Args:
        path: The file to read. Its suffix is matched case-insensitively, so
            `REPORT.PDF` loads like `report.pdf`.

    Returns:
        The document's full text, possibly empty for an image-only PDF.

    Raises:
        ValueError: if the extension has no loader. Callers that are walking a
            directory should check `SUPPORTED_EXTENSIONS` first and skip
            rather than let this fire.
        DocumentLoadError: if the file is supported but unreadable.
    """
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(
            f"No loader for {path.suffix!r}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    return loader(path)


def _split_keeping_separator(text: str, separator: str) -> list[str]:
    """Split on `separator`, re-attaching it to the end of each left piece.

    `str.split` discards the delimiter, which would mean the concatenation of
    the pieces no longer equals the input — paragraph breaks and the spaces
    between sentences would vanish from the stored text, and every chunk
    boundary would silently eat a character or two. Re-attaching keeps the
    split lossless, so `"".join(pieces) == text` holds and a retrieved chunk
    is a verbatim substring of the source file.
    """
    parts = text.split(separator)
    pieces = [part + separator for part in parts[:-1]]
    pieces.append(parts[-1])
    return [piece for piece in pieces if piece]


def _split_into_pieces(text: str, chunk_size: int, separators: tuple[str, ...]) -> list[str]:
    """Recursively break `text` into fragments no longer than `chunk_size`.

    The "recursive" in recursive character splitting is this: try the most
    semantically meaningful boundary first and only fall back to a coarser one
    for the fragments that are still too long. Paragraph breaks are tried
    before line breaks, line breaks before sentence ends, sentence ends before
    spaces. A fragment that already fits is returned untouched no matter how
    crude the separator that produced it, so well-formed prose is almost
    always cut at a paragraph or sentence boundary and only pathological input
    reaches the coarse rules.

    The last resort, once every separator is exhausted, is a hard slice every
    `chunk_size` characters. That branch is what handles a single token longer
    than a whole chunk — a base64 blob, a minified line, a URL with no spaces
    in it. Without it the function would either return an oversized piece or
    recurse forever on input it cannot divide; the guarantee that the
    separator list shrinks on every recursive call is what makes termination
    unconditional.

    Returns:
        Fragments, each at most `chunk_size` long, whose concatenation is
        exactly `text`.
    """
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        return [text[start : start + chunk_size] for start in range(0, len(text), chunk_size)]

    separator, remaining = separators[0], separators[1:]

    if separator not in text:
        return _split_into_pieces(text, chunk_size, remaining)

    pieces: list[str] = []
    for part in _split_keeping_separator(text, separator):
        if len(part) <= chunk_size:
            pieces.append(part)
        else:
            pieces.extend(_split_into_pieces(part, chunk_size, remaining))
    return pieces


def _merge_pieces(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Pack fragments into chunks of up to `chunk_size`, overlapping by `overlap`.

    Splitting alone would leave every paragraph as its own chunk, including
    one-line ones — which wastes the context budget on chunks too small to
    carry meaning. This greedily accumulates fragments until the next one
    would overflow, emits the chunk, and seeds the following chunk with the
    tail of the one just emitted.

    The overlap allowance is trimmed to `chunk_size - len(next_piece)` rather
    than taken as a flat `overlap`, because a fragment that is itself close to
    `chunk_size` leaves no room for a full tail. That trimming is what keeps
    the "no chunk exceeds chunk_size" guarantee true in the awkward cases
    instead of only the typical ones; the tail is shorter than requested
    there, never longer.

    Because the tail is a literal suffix of the previous chunk and a literal
    prefix of the next, the original text can be reconstructed exactly by
    stripping each chunk's shared prefix — which is what makes "no text is
    lost" checkable rather than merely asserted.
    """
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current)
            allowance = max(0, min(overlap, chunk_size - len(piece)))
            current = (current[-allowance:] if allowance else "") + piece
        else:
            current += piece

    if current:
        chunks.append(current)

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split a document into overlapping, boundary-aware chunks.

    Pure by design: no filesystem, no network, no configuration read beyond
    its own defaults. Chunking is the single decision with the largest effect
    on retrieval quality and the easiest to get subtly wrong, so it is kept
    directly unit-testable — feed it a string, assert on the list.

    Leading and trailing whitespace is stripped from the input as a whole (a
    file that ends with three blank lines should not produce a chunk of
    newlines), but nothing inside is altered: each returned chunk is a
    verbatim substring of the stripped input.

    Args:
        text: The document text. Empty or whitespace-only input yields `[]`
            rather than a chunk of nothing — a blank chunk cannot be embedded
            (`embeddings.embed_documents` rejects it) and would be a
            meaningless search result if it could.
        chunk_size: Maximum characters per chunk. This is a hard ceiling, not
            a target; short documents return a single, shorter chunk.
        overlap: Characters repeated between consecutive chunks. Must be
            smaller than `chunk_size`.

    Returns:
        The chunks in document order. Consecutive chunks share up to `overlap`
        characters.

    Raises:
        ValueError: if `chunk_size` is not positive, or `overlap` is negative
            or greater than or equal to `chunk_size`. An overlap at or above
            the chunk size means every chunk would be re-seeded with the whole
            of its predecessor and the splitter could never advance.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}.")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size}); "
            "otherwise each chunk would repeat the whole of the previous one."
        )

    stripped = text.strip()
    if not stripped:
        return []

    pieces = _split_into_pieces(stripped, chunk_size, _SEPARATORS)
    return _merge_pieces(pieces, chunk_size, overlap)


def _build_records(
    source: str,
    chunks: list[str],
    vectors: list[list[float]],
    ingested_at: datetime,
) -> list[dict]:
    """Zip chunks and their vectors into the `documents` collection schema.

    The stored shape, and why each field is there:

      * `text` — the chunk itself. This is what gets pasted into the Gemini
        prompt, and it is what `store.vector_search`'s `$project` returns.
      * `source` — the filename. It is the citation ("according to
        handbook.pdf"), the key that makes re-ingestion idempotent, and a
        declared `store.FILTER_FIELDS` entry so a future caller can pre-filter
        a search to a single document.
      * `chunk_index` — position within the file, 0-based. Cheap to store and
        the only way to put retrieved fragments back in reading order or to
        say "page 3 of 5" style provenance in the UI.
      * `store.EMBEDDING_FIELD` — the vector Atlas indexes. Named via the
        constant rather than the literal `"embedding"` so the field name has
        exactly one definition, shared with the index JSON in `store.py`.
      * `timestamp` — when this chunk was written, timezone-aware UTC. Same
        field name as the `memory` collection uses, deliberately: both
        collections are read back through the one `$project` whitelist in
        `store.py`, and a second name for the same concept would mean the
        projection had to list both.

    Timezone-aware `datetime` is used rather than a Unix float because BSON
    has a native date type; storing a real date keeps the values sortable and
    range-queryable in Atlas instead of being opaque numbers.
    """
    return [
        {
            "text": chunk,
            "source": source,
            "chunk_index": index,
            store.EMBEDDING_FIELD: vector,
            "timestamp": ingested_at,
        }
        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]


async def _embed_in_batches(chunks: list[str]) -> list[list[float]]:
    """Embed every chunk, capped at `EMBED_BATCH_SIZE` texts per request.

    `embeddings.embed_documents` sends whatever it is given as ONE
    `batchEmbedContents` call — the google-genai SDK stopped splitting large
    inputs for you, as its docstring records. The Gemini API enforces a
    per-request batch limit, so a 400-chunk PDF handed over in one call fails
    outright. Chunking the batches is therefore the caller's job, and this is
    the caller.

    Batches are awaited sequentially rather than fired concurrently with
    `asyncio.gather`. Ingestion is an offline CLI job with no latency budget,
    and the binding constraint is the embedding API's requests-per-minute
    quota, which parallelism spends faster, not slower. Sequential batches
    also fail predictably: chunk N onwards is unwritten, rather than a random
    subset.
    """
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        logger.debug(
            "Embedding chunks %d-%d of %d.", start, start + len(batch) - 1, len(chunks)
        )
        vectors.extend(await embeddings.embed_documents(batch))
    return vectors


async def ingest_file(path: Path) -> int:
    """Load, chunk, embed and store one file. Returns the chunks written.

    Order matters here: the file's previous chunks are deleted only AFTER its
    new vectors have been computed. Embedding is the step that fails — a bad
    API key, an exhausted quota, a network blip — and deleting first would
    turn a transient failure into a document that has vanished from the index
    entirely. Deleting late means the worst case is the old chunks surviving,
    which is a correct, if stale, corpus.

    Args:
        path: A file whose extension is in `SUPPORTED_EXTENSIONS`.

    Returns:
        The number of chunks inserted, or 0 if the file yielded no text (an
        image-only PDF, an empty file). Zero is a legitimate outcome, not an
        error, and the existing chunks are left alone in that case.

    Raises:
        DocumentLoadError: if the file cannot be read.
        RuntimeError: if GOOGLE_API_KEY or MONGODB_URI is unset.
        store.VectorStoreError: if the delete or the insert fails.
    """
    text = load_document(path)
    chunks = chunk_text(text)

    if not chunks:
        logger.warning("%s produced no text to index — skipping.", path.name)
        return 0

    vectors = await _embed_in_batches(chunks)

    source = path.name
    removed = await store.delete_chunks_by_source(config.DOCUMENTS_COLLECTION, source)
    if removed:
        logger.info("Replaced %d previously ingested chunks for %s.", removed, source)

    records = _build_records(source, chunks, vectors, datetime.now(timezone.utc))
    written = await store.upsert_chunks(config.DOCUMENTS_COLLECTION, records)

    logger.info("Ingested %s: %d chunks.", source, written)
    return written


async def ingest_directory(directory: Path = config.DOCUMENTS_DIR) -> dict:
    """Ingest every supported file in `directory`. The top-level entry point.

    Files are processed one at a time, and each is fully written before the
    next is read, so an interrupted run leaves a partially-populated but
    internally consistent corpus: whatever finished is correctly indexed, and
    re-running picks the rest up while replacing the ones already done.

    The vector index is ensured once at the start rather than per file.
    `store.ensure_vector_index` is best-effort by contract — on a cluster tier
    that forbids programmatic index management it logs the JSON to paste into
    the Atlas UI and returns False — so ingestion continues regardless. Doing
    it here means the very first run of the CLI is what surfaces a missing
    index, instead of the first voice question hours later.

    Args:
        directory: Where to look. Defaults to `config.DOCUMENTS_DIR`, i.e.
            `data/documents/`. Not recursive; subdirectories are ignored.

    Returns:
        A summary dict:
          * `files_processed` — files that contributed at least one chunk.
          * `chunks_written` — total chunks inserted across those files.
          * `files_skipped` — files ignored, by count.
          * `skipped` — those files' names, so the CLI can print exactly what
            was left out rather than making the user diff the folder against
            the log.

    Raises:
        FileNotFoundError: if `directory` does not exist. This is a real
            misconfiguration worth stopping on — an empty-but-present folder
            is fine and simply returns zeroes.
        DocumentLoadError, RuntimeError, store.VectorStoreError: propagated
            from `ingest_file`. A failure aborts the run rather than being
            collected into the summary, on the principle that a corpus you
            were told was built but silently is not is worse than a run that
            stopped and said why.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Document directory {directory} does not exist. Create it and add "
            f"{', '.join(SUPPORTED_EXTENSIONS)} files to ingest."
        )

    await store.ensure_vector_index(config.DOCUMENTS_COLLECTION)

    files_processed = 0
    chunks_written = 0
    skipped: list[str] = []

    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(
                "Skipping %s — unsupported extension %r (supported: %s).",
                path.name,
                path.suffix,
                ", ".join(SUPPORTED_EXTENSIONS),
            )
            skipped.append(path.name)
            continue

        written = await ingest_file(path)
        if written:
            files_processed += 1
            chunks_written += written
        else:
            skipped.append(path.name)

    summary = {
        "files_processed": files_processed,
        "chunks_written": chunks_written,
        "files_skipped": len(skipped),
        "skipped": skipped,
    }

    logger.info(
        "Ingestion complete: %d files, %d chunks written, %d skipped.",
        files_processed,
        chunks_written,
        len(skipped),
    )
    return summary
