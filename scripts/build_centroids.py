"""Build the category centroids used by ``POST /api/v1/documents/analyze``.

Reads a CSV manifest of labelled documents, embeds each one the way the endpoint does
(same extractor, chunking, chunk sampling and model) and saves one normalized mean
vector per category.

    uv run python scripts/build_centroids.py --manifest data/categories/manifest.csv

The manifest needs a header row with a path column (``path``, ``file`` or
``filename``) and a category column (``category`` or ``label``). Relative paths are
resolved against the manifest's directory. A document that cannot be read is
reported and skipped; the run fails only when no document could be used.

The service reads the file at startup, so it must be restarted after a rebuild. The
centroids belong to the embedding model that built them: changing ``EMBEDDING_MODEL``
means rebuilding them.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from app.analysis import build_centroids, document_vector, sample_chunks, save_centroids
from app.config import ANALYSIS_MAX_CHUNKS, CENTROIDS_PATH
from app.documents import DocumentProcessingError, extract_text, split_text
from app.embeddings import Embedder, get_embedder

PATH_COLUMNS = ("path", "file", "filename")
CATEGORY_COLUMNS = ("category", "label")


def read_manifest(manifest: Path) -> list[tuple[Path, str]]:
    """Return ``(document path, category)`` pairs from the manifest, in file order."""
    with manifest.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = {name.strip().lower(): name for name in reader.fieldnames or []}
        path_column = next((columns[name] for name in PATH_COLUMNS if name in columns), None)
        category_column = next((columns[name] for name in CATEGORY_COLUMNS if name in columns), None)
        if path_column is None or category_column is None:
            raise SystemExit(
                f"{manifest}: the header needs one of {PATH_COLUMNS} and one of {CATEGORY_COLUMNS}, "
                f"got {reader.fieldnames}."
            )

        entries: list[tuple[Path, str]] = []
        for line, row in enumerate(reader, start=2):
            raw_path = (row.get(path_column) or "").strip()
            category = (row.get(category_column) or "").strip()
            if not raw_path or not category:
                print(f"{manifest}:{line}: skipped, missing path or category.", file=sys.stderr)
                continue
            path = Path(raw_path)
            entries.append((path if path.is_absolute() else manifest.parent / path, category))
    return entries


def embed_document(path: Path, embedder: Embedder, max_chunks: int) -> np.ndarray:
    """Embed one document the way the analysis endpoint does."""
    chunks = split_text(extract_text(path.read_bytes(), filename=path.name))
    if not chunks:
        raise DocumentProcessingError("no extractable text")
    return document_vector(embedder.embed(sample_chunks(chunks, max_chunks)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with the labelled documents.")
    parser.add_argument(
        "--output", type=Path, default=CENTROIDS_PATH, help=f"Where to write the .npz (default: {CENTROIDS_PATH})."
    )
    args = parser.parse_args(argv)

    entries = read_manifest(args.manifest)
    embedder = get_embedder()
    labels: list[str] = []
    vectors: list[np.ndarray] = []
    for path, category in entries:
        try:
            vectors.append(embed_document(path, embedder, ANALYSIS_MAX_CHUNKS))
        except (DocumentProcessingError, OSError) as exc:
            print(f"{path}: skipped, {exc}", file=sys.stderr)
            continue
        labels.append(category)

    if not vectors:
        print("No document could be embedded; nothing was written.", file=sys.stderr)
        return 1

    save_centroids(build_centroids(labels, vectors), args.output)
    counts = Counter(labels)
    print(f"Wrote {len(counts)} centroids from {len(vectors)} of {len(entries)} documents to {args.output}:")
    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
