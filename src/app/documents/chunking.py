"""Split extracted text into overlapping, sentence-aligned chunks.

Chunks are what gets embedded and indexed, so their size must respect the
embedding model's input limit and their boundaries must not cut sentences in half:
a query about a sentence that straddles two chunks would otherwise match neither.

Sizes are measured in characters, not tokens. That keeps the splitter independent
of the embedding model's tokenizer; the defaults below are chosen so that a chunk
stays well inside multilingual-e5-small's 512-token window even after the
``passage:`` prefix is added (Portuguese averages 4-5 characters per token).

Overlap
-------
``chunk_overlap`` is the amount of text, in characters, repeated from the end of one
chunk at the beginning of the next. It exists so that context is not lost exactly at
the chunk boundary: the sentences closing chunk *n* are the same ones opening chunk
*n + 1*, so a passage spanning the boundary is fully contained in at least one of
them. The overlap is sentence-aligned, so it is a ceiling: the repeated stretch is
the largest run of whole trailing sentences that fits in ``chunk_overlap`` characters
(``chunk_overlap = 0`` disables it). ``chunk_size`` is a hard limit and always wins:
if carrying the overlap would push a chunk past it, the oldest overlapped sentences
are dropped first.

The only case where a sentence is cut is a single sentence longer than
``chunk_size``. It is split on word boundaries (and, for a single word longer than
``chunk_size``, on characters) so the size limit is never exceeded.
"""

from __future__ import annotations

import re

#: Default chunk size in characters (roughly 200-250 Portuguese tokens).
DEFAULT_CHUNK_SIZE = 1000

#: Default overlap in characters (roughly one or two sentences of context, 20% of the chunk).
DEFAULT_CHUNK_OVERLAP = 200

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
# End of sentence: terminal punctuation, optionally followed by a closing quote or
# bracket, followed by whitespace. Two lookbehinds because ``re`` needs fixed widths.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|(?<=[.!?…][\"'”’)\]])\s+")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, in order, with whitespace collapsed.

    Paragraph breaks (blank lines) always end a sentence. Single line breaks do not:
    PDF text comes with one newline per visual line, usually mid-sentence, so they
    are treated as plain spaces.
    """
    sentences: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        collapsed = " ".join(paragraph.split())
        if not collapsed:
            continue
        sentences.extend(part for part in _SENTENCE_BOUNDARY.split(collapsed) if part)
    return sentences


def split_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split ``text`` into chunks of at most ``chunk_size`` characters.

    Chunks are built from whole sentences and consecutive chunks share up to
    ``chunk_overlap`` characters of trailing sentences (see the module docstring for
    how the overlap behaves). Empty or whitespace-only text yields no chunks; text
    that fits in one chunk yields exactly one.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive number of characters.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    pieces = [piece for sentence in split_sentences(text) for piece in _fit_sentence(sentence, chunk_size)]

    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and _joined_length(current) + 1 + len(piece) > chunk_size:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, chunk_overlap)
            while current and _joined_length(current) + 1 + len(piece) > chunk_size:
                current.pop(0)
        current.append(piece)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _joined_length(parts: list[str]) -> int:
    """Length of ``parts`` once joined by single spaces."""
    return sum(len(part) for part in parts) + max(len(parts) - 1, 0)


def _overlap_tail(parts: list[str], chunk_overlap: int) -> list[str]:
    """Trailing whole sentences of ``parts`` that fit in ``chunk_overlap`` characters."""
    tail: list[str] = []
    for part in reversed(parts):
        if _joined_length([part, *tail]) > chunk_overlap:
            break
        tail.insert(0, part)
    return tail


def _fit_sentence(sentence: str, chunk_size: int) -> list[str]:
    """Return ``sentence`` as is, or split into word-aligned pieces no longer than ``chunk_size``."""
    if len(sentence) <= chunk_size:
        return [sentence]

    pieces: list[str] = []
    current: list[str] = []
    for word in sentence.split(" "):
        while len(word) > chunk_size:
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.append(word[:chunk_size])
            word = word[chunk_size:]
        if not word:
            continue
        if current and _joined_length(current) + 1 + len(word) > chunk_size:
            pieces.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces
