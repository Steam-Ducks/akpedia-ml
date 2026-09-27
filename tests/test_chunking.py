import pytest

from app.documents import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    split_sentences,
    split_text,
)

SENTENCES = [
    "The quality manual defines the inspection process.",
    "Every part is inspected before assembly.",
    "Rejected parts are tagged and quarantined.",
    "The supervisor reviews the tags weekly.",
    "Records are kept for ten years.",
    "Audits happen twice a year.",
]
TEXT = " ".join(SENTENCES)


def _ends_at_sentence_boundary(chunk: str) -> bool:
    return chunk.endswith((".", "!", "?"))


# --- Definition of Done -------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t\n"])
def test_empty_text_yields_no_chunks(text):
    assert split_text(text) == []


def test_text_smaller_than_a_chunk_yields_a_single_chunk():
    assert len(TEXT) < DEFAULT_CHUNK_SIZE

    assert split_text(TEXT) == [TEXT]


def test_long_text_is_split_into_several_chunks():
    chunks = split_text(TEXT, chunk_size=120, chunk_overlap=0)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert " ".join(chunks) == TEXT


# --- Sentences are never cut in half ----------------------------------------


def test_chunks_end_at_sentence_boundaries():
    chunks = split_text(TEXT, chunk_size=120, chunk_overlap=40)

    assert all(_ends_at_sentence_boundary(chunk) for chunk in chunks)
    for chunk in chunks:
        assert all(sentence in TEXT for sentence in split_sentences(chunk))


def test_a_sentence_that_does_not_fit_starts_the_next_chunk():
    chunks = split_text(TEXT, chunk_size=100, chunk_overlap=0)

    assert chunks[0] == "The quality manual defines the inspection process. Every part is inspected before assembly."
    assert chunks[1].startswith("Rejected parts are tagged and quarantined.")


def test_single_line_breaks_inside_a_sentence_are_not_boundaries():
    pdf_like = "Every part is\ninspected before\nassembly. Records are kept."

    assert split_text(pdf_like, chunk_size=50, chunk_overlap=0) == [
        "Every part is inspected before assembly.",
        "Records are kept.",
    ]


def test_paragraph_breaks_are_boundaries_even_without_punctuation():
    assert split_sentences("Section 1\n\nSection 2") == ["Section 1", "Section 2"]


def test_closing_quotes_after_punctuation_stay_with_their_sentence():
    assert split_sentences('He said "stop." Then he left.') == ['He said "stop."', "Then he left."]


# --- Overlap ------------------------------------------------------------------


def test_consecutive_chunks_share_the_trailing_sentences_of_the_previous_one():
    chunks = split_text(TEXT, chunk_size=120, chunk_overlap=45)

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:], strict=False):
        previous_sentences = split_sentences(previous)
        current_sentences = split_sentences(current)
        assert current_sentences[0] == previous_sentences[-1]


def test_overlap_is_sentence_aligned_and_never_exceeds_the_configured_size():
    chunks = split_text(TEXT, chunk_size=120, chunk_overlap=45)

    for previous, current in zip(chunks, chunks[1:], strict=False):
        shared = [sentence for sentence in split_sentences(current) if sentence in previous]
        assert shared
        assert len(" ".join(shared)) <= 45


def test_zero_overlap_repeats_nothing():
    chunks = split_text(TEXT, chunk_size=120, chunk_overlap=0)

    assert " ".join(chunks) == TEXT


def test_overlap_smaller_than_any_sentence_repeats_nothing():
    chunks = split_text(TEXT, chunk_size=120, chunk_overlap=10)

    assert " ".join(chunks) == TEXT


def test_chunk_size_is_a_hard_limit_even_with_overlap():
    chunks = split_text(TEXT, chunk_size=90, chunk_overlap=80)

    assert all(len(chunk) <= 90 for chunk in chunks)
    assert split_sentences(" ".join(chunks))[-1] == SENTENCES[-1]


def test_every_sentence_appears_in_order_regardless_of_overlap():
    chunks = split_text(TEXT, chunk_size=120, chunk_overlap=45)

    seen: list[str] = []
    for chunk in chunks:
        for sentence in split_sentences(chunk):
            if not seen or sentence != seen[-1]:
                seen.append(sentence)
    assert seen == SENTENCES


# --- Sentences longer than a chunk -------------------------------------------


def test_sentence_longer_than_chunk_size_is_split_on_word_boundaries():
    long_sentence = "word " * 30 + "end."

    chunks = split_text(long_sentence, chunk_size=24, chunk_overlap=0)

    assert all(len(chunk) <= 24 for chunk in chunks)
    assert all(not chunk.startswith(" ") and not chunk.endswith(" ") for chunk in chunks)
    assert " ".join(chunks) == long_sentence.strip()


def test_single_word_longer_than_chunk_size_is_hard_cut():
    assert split_text("abcdefghij", chunk_size=4, chunk_overlap=0) == ["abcd", "efgh", "ij"]


# --- Configuration ------------------------------------------------------------


def test_defaults_are_documented_and_consistent():
    assert DEFAULT_CHUNK_SIZE == 1000
    assert DEFAULT_CHUNK_OVERLAP == 200
    assert DEFAULT_CHUNK_OVERLAP < DEFAULT_CHUNK_SIZE


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "message"),
    [
        (0, 0, "chunk_size must be a positive"),
        (-5, 0, "chunk_size must be a positive"),
        (100, -1, "chunk_overlap cannot be negative"),
        (100, 100, "chunk_overlap must be smaller"),
        (100, 150, "chunk_overlap must be smaller"),
    ],
)
def test_invalid_configuration_is_rejected(chunk_size, chunk_overlap, message):
    with pytest.raises(ValueError, match=message):
        split_text(TEXT, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
