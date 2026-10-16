"""Suggestions made from an uploaded document: title, description and category.

Nothing here is persisted. ``akpedia-server`` uses the suggestions to pre-fill the
upload form; the user and the approver still decide.
"""

from app.analysis.categories import (
    CategoryCentroids,
    CategoryScores,
    build_centroids,
    document_vector,
    get_centroids,
    load_centroids,
    sample_chunks,
    save_centroids,
)
from app.analysis.suggestions import (
    MAX_DESCRIPTION_LENGTH,
    MAX_TITLE_LENGTH,
    MIN_DESCRIPTION_LENGTH,
    suggest_description,
    suggest_title,
    truncate_at_word,
)

__all__ = [
    "MAX_DESCRIPTION_LENGTH",
    "MAX_TITLE_LENGTH",
    "MIN_DESCRIPTION_LENGTH",
    "CategoryCentroids",
    "CategoryScores",
    "build_centroids",
    "document_vector",
    "get_centroids",
    "load_centroids",
    "sample_chunks",
    "save_centroids",
    "suggest_description",
    "suggest_title",
    "truncate_at_word",
]
