"""Extractor registry: decides which extractor handles each file."""

from __future__ import annotations

from pathlib import PurePosixPath

from app.documents.base import DocumentTextExtractor
from app.documents.errors import UnsupportedDocumentFormatError


def file_extension(filename: str) -> str:
    """Return the lowercase, dot-prefixed extension of a filename, or an empty string."""
    return PurePosixPath(filename.replace("\\", "/")).suffix.lower()


class ExtractorRegistry:
    """Maps file extensions and media types to the registered extractors.

    This is the extension point of the package: a new format is added with a single
    ``register`` call and no existing code changes.
    """

    def __init__(self) -> None:
        self._by_extension: dict[str, DocumentTextExtractor] = {}
        self._by_media_type: dict[str, DocumentTextExtractor] = {}

    def register(self, extractor: DocumentTextExtractor) -> DocumentTextExtractor:
        """Register an extractor for its extensions and media types."""
        if not extractor.extensions:
            raise ValueError(f"{type(extractor).__name__} must declare at least one extension.")
        for extension in extractor.extensions:
            self._by_extension[extension.lower()] = extractor
        for media_type in extractor.media_types:
            self._by_media_type[media_type.lower()] = extractor
        return extractor

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Supported extensions, sorted: useful for error messages and API documentation."""
        return tuple(sorted(self._by_extension))

    def supports(self, filename: str) -> bool:
        """Tell whether an extractor handles the extension of the given filename."""
        return file_extension(filename) in self._by_extension

    def resolve(self, *, filename: str | None = None, media_type: str | None = None) -> DocumentTextExtractor:
        """Return the extractor for the file, looking it up by extension and then by media type."""
        if filename:
            extractor = self._by_extension.get(file_extension(filename))
            if extractor is not None:
                return extractor
        if media_type:
            extractor = self._by_media_type.get(media_type.split(";")[0].strip().lower())
            if extractor is not None:
                return extractor

        identified = filename or media_type or "unnamed file"
        supported = ", ".join(self.supported_extensions) or "none"
        raise UnsupportedDocumentFormatError(f"Unsupported format for '{identified}'. Supported: {supported}.")
