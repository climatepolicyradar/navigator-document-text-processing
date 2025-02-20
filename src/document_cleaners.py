from abc import ABC, abstractmethod
from typing import Sequence, Optional
from logging import getLogger
import re

from src.models import Chunk, ChunkType

logger = getLogger(__name__)


class BaseDocumentCleaner(ABC):
    """Base class for perfoming cleaning on a sequence of chunks"""

    @abstractmethod
    def __call__(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Run document cleaning"""
        raise NotImplementedError()


class IdentityDocumentCleaner(BaseDocumentCleaner):
    """Returns all the chunks. Useful for testing."""

    def __call__(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Run document cleaning"""
        return list(chunks)


class ChunkTypeFilter(BaseDocumentCleaner):
    """Filter out chunks of specified types."""

    def __init__(self, types_to_remove: list[str]) -> None:
        """
        Args:

        :param types_to_remove: the types of chunk to remove
        """
        for _type in types_to_remove:
            try:
                ChunkType(_type)
            except NameError:
                logger.warning(
                    f"Blocks to filter should be of a known block type, removing {_type} "
                    f"from the list. "
                )
                types_to_remove.remove(_type)

        self.types_to_remove = types_to_remove

    def __call__(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Run chunk type filtering."""
        return [
            chunk for chunk in chunks if chunk.chunk_type not in self.types_to_remove
        ]


class RemoveShortTableCells(BaseDocumentCleaner):
    """
    Remove table cells under a certain number of characters, or are all numeric.

    These aren't useful for encoding or search.
    """

    def __init__(self, min_chars: int = 10, remove_all_numeric: bool = True) -> None:
        self.min_chars = min_chars
        self.remove_all_numeric = remove_all_numeric

    def __call__(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Run table cell filtering."""
        new_chunks: list[Chunk] = []

        for chunk in chunks:
            if chunk.chunk_type != ChunkType.TABLE_CELL:
                new_chunks.append(chunk)
                continue

            if len(chunk.text) < self.min_chars:
                continue

            # Matches strings that are entirely numeric, optionally with a +/- sign,
            # commas, spaces and decimal point
            if self.remove_all_numeric and re.match(
                r"^[+-]?[\d,\s]*\.?\d+$", chunk.text.strip()
            ):
                continue

            new_chunks.append(chunk)

        return new_chunks


class RemoveRepeatedAdjacentChunks(BaseDocumentCleaner):
    """
    Remove chunks of the same type that are repeated, keeping the first.

    This is useful for headers, footers and headings that may be repeated once per page
    in a document.
    """

    def __init__(
        self,
        chunk_types=[
            ChunkType.SECTION_HEADING,
            ChunkType.TITLE,
            ChunkType.PAGE_HEADER,
            ChunkType.PAGE_FOOTER,
            ChunkType.FOOTNOTE,
        ],
        ignore_case: bool = True,
    ) -> None:
        """
        Args:

        :param chunk_types: list of chunk types to check for repeating
        :param ignore_case: whether filtering ignores case. Defaults to True
        """
        self.chunk_types = chunk_types
        self.ignore_case = ignore_case

    def __call__(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Run repeated adjacent chunk filtering."""
        new_chunks: list[Chunk] = []
        current_chunk_of_type: dict[ChunkType, Optional[str]] = {
            chunk_type: None for chunk_type in self.chunk_types
        }

        for chunk in chunks:
            if chunk.chunk_type not in self.chunk_types:
                new_chunks.append(chunk)
                continue

            current_text = current_chunk_of_type[chunk.chunk_type]
            chunk_text = chunk.text.lower() if self.ignore_case else chunk.text

            match current_text:
                case None:
                    # First time seeing this chunk type
                    current_chunk_of_type[chunk.chunk_type] = chunk_text
                    new_chunks.append(chunk)
                case matched_text if matched_text != chunk_text:
                    # Different text than previous chunk of this type
                    current_chunk_of_type[chunk.chunk_type] = chunk_text
                    new_chunks.append(chunk)
                case _:
                    # Same text as previous chunk of this type, skip it
                    continue

        return new_chunks
