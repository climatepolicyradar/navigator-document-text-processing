from typing import Optional
from abc import ABC, abstractmethod
import logging

from pydantic import BaseModel

from cpr_sdk.parser_models import BlockType, ParserOutput

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    """A unit part of a document."""

    id: str
    text: str
    chunk_type: BlockType
    # TODO: do we want multiple headings here? this is what docling does.
    heading: Optional["Chunk"] = None
    # Bounding boxes can be arbitrary polygons according to Azure
    # TODO: is this true according to the backend or frontend?
    bounding_boxes: Optional[list[list[tuple[float, float]]]]
    pages: Optional[list[int]]
    tokens: Optional[list[str]] = None
    serialized_text: Optional[str] = None

    def _verify_bbox_and_pages(self) -> None:
        if self.bounding_boxes is not None and self.pages is not None:
            assert len(self.bounding_boxes) == len(self.pages)

    def _check_incompatible_properties(self, others: list["Chunk"]) -> list[str]:
        """Check if nullable properties are consistent across chunks to be merged."""
        nullable_properties = ["bounding_boxes", "pages"]
        properties_with_issues = []

        for prop in nullable_properties:
            self_is_none = getattr(self, prop) is None
            if any((getattr(chunk, prop) is None) != self_is_none for chunk in others):
                properties_with_issues.append(prop)

        return properties_with_issues

    def _check_optional_properties(self, others: list["Chunk"]) -> list[str]:
        """Check which optional properties will be lost in merge."""
        optional_properties = ["heading", "tokens", "serialized_text"]
        properties_with_issues = []

        for prop in optional_properties:
            if getattr(self, prop) is not None or any(
                getattr(chunk, prop) is not None for chunk in others
            ):
                properties_with_issues.append(prop)

        return properties_with_issues

    def merge(self, others: list["Chunk"], text_separator: str = " ") -> "Chunk":
        """
        Merge multiple chunks into a single chunk.

        The ID and chunk type are taken from the first chunk.
        """

        if not others:
            return self

        if not isinstance(others, list) or not all(
            isinstance(chunk, Chunk) for chunk in others
        ):
            raise ValueError("Chunks to be merged must be a list of Chunk objects.")

        incompatible_properties = self._check_incompatible_properties(others)
        if incompatible_properties:
            raise ValueError(
                f"Properties {incompatible_properties} of chunks being merged must be either all None or all not None."
            )

        optional_properties_with_issue = self._check_optional_properties(others)
        if optional_properties_with_issue:
            logger.warning(
                f"Properties {optional_properties_with_issue} of chunks being merged have been set for one or more chunks. These properties will be lost in the merge."
            )

        all_chunks = [self] + others
        all_texts = [chunk.text for chunk in all_chunks]
        if self.bounding_boxes is None:
            combined_bounding_boxes = None
        else:
            combined_bounding_boxes = [
                box
                for chunk in all_chunks
                if chunk.bounding_boxes is not None
                for box in chunk.bounding_boxes
            ]

        if self.pages is None:
            combined_pages = None
        else:
            combined_pages = [
                page
                for chunk in all_chunks
                if chunk.pages is not None
                for page in chunk.pages
            ]

        return Chunk(
            # TODO: can we better handle IDs when merging chunks?
            id=self.id,
            text=text_separator.join(all_texts),
            chunk_type=self.chunk_type,
            bounding_boxes=combined_bounding_boxes,
            pages=combined_pages,
            heading=None,
            tokens=None,
            serialized_text=None,
        )


class PipelineComponent(ABC):
    """
    A component of the pipeline.

    When called, takes a list of chunks as input and returns a list of chunks. This
    should be used as the base class for every pipeline component except for the
    encoder.
    """

    @abstractmethod
    def __call__(self, chunks: list[Chunk]) -> list[Chunk]:
        """Base class for any pipeline component."""
        raise NotImplementedError


class ParserOutputWithChunks(ParserOutput):
    """A parser output with chunks."""

    chunks: list[Chunk]
