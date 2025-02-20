from typing import Optional
from abc import ABC, abstractmethod

from enum import Enum
from pydantic import BaseModel


class ChunkType(str, Enum):
    """Possible types of chunk"""

    # TODO: should this replace BlockType in the SDK?

    TEXT = "Text"
    TITLE = "Title"
    LIST = "List"
    TABLE = "Table"
    TABLE_CELL = "TableCell"
    FIGURE = "Figure"
    PAGE_HEADER = "pageHeader"
    PAGE_FOOTER = "pageFooter"
    PAGE_NUMBER = "pageNumber"
    SECTION_HEADING = "sectionHeading"
    DOCUMENT_HEADER = "Document Header"
    FOOTNOTE = "footnote"


class Chunk(BaseModel):
    """A unit part of a document."""

    id: str
    text: str
    chunk_type: ChunkType
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
