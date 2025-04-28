from typing import Annotated, Optional
from abc import ABC, abstractmethod
import logging
import inspect
import hashlib

from pydantic import BaseModel, field_validator, Field

from cpr_sdk.parser_models import BlockType, ParserOutput

logger = logging.getLogger(__name__)


def get_class_code_hash(cls) -> str:
    """
    Generate a hash of the source code for a class.

    NOTE: this is also sensitive to changes in docstrings.
    """
    try:
        source_code = inspect.getsource(cls)
        md5sum = hashlib.md5(source_code.encode()).hexdigest()
        return md5sum[:8]
    except (TypeError, OSError):
        logger.warning(f"Could not generate hash for component {cls.__name__}")
        return "unknown"


class Chunk(BaseModel):
    """A unit part of a document."""

    id: Annotated[int, Field(strict=True, ge=0)]
    text: str
    chunk_type: BlockType
    heading: Optional["Chunk"] = None
    bounding_boxes: Optional[list[list[tuple[float, float]]]]
    pages: Optional[list[int]]
    tokens: Optional[list[str]] = None
    serialized_text: Optional[str] = None

    @field_validator("bounding_boxes", mode="after")
    @classmethod
    def _verify_bounding_boxes(cls, value) -> None:
        """Verify bounding boxes are compatible their downstream use in the frontend.

        See https://github.com/climatepolicyradar/navigator-frontend/blob/b7f25a3fd2ed815fbb09416b9a046f0f1c41a0cf/src/hooks/usePDFPreview.ts#L20
        """

        coordinate_spec = "[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]"

        if value is not None:
            for box in value:
                if len(box) != 4:
                    raise ValueError("Bounding boxes must have exactly 4 points.")

                # These are the coordinates as defined in the frontend.
                # Note that this ignores some of the coordinates, in the bounding box
                # returned by the parser, so automatically guarantees that the bounding
                # boxes are rectangular.
                # Bounding boxes aren't actually rectangular in practice, so we don't
                # want to validate that here.
                xmin = box[0][0]
                ymin = box[0][1]
                xmax = box[1][0]
                ymax = box[2][1]

                if not (xmin < xmax and ymin < ymax):
                    raise ValueError(
                        f"Minimum and maximum x and y coordinates are not set as expected. Use the pattern {coordinate_spec}"
                    )

        return value

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

        The chunk type is taken from the first chunk.
        The ID is the minimum ID of the chunks being merged.
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
            id=min(chunk.id for chunk in all_chunks),
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

    def __repr__(self) -> str:
        """Return a string representation of the pipeline component."""
        args_repr = ";".join([f"{k}={v}" for k, v in self.__dict__.items()])

        return f"{self.__class__.__name__}___{args_repr}___{get_class_code_hash(self.__class__)}"


class ParserOutputWithChunks(ParserOutput):
    """A parser output with chunks."""

    chunks: list[Chunk]
