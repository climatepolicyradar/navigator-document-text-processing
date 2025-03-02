from typing import Optional

from src.models import Chunk, PipelineComponent


class BasicSerializer(PipelineComponent):
    """
    The most basic serializer.

    Returns the chunk with text added in the `serialized_text` field.
    """

    def __call__(self, chunks: list) -> list[Chunk]:
        """Run serialization."""
        new_chunks = list(chunks)

        for chunk in new_chunks:
            chunk.serialized_text = chunk.text

        return new_chunks


class HeadingAwareSerializer(PipelineComponent):
    """
    Returns the text of the chunk and its heading according to a template.

    Template defaults to "{text} – {heading}". If the chunk doesn't have an associated
    heading, just the text will be returned.
    """

    def __init__(self, template: Optional[str]) -> None:
        self.template: str = template or "{text} – {heading}"

    def __call__(self, chunks: list[Chunk]) -> list[Chunk]:
        """Run serialization."""

        new_chunks = list(chunks)

        for chunk in new_chunks:
            chunk.serialized_text = (
                self.template.format(text=chunk.text, heading=chunk.heading)
                if chunk.heading
                else chunk.text
            )

        return chunks


class VerboseHeadingAwareSerializer(HeadingAwareSerializer):
    """
    Like a heading-aware serializer, but provides a description of what the heading is.

    Adapted from dsRAG, who call a version of this where LLMs create headings
    'AutoContext'.
    (https://github.com/D-Star-AI/dsRAG/tree/main?tab=readme-ov-file#autocontext-contextual-chunk-headers)
    """

    def __init__(self) -> None:
        super().__init__(
            template="Section context: this excerpt is from the section titled '{heading}'. {text}"
        )
