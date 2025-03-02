"""Classes for chunking documents."""

from typing import Sequence

from src.models import Chunk, PipelineComponent
from src.utils import filter_and_warn_for_unknown_types


class IdentityChunker(PipelineComponent):
    """Returns the text blocks converted to chunks, with no attempts at changing them."""

    def __call__(
        self,
        chunks: Sequence[Chunk],
    ) -> list[Chunk]:
        """Run chunker."""

        return list(chunks)


class FixedLengthChunker(PipelineComponent):
    """
    Chunks the document into fixed length chunks with a maximum number of words.

    If a single chunk is longer than the maximum number of words, it is returned as-is.
    """

    def __init__(
        self, max_chunk_words: int = 200, block_types_to_chunk: list[str] = ["Text"]
    ) -> None:
        self.max_chunk_words = max_chunk_words
        self.block_types_to_chunk = filter_and_warn_for_unknown_types(
            block_types_to_chunk
        )

    def __call__(self, chunks: list[Chunk]) -> list[Chunk]:
        """Run chunker."""

        new_chunks = []
        current_chunk = None

        for chunk in chunks:
            text_length = len(chunk.text.split())

            # If chunk type is not to be chunked, or it's already too long, add it
            # directly
            if (chunk.chunk_type not in self.block_types_to_chunk) or (
                text_length >= self.max_chunk_words
            ):
                if current_chunk is not None:
                    new_chunks.append(current_chunk)
                    current_chunk = None
                new_chunks.append(chunk)
                continue

            if current_chunk is None:
                current_chunk = chunk
            else:
                merged_chunk = current_chunk.merge([chunk])
                merged_length = len(merged_chunk.text.split())

                if merged_length <= self.max_chunk_words:
                    current_chunk = merged_chunk
                else:
                    # Adding this chunk would exceed limit, so add current_chunk and
                    # start a new one
                    new_chunks.append(current_chunk)
                    current_chunk = chunk

        if current_chunk is not None:
            new_chunks.append(current_chunk)

        return new_chunks
