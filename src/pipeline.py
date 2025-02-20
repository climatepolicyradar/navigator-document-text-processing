from typing import Optional, Sequence

import numpy as np
from cpr_sdk.parser_models import ParserOutput, PDFTextBlock

from src.models import Chunk, ChunkType
from src.chunkers import BaseChunker
from src.document_cleaners import BaseDocumentCleaner
from src.serializers import BaseSerializer
from src.encoders import BaseEncoder


def parser_output_to_chunks(parser_output: ParserOutput) -> list[Chunk]:
    """Convert a parser output to a list of chunks."""
    if not parser_output.text_blocks:
        return []

    chunks = [
        Chunk(
            id=text_block.text_block_id,
            text=text_block.to_string(),
            chunk_type=ChunkType(text_block.type),
            bounding_boxes=[text_block.coords]
            if isinstance(text_block, PDFTextBlock) and text_block.coords
            else None,
            pages=[text_block.page_number]
            if isinstance(text_block, PDFTextBlock)
            else None,
        )
        for text_block in parser_output.text_blocks
    ]

    return chunks


class Pipeline:
    """Pipeline for document processing."""

    def __init__(
        self,
        chunker: BaseChunker,
        document_cleaners: Sequence[BaseDocumentCleaner],
        serializer: BaseSerializer,
        encoder: Optional[BaseEncoder] = None,
    ) -> None:
        self.chunker = chunker
        self.document_cleaners = document_cleaners
        self.serializer = serializer
        self.encoder = encoder

        self.pipeline_return_type = list[str] if self.encoder is None else np.ndarray

    def get_empty_response(self) -> list[str] | np.ndarray:
        """Return an empty list or array depending on the pipeline configuration."""
        return [] if self.encoder is None else np.empty((0, self.encoder.dimension))

    def __call__(
        self,
        document: ParserOutput,
        encoder_batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> list[str] | np.ndarray:
        """Run the pipeline on a single document."""

        if self.encoder is not None and encoder_batch_size is None:
            raise ValueError(
                "This pipeline contains an encoder but no batch size was set. Please set a batch size."
            )

        chunks = parser_output_to_chunks(document)

        chunks: Sequence[Chunk] = self.chunker(chunks)

        for cleaner in self.document_cleaners:
            chunks = cleaner(chunks)

        # If there are no chunks at this point, return an empty response
        if chunks == []:
            return self.get_empty_response()

        serialized_text: list[str] = [self.serializer(chunk) for chunk in chunks]

        if self.encoder is None:
            return serialized_text
        else:
            return self.encoder.encode_batch(
                text_batch=serialized_text,
                batch_size=encoder_batch_size,  # type: ignore
                device=device,
            )
