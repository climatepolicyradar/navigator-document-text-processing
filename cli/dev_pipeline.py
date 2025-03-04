import json
from pathlib import Path
from cpr_sdk.parser_models import ParserOutput
import typer
import numpy as np
from typing import Optional

from cpr_sdk.parser_models import BlockType
from tqdm.auto import tqdm

from src.pipeline import Pipeline
from src.models import ParserOutputWithChunks
from src import chunk_processors, chunkers, serializers, encoders

OUTPUT_DIR = Path(__file__).parent / "data/dev_pipeline_output"


def run_on_document(document_path: Path):
    """
    Run a development pipeline on a parser output JSON.

    Outputs a JSON file with chunks, and optionally a numpy file with embeddings.
    """

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    parser_output = ParserOutput.model_validate(json.loads(document_path.read_text()))

    pipeline = Pipeline(
        components=[
            chunk_processors.RemoveShortTableCells(),
            chunk_processors.RemoveRepeatedAdjacentChunks(),
            chunk_processors.ChunkTypeFilter(types_to_remove=["pageNumber"]),
            chunk_processors.RemoveFalseCheckboxes(),
            chunk_processors.CombineTextChunksIntoList(),
            chunk_processors.CombineSuccessiveSameTypeChunks(
                chunk_types_to_combine=[BlockType.TABLE_CELL],
                merge_into_chunk_type=BlockType.TABLE,
            ),
            chunk_processors.SplitTextIntoSentences(),
            chunkers.FixedLengthChunker(max_chunk_words=150),
            chunk_processors.AddHeadings(),
            serializers.VerboseHeadingAwareSerializer(),
        ],
        encoder=encoders.SBERTEncoder(model_name="BAAI/bge-small-en-v1.5"),
    )

    chunks, embeddings = pipeline(parser_output, encoder_batch_size=16)

    parser_output_with_chunks = ParserOutputWithChunks(
        chunks=chunks,
        **parser_output.model_dump(),
    )
    output_path = OUTPUT_DIR / f"{document_path.stem}.json"
    output_path.write_text(parser_output_with_chunks.model_dump_json(indent=4))

    if embeddings is not None:
        embeddings_path = OUTPUT_DIR / f"{document_path.stem}.npy"
        np.save(embeddings_path, embeddings)


def run_on_dir(dir_path: Path, limit: Optional[int] = None):
    for file in tqdm(list(dir_path.glob("*.json"))[:limit]):
        run_on_document(file)


if __name__ == "__main__":
    typer.run(run_on_dir)
