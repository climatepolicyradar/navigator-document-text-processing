import json
from pathlib import Path
from cpr_sdk.parser_models import ParserOutput
import typer
import numpy as np
from typing import Optional
import pandas as pd

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
            chunk_processors.CombineSuccessiveSameTypeChunks(
                chunk_types_to_combine=[
                    BlockType.TITLE,
                    BlockType.TITLE_LOWER_CASE,
                    BlockType.SECTION_HEADING,
                ]
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


def create_metadata_csv(
    output_dir: Path = OUTPUT_DIR, output_file: str = "document_metadata.csv"
):
    """
    Create a CSV file containing document metadata from all JSON files in the output directory.

    Args:
        output_dir: Directory containing the JSON files
        output_file: Name of the output CSV file
    """

    if not output_dir.exists():
        return

    json_files = list(output_dir.glob("*.json"))
    if not json_files:
        return

    # Collect metadata from all files
    all_metadata = []
    metadata_keys = set()

    for file_path in json_files:
        try:
            data = json.loads(file_path.read_text())
            if "document_metadata" in data:
                metadata = data["document_metadata"]
                # Flatten nested metadata if present
                if "metadata" in metadata and isinstance(metadata["metadata"], dict):
                    for key, value in metadata["metadata"].items():
                        metadata[f"metadata_{key}"] = value
                    del metadata["metadata"]

                all_metadata.append(metadata)
                metadata_keys.update(metadata.keys())
        except Exception:
            pass

    if not all_metadata:
        return

    # Convert to DataFrame for easy CSV export
    df = pd.DataFrame(all_metadata)

    # Save to CSV
    csv_path = output_dir / output_file
    df.to_csv(csv_path, index=False)


def run_on_dir(dir_path: Path, limit: Optional[int] = None):
    if dir_path.is_file():
        print(
            "A file was provided, running on a single file. You can also pass a directory to run on all files in the directory."
        )
        run_on_document(dir_path)
        return
    else:
        for file in tqdm(list(dir_path.glob("*.json"))[:limit]):
            run_on_document(file)

    create_metadata_csv()


if __name__ == "__main__":
    typer.run(run_on_dir)
