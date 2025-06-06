"""CLI to convert JSON documents outputted by the PDF parsing pipeline to embeddings."""

import logging
import logging.config
import os
from pathlib import Path
from typing import Optional

import click
import numpy as np
from tqdm.auto import tqdm

from src.languages import get_docs_of_supported_language
from src.encoders import SBERTEncoder
from src import config
from src.utils import (
    get_files_to_process,
    get_Text2EmbeddingsInput_array,
)
from src.s3 import check_file_exists_in_s3, write_json_to_s3, save_ndarray_to_s3_as_npy
from src.pipeline import Pipeline
from src.chunkers import IdentityChunker
from src.serializers import BasicSerializer
from src.chunk_processors import (
    ChunkTypeFilter,
    RemoveShortTableCells,
    RemoveRepeatedAdjacentChunks,
    RemoveFalseCheckboxes,
)
from src import tokenizers

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
            "formatter": "json",
        },
    },
    "loggers": {},
    "root": {
        "handlers": ["console"],
        "level": LOG_LEVEL,
    },
    "formatters": {"json": {"()": "pythonjsonlogger.jsonlogger.JsonFormatter"}},
}

logger = logging.getLogger(__name__)
logging.config.dictConfig(DEFAULT_LOGGING)


@click.command()
@click.argument(
    "input-dir",
)
@click.argument(
    "output-dir",
)
@click.option(
    "--s3",
    is_flag=True,
    required=False,
    help="Whether or not we are reading from and writing to S3.",
)
@click.option(
    "--redo",
    "-r",
    help="Redo encoding for files that have already been parsed. By default, "
    "files with IDs that already exist in the output directory are skipped.",
    is_flag=True,
    default=False,
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "mps", "cpu"]),
    help="Device to use for embeddings generation",
    required=True,
    default="cpu",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Optionally limit the number of text samples to process. Useful for debugging.",
)
def run_as_cli(
    input_dir: str,
    output_dir: str,
    s3: bool,
    redo: bool,
    device: str,
    limit: Optional[int],
):
    """
    Run CLI to produce embeddings from document parser JSON outputs.

    Each embeddings file is called {id}.json where {id} is the document ID of the
    input. Its first line is the description embedding and all other lines are
    embeddings of each of the text blocks in the document in order. Encoding will
    run CPU unless device is set to 'cuda' or 'mps'.

    Args: input_dir: Directory containing JSON files output_dir: Directory to save
    embeddings to s3: Whether we are reading from and writing to S3. redo: Redo
    encoding for files that have already been parsed. By default, files with IDs that
    already exist in the output directory are skipped. limit (Optional[int]):
    Optionally limit the number of text samples to process. Useful for debugging.
    device (str): Device to use for embeddings generation. Must be either "cuda", "mps",
    or "cpu".
    """

    return run_embeddings_generation(
        input_dir=input_dir,
        output_dir=output_dir,
        s3=s3,
        redo=redo,
        device=device,
        limit=limit,
    )


def run_embeddings_generation(
    input_dir: str,
    output_dir: str,
    s3: bool,
    redo: bool,
    device: str,
    limit: Optional[int],
):
    """
    Run CLI to produce embeddings from document parser JSON outputs.

    See docstring for run_as_cli for details.
    """

    logger.info(
        "Running embeddings generation...",
        extra={
            "props": {
                "input_dir": input_dir,
                "output_dir": output_dir,
                "s3": s3,
                "redo": redo,
                "device": device,
                "limit": limit,
            }
        },
    )

    logger.info("Identifying files to process.")
    files_to_process_ids = get_files_to_process(s3, input_dir, output_dir, redo, limit)
    logger.info(
        f"Found {len(files_to_process_ids)} files to process.",
        extra={"props": {"files_to_process_ids": files_to_process_ids}},
    )

    logger.info("Constructing Text2EmbeddingsInput objects from parser output jsons.")
    tasks = get_Text2EmbeddingsInput_array(input_dir, s3, files_to_process_ids)

    logger.info(
        "Filtering tasks to those with supported languages.",
        extra={"props": {"target_languages": config.TARGET_LANGUAGES}},
    )
    tasks = get_docs_of_supported_language(tasks)
    logger.info(
        f"Found {len(tasks)} tasks with supported languages.",
        extra={
            "props": {
                "tasks": [
                    {
                        "lang": task.languages,
                        "translated": task.translated,
                        "document_id": task.document_id,
                    }
                    for task in tasks
                ]
            }
        },
    )

    logger.info(f"Loading sentence-transformer model {config.SBERT_MODEL}")
    encoder = SBERTEncoder(config.SBERT_MODEL)

    pipeline = Pipeline(
        components=[
            IdentityChunker(),
            ChunkTypeFilter(types_to_remove=config.BLOCKS_TO_FILTER),
            RemoveShortTableCells(min_chars=0, remove_all_numeric=True),
            RemoveRepeatedAdjacentChunks(),
            BasicSerializer(),
            RemoveFalseCheckboxes(),
            tokenizers.NLTKWordTokenizer(),
        ],
        encoder=encoder,
    )

    logger.info(
        "Encoding text from documents.",
        extra={
            "props": {
                "ENCODING_BATCH_SIZE": config.ENCODING_BATCH_SIZE,
                "tasks_number": len(tasks),
            }
        },
    )
    for task in tqdm(tasks, unit="docs"):
        task_output_path = os.path.join(output_dir, task.document_id + ".json")

        try:
            write_json_to_s3(
                task.model_dump_json(indent=2), task_output_path
            ) if s3 else Path(task_output_path).write_text(
                task.model_dump_json(indent=2)
            )
        except Exception as e:
            logger.info(
                "Failed to write embeddings data to s3.",
                extra={"props": {"task_output_path": task_output_path, "exception": e}},
            )

        embeddings_output_path = os.path.join(output_dir, task.document_id + ".npy")

        file_exists = (
            check_file_exists_in_s3(embeddings_output_path)
            if s3
            else os.path.exists(embeddings_output_path)
        )
        if file_exists:
            logger.info(
                f"Embeddings output file '{embeddings_output_path}' already exists, "
                "skipping processing."
            )
            continue

        description_embedding = encoder.encode(task.document_description, device=device)

        if task.html_data is not None and task.html_data.has_valid_text is False:
            _, text_embeddings = pipeline.get_empty_response()
        else:
            _, text_embeddings = pipeline(
                task,
                encoder_batch_size=config.ENCODING_BATCH_SIZE,
                device=device,
            )

        combined_embeddings = (
            np.vstack([description_embedding, text_embeddings])
            if text_embeddings.shape[0] > 0
            else description_embedding.reshape(1, -1)
        )

        save_ndarray_to_s3_as_npy(
            combined_embeddings, embeddings_output_path
        ) if s3 else np.save(embeddings_output_path, combined_embeddings)


if __name__ == "__main__":
    run_as_cli()
