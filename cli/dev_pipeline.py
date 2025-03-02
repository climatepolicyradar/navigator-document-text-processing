import json
from pathlib import Path
from cpr_sdk.parser_models import ParserOutput
import typer
from rich.console import Console
from rich.text import Text


from src.pipeline import Pipeline
from src import chunk_processors, chunkers, serializers


def run_on_document(document_path: Path):
    """
    Print chunks produced by a pipeline run on a parser input JSON.

    :param document_path: path to parserinput json.
    """
    parser_output = ParserOutput.model_validate(json.loads(document_path.read_text()))

    pipeline = Pipeline(
        components=[
            chunk_processors.RemoveShortTableCells(),
            chunk_processors.RemoveRepeatedAdjacentChunks(),
            chunk_processors.ChunkTypeFilter(types_to_remove=["pageNumber"]),
            chunk_processors.RemoveFalseCheckboxes(),
            chunk_processors.CombineTextChunksIntoList(),
            chunk_processors.SplitTextIntoSentences(),
            chunkers.FixedLengthChunker(max_chunk_words=150),
            chunk_processors.AddHeadings(),
            serializers.VerboseHeadingAwareSerializer(),
        ]
    )

    result = pipeline(parser_output)

    console = Console()
    with console.pager(styles=True):
        # Define styles for each block type

        for chunk in result:
            # Create main text content with style based on type
            content = Text(chunk.text, style="white")
            content.append("\n")  # Add spacing after content

            # Create metadata footer
            metadata = Text()
            metadata.append("ID: ", style="dim")
            metadata.append(chunk.id, style="cyan dim")
            metadata.append(" | Type: ", style="dim")
            metadata.append(chunk.chunk_type.value, style="magenta dim")
            if chunk.pages:
                metadata.append(" | Pages: ", style="dim")
                metadata.append(str(chunk.pages), style="blue dim")
            if chunk.heading:
                metadata.append("\nHeading: ", style="dim")
                metadata.append(
                    chunk.heading.text[:50] + "..."
                    if len(chunk.heading.text) > 50
                    else chunk.heading.text,
                    style="yellow dim",
                )

            # Print content and metadata with spacing
            console.print(content + metadata)
            console.print()  # Add blank line between chunks


if __name__ == "__main__":
    typer.run(run_on_document)
