"""
A CLI for inspecting and searching document chunks created by dev_pipeline.py.

With this, you can:
- View all document chunks or a specific chunk by index
- Run multiple queries one-by-one against a document using vector search

Usage:
    View chunks:
        python cli/inspect_chunks.py view <json_file_path> [--chunk-index <index>]
    Interactive search:
        python cli/inspect_chunks.py interactive <json_file_path>
"""

import json
import numpy as np
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.text import Text
from rich.prompt import Prompt
from rich.layout import Layout
from rich.panel import Panel
from rich import box

from cpr_sdk.parser_models import BlockType

from src.models import ParserOutputWithChunks
from src.encoders import SBERTEncoder

app = typer.Typer()
console = Console()


def load_data(file_path: Path):
    """Load chunks and embeddings data from files."""
    # Load JSON data
    json_path = file_path.with_suffix(".json")
    if not json_path.exists():
        console.print(f"[bold red]Error:[/] File {json_path} does not exist.")
        raise typer.Exit(1)

    data = ParserOutputWithChunks.model_validate(json.loads(json_path.read_text()))

    # Load embeddings
    embeddings_path = file_path.with_suffix(".npy")
    if not embeddings_path.exists():
        console.print(
            f"[bold yellow]Warning:[/] Embeddings file {embeddings_path} not found."
        )
        embeddings = None
    else:
        embeddings = np.load(embeddings_path)

        # Validate that embeddings count matches chunks count
        if len(data.chunks) != embeddings.shape[0]:
            console.print(
                f"[bold red]Error:[/] Mismatch between chunk count ({len(data.chunks)}) and embeddings count ({embeddings.shape[0]})."
            )
            embeddings = None

    return data, embeddings


def display_chunk(
    chunk, index: Optional[int] = None, similarity: Optional[float] = None
):
    """Display a single chunk with nice formatting."""
    # Create main text content
    content = Text(chunk.text, style="white")
    content.append("\n")  # Add spacing after content

    # Create metadata footer
    metadata = Text()
    if index is not None:
        metadata.append(f"#{index} ", style="green bold dim")
    metadata.append("ID: ", style="dim")
    metadata.append(chunk.id, style="cyan dim")
    metadata.append(" | Type: ", style="dim")
    metadata.append(chunk.chunk_type.value, style="magenta dim")
    if chunk.pages:
        metadata.append(" | Pages: ", style="dim")
        metadata.append(str(chunk.pages), style="blue dim")
    if similarity is not None:
        metadata.append(" | Similarity: ", style="dim")
        metadata.append(f"{similarity:.4f}", style="green dim")
    if chunk.heading:
        metadata.append("\nHeading: ", style="dim")
        metadata.append(
            chunk.heading.text[:50] + "..."
            if len(chunk.heading.text) > 50
            else chunk.heading.text,
            style="yellow dim",
        )

    return content + metadata


def extract_headings(data):
    """Extract headings from chunks and return a structured view."""
    heading_types = [
        BlockType.SECTION_HEADING,
        BlockType.TITLE,
        BlockType.PAGE_HEADER,
    ]

    return [chunk for chunk in data.chunks if chunk.chunk_type in heading_types]


def create_layout(main_content, headings_content):
    """Create a split layout with headings sidebar and main content."""
    layout = Layout()

    # Split into two columns
    layout.split_column(Layout(name="header", size=1), Layout(name="main", ratio=1))

    # Split main area into sidebar and content
    layout["main"].split_row(
        Layout(name="sidebar", size=30), Layout(name="content", ratio=1)
    )

    # Add content to each section
    layout["header"].update(Panel("Document Viewer", box=box.HEAVY_HEAD))
    layout["sidebar"].update(Panel(headings_content, title="Headings", box=box.ROUNDED))
    layout["content"].update(Panel(main_content, box=box.ROUNDED))

    return layout


@app.command()
def view(file_path: Path, chunk_index: Optional[int] = None):
    """
    View chunks from the processed document.

    If chunk_index is provided, only that chunk is displayed.
    Otherwise, all chunks are displayed in a pager.
    """
    data, _ = load_data(file_path)
    headings = extract_headings(data)

    # Create headings sidebar content
    headings_content = Text()
    for i, heading in enumerate(headings):
        headings_content.append(f"{i+1}. ", style="green")
        headings_content.append(f"{heading.text}\n", style="yellow")

    if chunk_index is not None:
        if 0 <= chunk_index < len(data.chunks):
            content = display_chunk(data.chunks[chunk_index], chunk_index)
            layout = create_layout(content, headings_content)
            console.print(layout)
        else:
            console.print(
                f"[bold red]Error:[/] Chunk index {chunk_index} is out of range (0-{len(data.chunks)-1})."
            )
    else:
        # Display all chunks with a single headings sidebar
        with console.pager(styles=True):
            console.print(Panel(headings_content, title="Headings", title_align="left"))
            console.print("\n")

            # Display all chunks one after another
            for i, chunk in enumerate(data.chunks):
                content = display_chunk(chunk, i)
                console.print(content)
                console.print()  # Add blank line between chunks


@app.command()
def interactive(file_path: Path):
    """Start an interactive search session."""
    data, embeddings = load_data(file_path)

    if embeddings is None:
        console.print("[bold red]Error:[/] Embeddings not available for search.")
        raise typer.Exit(1)

    encoder = SBERTEncoder(model_name="BAAI/bge-small-en-v1.5")

    console.print(
        f"[bold green]Interactive Search Mode[/] - Document: {file_path.stem}"
    )
    console.print(f"Total chunks: {len(data.chunks)}")
    console.print("[dim]Type 'exit' or 'quit' to end the session[/dim]\n")

    while True:
        query = Prompt.ask("[bold blue]Search query (or 'quit')[/]")
        if query.lower() in ("exit", "quit"):
            break

        top_k = 20

        # Encode the query
        query_embedding = encoder.encode(query)

        # Calculate similarities
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_indices = np.argsort(-similarities)[:top_k]

        # Display results
        console.print(f"\n[bold]Top {top_k} results for:[/] {query}\n")

        for i, idx in enumerate(top_indices):
            similarity = similarities[idx]
            chunk = data.chunks[idx]

            # Create a summarized view
            preview = chunk.text
            result = Text()
            result.append(f"#{idx}, {chunk.chunk_type.value}", style="green bold")
            result.append(f"[Similarity: {similarity:.4f}] ", style="cyan")
            if chunk.heading:
                result.append(f"[Heading: {chunk.heading.text}...] ", style="yellow")
            result.append(preview, style="white")
            console.print(result)
            console.print()

        console.print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    app()
