from cpr_sdk.parser_models import BlockType

from src.models import Chunk
from src.chunk_processors import (
    RemoveShortTableCells,
    RemoveRepeatedAdjacentChunks,
    AddHeadings,
)


def test_remove_short_table_cells_drop_numeric():
    """Test filtering of short and numeric table cells."""
    cleaner = RemoveShortTableCells(min_chars=5, remove_all_numeric=True)
    chunks = [
        Chunk(
            text="short",
            chunk_type=BlockType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
            id="1",
        ),
        Chunk(
            text="this is long enough",
            chunk_type=BlockType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
            id="2",
        ),
        Chunk(
            text="123.45",
            chunk_type=BlockType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
            id="3",
        ),
        Chunk(
            text="1,234.56",
            chunk_type=BlockType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
            id="4",
        ),
        Chunk(
            text="-123.45",
            chunk_type=BlockType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
            id="5",
        ),
        Chunk(
            text="not a table cell",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="6",
        ),
        Chunk(
            text="12345 with text",
            chunk_type=BlockType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
            id="7",
        ),
    ]

    result = cleaner(chunks)

    assert len(result) == 4
    assert result[0].text == "short"
    assert result[1].text == "this is long enough"
    assert result[2].text == "not a table cell"
    assert result[3].text == "12345 with text"


def test_remove_short_table_cells_keep_numeric():
    """Test keeping numeric cells when remove_all_numeric is False."""
    cleaner = RemoveShortTableCells(min_chars=6, remove_all_numeric=False)

    chunks = [
        Chunk(
            text="short",
            chunk_type=BlockType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
            id="1",
        ),
        Chunk(
            text="123.45",
            chunk_type=BlockType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
            id="2",
        ),
        Chunk(
            text="1,234.56",
            chunk_type=BlockType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
            id="3",
        ),
    ]

    result = cleaner(chunks)

    assert len(result) == 2
    assert result[0].text == "123.45"
    assert result[1].text == "1,234.56"


def test_remove_repeated_adjacent_chunks():
    """Test filtering of repeated chunks of the same type."""
    cleaner = RemoveRepeatedAdjacentChunks()
    chunks = [
        Chunk(
            text="Header",
            chunk_type=BlockType.PAGE_HEADER,
            bounding_boxes=None,
            pages=None,
            id="1",
        ),
        Chunk(
            text="Some content",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="2",
        ),
        Chunk(
            text="Header",
            chunk_type=BlockType.PAGE_HEADER,
            bounding_boxes=None,
            pages=None,
            id="3",
        ),
        Chunk(
            text="Different Header",
            chunk_type=BlockType.PAGE_HEADER,
            bounding_boxes=None,
            pages=None,
            id="4",
        ),
        Chunk(
            text="footnote",
            chunk_type=BlockType.FOOT_NOTE,
            bounding_boxes=None,
            pages=None,
            id="5",
        ),
        Chunk(
            text="important title",
            chunk_type=BlockType.TITLE,
            bounding_boxes=None,
            pages=None,
            id="6",
        ),
        Chunk(
            text="footnote",
            chunk_type=BlockType.FOOT_NOTE,
            bounding_boxes=None,
            pages=None,
            id="7",
        ),
    ]

    result = cleaner(chunks)

    assert len(result) == 5
    assert result[0].text == "Header"
    assert result[1].text == "Some content"
    assert result[2].text == "Different Header"
    assert result[3].text == "footnote"
    assert result[4].text == "important title"


def test_remove_repeated_adjacent_chunks_case_sensitive():
    """Test case-sensitive filtering of repeated chunks."""
    cleaner = RemoveRepeatedAdjacentChunks(ignore_case=False)
    chunks = [
        Chunk(
            text="Header",
            chunk_type=BlockType.PAGE_HEADER,
            bounding_boxes=None,
            pages=None,
            id="1",
        ),
        Chunk(
            text="HEADER",
            chunk_type=BlockType.PAGE_HEADER,
            bounding_boxes=None,
            pages=None,
            id="2",
        ),
    ]

    result = cleaner(chunks)

    assert len(result) == 2
    assert result[0].text == "Header"
    assert result[1].text == "HEADER"


def test_add_headings():
    """Test adding headings to chunks with title and section headings."""
    cleaner = AddHeadings()
    chunks = [
        Chunk(
            text="Document Title",
            chunk_type=BlockType.TITLE,
            bounding_boxes=None,
            pages=None,
            id="1",
        ),
        Chunk(
            text="Regular text",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="2",
        ),
        Chunk(
            text="Section 1",
            chunk_type=BlockType.SECTION_HEADING,
            bounding_boxes=None,
            pages=None,
            id="3",
        ),
        Chunk(
            text="Text under section 1",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="4",
        ),
        Chunk(
            text="More text under section 1",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="5",
        ),
        Chunk(
            text="Page Header",
            chunk_type=BlockType.PAGE_HEADER,
            bounding_boxes=None,
            pages=None,
            id="6",
        ),
        Chunk(
            text="Text under page header",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="7",
        ),
    ]

    results = cleaner(chunks)

    # Title shouldn't have a heading
    assert results[0].heading is None
    assert all(chunk.heading is not None for chunk in results[1:])

    # Text under the title
    assert results[1].heading.id == results[0].id  # type: ignore

    # Section headings under the title
    assert results[2].heading.id == results[0].id  # type: ignore
    assert results[5].heading.id == results[0].id  # type: ignore

    # Text under the first section heading
    assert results[3].heading.id == results[2].id  # type: ignore
    assert results[4].heading.id == results[2].id  # type: ignore

    # Text under the second section heading
    assert results[6].heading.id == results[5].id  # type: ignore
