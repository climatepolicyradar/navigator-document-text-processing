import pytest
from src.models import Chunk
from cpr_sdk.parser_models import BlockType


def test_merge_with_bounding_boxes_and_pages():
    """Test merging chunks with bounding boxes and pages."""
    chunk1 = Chunk(
        id="1",
        text="Hello",
        chunk_type=BlockType.TEXT,
        bounding_boxes=[[(0, 0), (1, 0), (1, 1), (0, 1)]],
        pages=[1],
    )
    chunk2 = Chunk(
        id="2",
        text="World",
        chunk_type=BlockType.TEXT,
        bounding_boxes=[[(2, 2), (3, 2), (3, 3), (2, 3)]],
        pages=[2],
    )

    merged = chunk1.merge([chunk2])

    assert merged.id == "1"
    assert merged.text == "Hello World"
    assert merged.chunk_type == BlockType.TEXT
    assert merged.pages == [1, 2]
    assert merged.bounding_boxes == [
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        [(2, 2), (3, 2), (3, 3), (2, 3)],
    ]


def test_merge_multiple() -> None:
    """Test successively merging chunks."""
    chunk1 = Chunk(
        id="1",
        text="Hello ",
        chunk_type=BlockType.TEXT,
        bounding_boxes=[[(0, 0), (1, 0), (1, 1), (0, 1)]],
        pages=[1],
    )
    chunk2 = Chunk(
        id="2",
        text="World",
        chunk_type=BlockType.TEXT,
        bounding_boxes=[[(2, 2), (3, 2), (3, 3), (2, 3)]],
        pages=[2],
    )
    chunk3 = Chunk(
        id="3",
        text="!",
        chunk_type=BlockType.TEXT,
        bounding_boxes=[[(4, 4), (5, 4), (5, 5), (4, 5)]],
        pages=[3],
    )

    merged = chunk1.merge([chunk2, chunk3], text_separator="")

    assert merged.id == "1"
    assert merged.text == "Hello World!"
    assert merged.chunk_type == BlockType.TEXT
    assert merged.pages == [1, 2, 3]
    assert merged.bounding_boxes == [
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        [(2, 2), (3, 2), (3, 3), (2, 3)],
        [(4, 4), (5, 4), (5, 5), (4, 5)],
    ]


def test_merge_incompatible_properties():
    """Test merging chunks with incompatible properties raises ValueError."""
    chunk1 = Chunk(
        id="1",
        text="Hello",
        chunk_type=BlockType.TEXT,
        bounding_boxes=[[(0, 0), (1, 0), (1, 1), (0, 1)]],
        pages=[1],
    )
    chunk2 = Chunk(
        id="2", text="World", chunk_type=BlockType.TEXT, bounding_boxes=None, pages=None
    )

    with pytest.raises(ValueError):
        chunk1.merge([chunk2])


def test_merge_with_optional_properties():
    """Test merging chunks with optional properties results in those properties being None."""
    chunk1 = Chunk(
        id="1",
        text="Hello",
        chunk_type=BlockType.TEXT,
        bounding_boxes=None,
        pages=None,
        tokens=["Hello"],
        heading=Chunk(
            id="h1",
            text="Title",
            chunk_type=BlockType.SECTION_HEADING,
            bounding_boxes=None,
            pages=None,
        ),
    )
    chunk2 = Chunk(
        id="2",
        text="World",
        chunk_type=BlockType.TEXT,
        bounding_boxes=None,
        pages=None,
        tokens=None,
        heading=None,
    )

    merged = chunk1.merge([chunk2])
    assert merged.tokens is None
    assert merged.heading is None


def test_merge_custom_separator():
    """Test merging chunks with a custom text separator."""
    chunk1 = Chunk(
        id="1", text="Hello", chunk_type=BlockType.TEXT, bounding_boxes=None, pages=None
    )
    chunk2 = Chunk(
        id="2", text="World", chunk_type=BlockType.TEXT, bounding_boxes=None, pages=None
    )

    merged = chunk1.merge([chunk2], text_separator="\n")
    assert merged.text == "Hello\nWorld"
