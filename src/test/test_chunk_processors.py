import pytest

from cpr_sdk.parser_models import BlockType

from src.models import Chunk
from src.chunk_processors import (
    RemoveShortTableCells,
    RemoveRepeatedAdjacentChunks,
    AddHeadings,
    RemoveRegexPattern,
    RemoveFalseCheckboxes,
    CombineSuccessiveSameTypeChunks,
    CombineTextChunksIntoList,
    SplitTextIntoSentences,
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


@pytest.mark.parametrize(
    "processor",
    [
        RemoveRegexPattern(pattern=r"\s?:(?:un)?selected:\s?", replace_with=" "),
        RemoveFalseCheckboxes(),
    ],
)
def test_remove_selection_patterns(processor):
    """Test removal of :selected: and :unselected: patterns from chunks."""

    chunks = [
        Chunk(
            text=":selected:",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="1",
        ),
        Chunk(
            text="Some :selected: text",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="2",
        ),
        Chunk(
            text="Multiple :selected: and :unselected: patterns",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="3",
        ),
        Chunk(
            text=":unselected:",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="4",
        ),
        Chunk(
            text=":unselected: :selected:",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="4",
        ),
        Chunk(
            text="Normal text",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="5",
        ),
    ]

    result = processor(chunks)

    assert len(result) == 3

    assert result[0].text == "Some text"
    assert result[1].text == "Multiple and patterns"
    assert result[2].text == "Normal text"


def test_combine_successive_same_type_chunks():
    """Test combining successive chunks of the same type."""
    processor = CombineSuccessiveSameTypeChunks(
        chunk_types_to_combine=[BlockType.TEXT, BlockType.PAGE_HEADER],
        text_separator=" ",
    )
    chunks = [
        Chunk(
            text="First text",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="1",
        ),
        Chunk(
            text="Second text",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="2",
        ),
        Chunk(
            text="Header 1",
            chunk_type=BlockType.PAGE_HEADER,
            bounding_boxes=None,
            pages=None,
            id="3",
        ),
        Chunk(
            text="Header 2",
            chunk_type=BlockType.PAGE_HEADER,
            bounding_boxes=None,
            pages=None,
            id="4",
        ),
        Chunk(
            text="Title",
            chunk_type=BlockType.TITLE,
            bounding_boxes=None,
            pages=None,
            id="5",
        ),
        Chunk(
            text="Third text",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="6",
        ),
    ]

    result = processor(chunks)

    assert len(result) == 4
    assert result[0].text == "First text Second text"
    assert result[0].chunk_type == BlockType.TEXT
    assert result[1].text == "Header 1 Header 2"
    assert result[1].chunk_type == BlockType.PAGE_HEADER
    assert result[2].text == "Title"
    assert result[2].chunk_type == BlockType.TITLE
    assert result[3].text == "Third text"
    assert result[3].chunk_type == BlockType.TEXT


def test_combine_text_chunks_into_list():
    """Test combining text chunks into list chunks when they match list patterns."""
    processor = CombineTextChunksIntoList()
    chunks = [
        Chunk(
            text="Regular text",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="1",
        ),
        Chunk(
            text="• First bullet point\n- Second bullet point\n1. Third bullet point",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="2",
        ),
        Chunk(
            text="1. Numbered item",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="4",
        ),
        Chunk(
            text="Title",
            chunk_type=BlockType.TITLE,
            bounding_boxes=None,
            pages=None,
            id="5",
        ),
        Chunk(
            text="a) Another list item",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="6",
        ),
        Chunk(
            text="[b] Final list item",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="7",
        ),
    ]

    result = processor(chunks)

    assert len(result) == 4
    # First chunk should be regular text
    assert result[0].text == "Regular text"
    assert result[0].chunk_type == BlockType.TEXT

    # Second chunk should be combined list items
    assert (
        result[1].text
        == "• First bullet point\n- Second bullet point\n1. Third bullet point\n1. Numbered item"
    )
    assert result[1].chunk_type == BlockType.LIST

    # Third chunk should be the title
    assert result[2].text == "Title"
    assert result[2].chunk_type == BlockType.TITLE

    # Fourth chunk should be combined list items
    assert result[3].text == "a) Another list item\n[b] Final list item"
    assert result[3].chunk_type == BlockType.LIST


def test_combine_text_chunks_into_list_across_chunks():
    """Test that list patterns are detected and combined across multiple chunks."""
    processor = CombineTextChunksIntoList()
    chunks = [
        Chunk(
            text="Here are the main points:",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="1",
        ),
        Chunk(
            text="• First important point",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="2",
        ),
        Chunk(
            text="that continues in another chunk",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="3",
        ),
        Chunk(
            text="• Second point",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="4",
        ),
        Chunk(
            text="Some unrelated text.",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="5",
        ),
        Chunk(
            text="Additional items to consider:",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="6",
        ),
        Chunk(
            text="1. First numbered item",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="7",
        ),
    ]

    result = processor(chunks)

    # 4. Second list's numbered item
    assert len(result) == 3

    # First chunk should be the combined list with introduction
    assert result[0].chunk_type == BlockType.LIST
    assert (
        result[0].text
        == "Here are the main points:\n• First important point that continues in another chunk\n• Second point"
    )

    # Second chunk should be the unrelated text
    assert result[1].chunk_type == BlockType.TEXT
    assert result[1].text == "Some unrelated text."

    # Third chunk should be the second list introduction with the numbered item
    assert result[2].chunk_type == BlockType.LIST
    assert result[2].text == "Additional items to consider:\n1. First numbered item"


def test_split_text_into_sentences_basic():
    """Test splitting text chunks into sentences."""
    processor = SplitTextIntoSentences()
    chunks = [
        Chunk(
            text="This is sentence one... This is sentence two.",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="1",
        ),
        Chunk(
            text="This is an incomplete",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="2",
        ),
        Chunk(
            text="sentence that spans chunks.",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="3",
        ),
        Chunk(
            text="A title chunk",
            chunk_type=BlockType.TITLE,
            bounding_boxes=None,
            pages=None,
            id="4",
        ),
        Chunk(
            text="Back to sentences! With multiple parts.",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="5",
        ),
    ]

    result = processor(chunks)

    assert len(result) == 6
    assert result[0].text == "This is sentence one..."
    assert result[1].text == "This is sentence two."
    assert result[2].text == "This is an incomplete sentence that spans chunks."
    assert result[3].text == "A title chunk"
    assert result[3].chunk_type == BlockType.TITLE
    assert result[4].text == "Back to sentences!"
    assert result[5].text == "With multiple parts."

    assert all(
        chunks[i].chunk_type == BlockType.TEXT for i in range(len(chunks)) if i != 3
    )


def test_split_text_into_sentences_preserve_non_text_chunks():
    """Test that non-text chunks are preserved in order."""
    processor = SplitTextIntoSentences()
    chunks = [
        Chunk(
            text="First sentence.",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="1",
        ),
        Chunk(
            text="A header",
            chunk_type=BlockType.PAGE_HEADER,
            bounding_boxes=None,
            pages=None,
            id="2",
        ),
        Chunk(
            text="Second sentence. Third sentence.",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            id="3",
        ),
    ]

    result = processor(chunks)

    assert len(result) == 4
    assert result[0].text == "First sentence."
    assert result[1].chunk_type == BlockType.PAGE_HEADER
    assert result[2].text == "Second sentence."
    assert result[3].text == "Third sentence."


def test_split_text_into_sentences_merge_metadata():
    """Test that sentences spanning chunks have their metadata properly merged."""
    processor = SplitTextIntoSentences()
    chunks = [
        Chunk(
            text="This is the start of",
            chunk_type=BlockType.TEXT,
            bounding_boxes=[[(0, 0), (10, 10), (20, 20), (30, 30)]],
            pages=[1],
            id="1",
        ),
        Chunk(
            text="a sentence that spans chunks.",
            chunk_type=BlockType.TEXT,
            bounding_boxes=[[(40, 40), (50, 50), (60, 60), (70, 70)]],
            pages=[2],
            id="2",
        ),
        Chunk(
            text="A complete sentence.",
            chunk_type=BlockType.TEXT,
            bounding_boxes=[[(80, 80), (90, 90), (100, 100), (110, 110)]],
            pages=[3],
            id="3",
        ),
    ]

    result = processor(chunks)

    assert len(result) == 2

    assert result[0].text == "This is the start of a sentence that spans chunks."
    assert result[0].bounding_boxes == [
        [(0, 0), (10, 10), (20, 20), (30, 30)],
        [(40, 40), (50, 50), (60, 60), (70, 70)],
    ]
    assert result[0].pages == [1, 2]

    assert result[1].text == "A complete sentence."
    assert result[1].bounding_boxes == [[(80, 80), (90, 90), (100, 100), (110, 110)]]
    assert result[1].pages == [3]


@pytest.mark.parametrize(
    "input_chunks,expected_text",
    [
        (
            [
                Chunk(
                    text="This is not only an environmental imperative, but an economic one too, as countries around the world start to shift toward low-emissions policies, affecting global trade as well as demand for goods and resources.",
                    chunk_type=BlockType.TEXT,
                    bounding_boxes=None,
                    pages=None,
                    id="1",
                ),
                Chunk(
                    text=", extreme weather, disasters) as well as long- term climatic shifts that impact water security, food security, and human health (DFFE 2019), with a particular focus on vulnerable groups, particularly rural communities, the poor, women, the youth, and children.",
                    chunk_type=BlockType.TEXT,
                    bounding_boxes=None,
                    pages=None,
                    id="2",
                ),
            ],
            "This is not only an environmental imperative, but an economic one too, as countries around the world start to shift toward low-emissions policies, affecting global trade as well as demand for goods and resources, extreme weather, disasters) as well as long- term climatic shifts that impact water security, food security, and human health (DFFE 2019), with a particular focus on vulnerable groups, particularly rural communities, the poor, women, the youth, and children.",
        ),
        (
            [
                Chunk(
                    text="The health impacts from the burning of fossil fuels (a major driver of climate change) also impacts poorer communities, further highlighting these inequities (Gray 2019; Madonsela et al.",
                    chunk_type=BlockType.TEXT,
                    bounding_boxes=None,
                    pages=None,
                    id="3",
                ),
                Chunk(
                    text="2022).",
                    chunk_type=BlockType.TEXT,
                    bounding_boxes=None,
                    pages=None,
                    id="4",
                ),
            ],
            "The health impacts from the burning of fossil fuels (a major driver of climate change) also impacts poorer communities, further highlighting these inequities (Gray 2019; Madonsela et al. 2022).",
        ),
    ],
)
def test_split_text_into_sentences_complex_cases(input_chunks, expected_text):
    """Test sentence splitting with complex cases involving citations and split sentences."""
    if expected_text.startswith("This is not only an environmental imperative"):
        pytest.skip(
            reason="Sentence splitting doesn't yet handle OCR errors which introduce extra punctuation between chunks."
        )

    processor = SplitTextIntoSentences()
    result = processor(input_chunks)

    assert len(result) == 1
    assert result[0].text == expected_text
    assert result[0].chunk_type == BlockType.TEXT
