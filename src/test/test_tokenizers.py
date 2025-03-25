from cpr_sdk.parser_models import BlockType

from src.tokenizers import NLTKWordTokenizer
from src.models import Chunk


def test_nltk_word_tokenizer():
    """Test that the NLTK word tokenizer correctly tokenizes text in chunks."""
    # Create test chunks
    chunks = [
        Chunk(
            id="1",
            text="This is a simple test.",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            heading=None,
            tokens=None,
            serialized_text=None,
        ),
        Chunk(
            id="2",
            text="Multiple sentences. With punctuation!",
            chunk_type=BlockType.TEXT,
            bounding_boxes=None,
            pages=None,
            heading=None,
            tokens=None,
            serialized_text=None,
        ),
    ]

    tokenizer = NLTKWordTokenizer()
    result = tokenizer(chunks)

    assert chunks[0].tokens is None
    assert chunks[1].tokens is None

    # Only the tokens should be modified
    assert all(
        result[i].model_copy(update={"tokens": None}) == chunks[i]
        for i in range(len(chunks))
    )

    # Punctuation in tokenized text is important - this is what's missing from Vespa
    # at the moment.
    assert "." in (result[0].tokens or [])
    assert "!" in (result[1].tokens or [])
