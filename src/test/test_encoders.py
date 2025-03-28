import numpy as np

from src import config
from src.encoders import SBERTEncoder, sliding_window


def test_encoder():
    """Assert that we can instantiate an encoder object and encode textual data using the class methods."""

    encoder = SBERTEncoder(config.SBERT_MODEL)

    assert encoder is not None

    assert isinstance(encoder.encode("Hello world!"), np.ndarray)

    assert isinstance(encoder.encode_batch(["Hello world!"] * 100), np.ndarray)

    assert encoder.dimension == 768


def test_encoder_sliding_window():
    """Assert that we can encode long texts using a sliding window."""

    encoder = SBERTEncoder(config.SBERT_MODEL)

    long_text = "Hello world! " * 50
    short_text = "Hello world!"

    batch_to_encode = [short_text, long_text, short_text, short_text]
    embeddings = encoder._encode_batch_using_sliding_window(
        batch_to_encode, batch_size=32
    )

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(batch_to_encode)
    assert embeddings.shape[1] == encoder.dimension

    # embeddings of all short texts should be the same
    assert np.array_equal(embeddings[0, :], embeddings[2, :])
    assert np.array_equal(embeddings[0, :], embeddings[3, :])

    # embedding of long text should not be the same as short text
    assert not np.array_equal(embeddings[0, :], embeddings[1, :])


def test_sliding_window():
    """Tests that the sliding_window function returns the correct embeddings."""
    text = "Hello world! " * 50
    window_size = 10
    stride = 5

    windows = sliding_window(text=text, window_size=window_size, stride=stride)

    assert windows[0] == "Hello worl"
    assert windows[1] == " world! He"
