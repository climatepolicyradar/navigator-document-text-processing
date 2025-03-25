from nltk.tokenize import word_tokenize

from src.models import PipelineComponent, Chunk


class NLTKWordTokenizer(PipelineComponent):
    """Tokenizes text using the default NLTK word tokenizer."""

    def tokenize(self, text: str) -> list[str]:
        """Tokenize English text using the default NLTK word tokenizer."""
        return word_tokenize(text, language="english")

    def __call__(self, chunks: list[Chunk]) -> list[Chunk]:
        """Tokenize the text of each chunk."""
        return [
            chunk.model_copy(update={"tokens": self.tokenize(chunk.text)})
            for chunk in chunks
        ]
