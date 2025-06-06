"""
Components which processes sequences of chunks in the pipeline.

These could filter, clean or modify the chunks in some way.
"""

from typing import Sequence, Optional, Literal
from logging import getLogger
import re

import pysbd
from cpr_sdk.parser_models import BlockType

from src.models import Chunk, PipelineComponent
from src.utils import filter_and_warn_for_unknown_types

logger = getLogger(__name__)


class IdentityChunkProcessor(PipelineComponent):
    """Returns all the chunks. Useful for testing."""

    def __call__(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Run document cleaning"""
        return list(chunks)


class ChunkTypeFilter(PipelineComponent):
    """Filter out chunks of specified types."""

    def __init__(self, types_to_remove: list[str]) -> None:
        """
        Args:

        :param types_to_remove: the types of chunk to remove
        """

        self.types_to_remove = filter_and_warn_for_unknown_types(types_to_remove)

    def __call__(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Run chunk type filtering."""
        return [
            chunk for chunk in chunks if chunk.chunk_type not in self.types_to_remove
        ]


class RemoveShortTableCells(PipelineComponent):
    """
    Remove table cells under a certain number of characters, or are all numeric.

    These aren't useful for encoding or search.
    """

    def __init__(self, min_chars: int = 0, remove_all_numeric: bool = True) -> None:
        self.min_chars = min_chars
        self.remove_all_numeric = remove_all_numeric

    def __call__(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Run table cell filtering."""
        new_chunks: list[Chunk] = []

        for chunk in chunks:
            if chunk.chunk_type != BlockType.TABLE_CELL:
                new_chunks.append(chunk)
                continue

            if len(chunk.text) < self.min_chars:
                continue

            # Matches strings that are entirely numeric, optionally with a +/- sign,
            # commas, spaces and decimal point
            if self.remove_all_numeric and re.match(
                r"^[+-]?[\d,\s]*\.?\d+$", chunk.text.strip()
            ):
                continue

            new_chunks.append(chunk)

        return new_chunks


class RemoveRepeatedAdjacentChunks(PipelineComponent):
    """
    Remove chunks of the same type that are repeated, keeping the first.

    This is useful for headers, footers and headings that may be repeated once per page
    in a document.
    """

    def __init__(
        self,
        chunk_types=[
            BlockType.SECTION_HEADING,
            BlockType.TITLE,
            BlockType.PAGE_HEADER,
            BlockType.PAGE_FOOTER,
            BlockType.FOOT_NOTE,
        ],
        ignore_case: bool = True,
    ) -> None:
        """
        Args:

        :param chunk_types: list of chunk types to check for repeating
        :param ignore_case: whether filtering ignores case. Defaults to True
        """
        self.chunk_types = chunk_types
        self.ignore_case = ignore_case

    def __call__(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Run repeated adjacent chunk filtering."""
        new_chunks: list[Chunk] = []
        current_chunk_of_type: dict[BlockType, Optional[str]] = {
            chunk_type: None for chunk_type in self.chunk_types
        }

        for chunk in chunks:
            if chunk.chunk_type not in self.chunk_types:
                new_chunks.append(chunk)
                continue

            current_text = current_chunk_of_type[chunk.chunk_type]
            chunk_text = chunk.text.lower() if self.ignore_case else chunk.text

            match current_text:
                case None:
                    # First time seeing this chunk type
                    current_chunk_of_type[chunk.chunk_type] = chunk_text
                    new_chunks.append(chunk)
                case matched_text if matched_text != chunk_text:
                    # Different text than previous chunk of this type
                    current_chunk_of_type[chunk.chunk_type] = chunk_text
                    new_chunks.append(chunk)
                case _:
                    # Same text as previous chunk of this type, skip it
                    continue

        return new_chunks


class AddHeadings(PipelineComponent):
    """
    Add headings to chunks.

    Only works at a single-level. This means that (subheading -> heading) and
    (text -> subheading) relationships will exist, but not (text -> heading) if there
    is a subheading between them.
    """

    def __init__(self) -> None:
        self.heading_types = {BlockType.TITLE, BlockType.TITLE_LOWER_CASE}
        self.subheading_types = {BlockType.SECTION_HEADING, BlockType.PAGE_HEADER}

    def __call__(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Add headings to chunks."""

        current_heading = None
        current_subheading = None

        # Make a copy of the chunks
        new_chunks = list(chunks)

        for chunk in new_chunks:
            if chunk.chunk_type in self.heading_types:
                current_heading = chunk
                current_subheading = None
                # A heading is the top level, so we don't want it to have a heading
                # itself
                continue
            elif chunk.chunk_type in self.subheading_types:
                current_subheading = chunk
                chunk.heading = current_heading
                continue

            chunk.heading = current_subheading or current_heading

        return new_chunks


class RemoveRegexPattern(PipelineComponent):
    """
    Remove text from chunks that matches a regex pattern.

    :param pattern: the regex pattern to match
    :param replace_with: the text to replace the pattern with
    :param skip_partial_replacements: if True, don't do any text replacement – just remove
    chunks that match the pattern exactly. If False, replace the pattern with the text
    specified.
    :param chunk_types: if specified, only process chunks of these types
    """

    def __init__(
        self,
        pattern: str,
        replace_with: str = " ",
        skip_partial_replacements: bool = False,
        chunk_types: list[BlockType] = [],
        ignore_case: bool = False,
    ) -> None:
        self.pattern = pattern
        self.replace_with = replace_with
        self.skip_partial_replacements = skip_partial_replacements
        self.chunk_types = chunk_types or None
        self.ignore_case = ignore_case

    def __call__(self, chunks: list[Chunk]) -> list[Chunk]:
        """Run regex pattern removal."""

        new_chunks: list[Chunk] = []

        for chunk in chunks:
            if self.chunk_types and chunk.chunk_type not in self.chunk_types:
                new_chunks.append(chunk)
                continue

            # If the entire text matches the pattern, skip this chunk
            if re.match(
                f"^{self.pattern}$",
                chunk.text,
                re.IGNORECASE if self.ignore_case else 0,
            ):
                continue

            if self.skip_partial_replacements:
                # No text replacement – add the chunk unchanged
                new_chunks.append(chunk)
            else:
                new_text = re.sub(
                    self.pattern,
                    self.replace_with,
                    chunk.text,
                    re.IGNORECASE if self.ignore_case else 0,
                )

                new_chunk = chunk.model_copy()
                new_chunk.text = new_text.strip()

                if new_chunk.text:
                    new_chunks.append(new_chunk)

        return new_chunks


class RemoveFalseCheckboxes(RemoveRegexPattern):
    """
    Remove false checkboxes from the Azure output.

    These are :selected: and :unselected: patterns.
    """

    def __init__(self) -> None:
        super().__init__(pattern=r"\s?:(?:un)?selected:\s?", replace_with=" ")


class RemoveMisclassifiedPageNumbers(RemoveRegexPattern):
    """
    Remove page header and footer typed chunks that are actually page numbers.

    These should be detected as PAGE_NUMBER type chunks, but aren't always.
    """

    def __init__(self) -> None:
        super().__init__(
            pattern=r"^(?:page\s*)?\s*\d+$",
            skip_partial_replacements=True,
            chunk_types=[BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER],
            ignore_case=True,
        )


class RemoveChunksUnderLength(PipelineComponent):
    """Remove chunks that are under a certain length."""

    def __init__(self, min_num_characters: int) -> None:
        self.min_num_characters = min_num_characters

    def __call__(self, chunks: list[Chunk]) -> list[Chunk]:
        """Run chunk length filtering."""
        return [chunk for chunk in chunks if len(chunk.text) >= self.min_num_characters]


class CombineSuccessiveSameTypeChunks(PipelineComponent):
    """
    Combines successive chunks of the same type in a sequence of chunks.

    :param chunk_types_to_combine: chunk types to be considered for combining. Only
    chunks of the same type will be combined.
    :param text_separator: the separator to use between chunks when merging
    :param merge_into_chunk_type: can be used to merge sequences of chunks into a
    different chunk type. Defaults to None, meaning chunk types will not be changed.
    """

    def __init__(
        self,
        chunk_types_to_combine: list[BlockType],
        text_separator="\n",
        merge_into_chunk_type: Optional[BlockType] = None,
    ) -> None:
        self.chunk_types_to_combine = chunk_types_to_combine
        self.text_separator = text_separator
        self.merge_into_chunk_type = merge_into_chunk_type

    def _set_chunk_type(self, merged_chunk: Chunk) -> Chunk:
        if self.merge_into_chunk_type:
            merged_chunk.chunk_type = self.merge_into_chunk_type
        return merged_chunk

    def __call__(self, chunks: list[Chunk]) -> list[Chunk]:
        """Run chunk combining."""
        new_chunks: list[Chunk] = []
        current_chunk = None

        for chunk in chunks:
            # If the chunk is in types we don't want to combine, we add the previous
            # working 'current_chunk' if there is one.
            if chunk.chunk_type not in self.chunk_types_to_combine:
                if current_chunk:
                    new_chunks.append(self._set_chunk_type(current_chunk))
                    current_chunk = None
                # We also add this chunk and skip to the next iteration.
                new_chunks.append(chunk)
                continue

            # First time seeing a chunk of a type we want to handle
            if not current_chunk:
                current_chunk = chunk
                continue

            # If it's the same type, merge the chunks.
            if chunk.chunk_type == current_chunk.chunk_type:
                current_chunk = current_chunk.merge(
                    [chunk], text_separator=self.text_separator
                )
            # Otherwise, add the current chunk and set a new one.
            else:
                new_chunks.append(self._set_chunk_type(current_chunk))
                current_chunk = chunk

        if current_chunk:
            new_chunks.append(self._set_chunk_type(current_chunk))

        return new_chunks


class CombineTextChunksIntoList(PipelineComponent):
    """
    Combines consecutive text chunks that match a list item pattern into list chunks.

    Also incorporates preceding chunks that end with a colon as part of the list,
    since these often introduce the list.

    If used in a pipeline with `CombineSuccessiveSameTypeChunks` on type TEXT, this
    should go before that.

    TODO: this will join multiple consecutive lists with introductions ending in colons.
    """

    def __init__(self, text_separator: str = "\n") -> None:
        self.text_separator = text_separator
        self.list_item_pattern = r"(^|\n)(?:•|-|·|(?:[\(|\[]?[0-9a-zA-Z]{0,3}[\.|\)|\]|\-|\–])|(?:[A-Za-z][-–])).*?"

    def __call__(self, chunks: list[Chunk]) -> list[Chunk]:
        """Run list item combining."""
        new_chunks: list[Chunk] = []
        current_list_chunk = None
        potential_list_continuation = False
        potential_list_intro = None

        for chunk in chunks:
            # Skip if not a text chunk
            if chunk.chunk_type != BlockType.TEXT:
                if current_list_chunk:
                    new_chunks.append(current_list_chunk)
                    current_list_chunk = None
                if potential_list_intro:
                    new_chunks.append(potential_list_intro)
                    potential_list_intro = None
                potential_list_continuation = False
                new_chunks.append(chunk)
                continue

            # If there is any list item within the chunk, treat it all as a list
            if re.findall(self.list_item_pattern, chunk.text):
                if potential_list_intro:
                    # Create a new list chunk incorporating the introduction
                    if not current_list_chunk:
                        current_list_chunk = potential_list_intro.model_copy(
                            update={"chunk_type": BlockType.LIST}
                        ).merge([chunk], text_separator=self.text_separator)
                    else:
                        # If we already have a list in progress and find a new introduction,
                        # finish the current list and start a new one
                        new_chunks.append(current_list_chunk)
                        current_list_chunk = potential_list_intro.model_copy(
                            update={"chunk_type": BlockType.LIST}
                        ).merge([chunk], text_separator=self.text_separator)
                    potential_list_intro = None
                elif current_list_chunk:
                    # Merge with existing list chunk
                    current_list_chunk = current_list_chunk.merge(
                        [chunk], text_separator=self.text_separator
                    )
                else:
                    # Create new list chunk
                    current_list_chunk = chunk.model_copy(
                        update={"chunk_type": BlockType.LIST}
                    )
                potential_list_continuation = True
            # Check if this might be a continuation of a list item (doesn't look like
            # a complete sentence)
            elif (
                potential_list_continuation
                and current_list_chunk
                and (
                    (
                        chunk.text[0].islower()
                        if chunk.text and chunk.text.strip()
                        else False
                    )
                    or not chunk.text.strip().endswith((".", "!", "?"))
                )
            ):
                current_list_chunk = current_list_chunk.merge(
                    [chunk], text_separator=" "
                )
            # Check if this chunk ends with a colon - potential list introduction
            elif chunk.text.strip().endswith(":"):
                if current_list_chunk:
                    new_chunks.append(current_list_chunk)
                    current_list_chunk = None
                potential_list_intro = chunk
                potential_list_continuation = False
            else:
                if current_list_chunk:
                    new_chunks.append(current_list_chunk)
                    current_list_chunk = None
                if potential_list_intro:
                    new_chunks.append(potential_list_intro)
                    potential_list_intro = None
                potential_list_continuation = False
                new_chunks.append(chunk)

        if current_list_chunk:
            new_chunks.append(current_list_chunk)
        elif potential_list_intro:
            new_chunks.append(potential_list_intro)

        return new_chunks


class SplitTextIntoSentences(PipelineComponent):
    """
    Route text chunks to the appropriate sentence splitter.

    This router selects which sentence splitter implementation to use based on
    the splitter_type parameter.
    """

    def __init__(
        self,
        splitter_type: Literal["pysbd", "basic"] = "pysbd",
        chunk_types_to_ignore: list[BlockType] = [
            BlockType.PAGE_HEADER,
            BlockType.PAGE_FOOTER,
            BlockType.FOOT_NOTE,
        ],
        splitter_kwargs: dict = {},
    ) -> None:
        """Initialize the router with the selected splitter type."""
        self.splitter_type = splitter_type.lower()

        if self.splitter_type == "pysbd":
            self.splitter = SplitTextIntoSentencesPysbd(
                chunk_types_to_ignore=chunk_types_to_ignore,
                **splitter_kwargs,
            )
        elif self.splitter_type == "basic":
            self.splitter = SplitTextIntoSentencesBasic(
                chunk_types_to_ignore=chunk_types_to_ignore,
                **splitter_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown splitter type: {splitter_type}. "
                "Supported types are 'pysbd' and 'basic'."
            )

    def __call__(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Chunk text into sentences."""
        return self.splitter(chunks)


class SplitTextIntoSentencesBasic(PipelineComponent):
    """
    Split chunks of type TEXT in to sentences. Handles sentences which go across chunks.

    :param chunk_types_to_ignore: List of chunk types that won't be split into sentences.
    These chunks are passed through unchanged and are treated as "invisible" to the sentence
    splitter. This means a sentence can span across multiple chunks if they are separated
    by chunks of these ignored types (like headers or footers). Defaults to PAGE_HEADER,
    PAGE_FOOTER, FOOT_NOTE and PAGE_NUMBER.
    """

    def __init__(
        self,
        chunk_types_to_ignore: list[BlockType] = [
            BlockType.PAGE_HEADER,
            BlockType.PAGE_FOOTER,
            BlockType.FOOT_NOTE,
            BlockType.PAGE_NUMBER,
        ],
    ) -> None:
        # Common abbreviations that shouldn't cause sentence splits
        self.common_abbreviations = [
            r"et al\.",
            r"etc\.",
            r"i\.e\.",
            r"e\.g\.",
            r"vs\.",
            r"Mr\.",
            r"Mrs\.",
            r"Dr\.",
            r"Prof\.",
            r"Inc\.",
            r"Ltd\.",
            r"Co\.",
            r"Jr\.",
            r"Sr\.",
            r"St\.",
            r"Ave\.",
            r"Blvd\.",
            r"Rd\.",
            r"Ph\.D\.",
            r"M\.D\.",
        ]

        self.complete_sentence_pattern = re.compile(
            r"[^.!?…]+[.!?…]+(?=\s|\Z)", re.MULTILINE
        )

        self.chunk_types_to_ignore = chunk_types_to_ignore

    def __call__(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Run sentence splitting."""
        new_chunks: list[Chunk] = []
        incomplete_chunk = None
        skipped_chunk_buffer = []

        for chunk in chunks:
            if chunk.chunk_type in self.chunk_types_to_ignore:
                skipped_chunk_buffer.append(chunk)
                continue

            if chunk.chunk_type != BlockType.TEXT:
                if incomplete_chunk:
                    # Add any incomplete sentence before non-text chunk
                    new_chunks.append(incomplete_chunk)
                    incomplete_chunk = None
                new_chunks.append(chunk)
                new_chunks.extend(skipped_chunk_buffer)
                skipped_chunk_buffer = []
                continue

            if incomplete_chunk:
                text = f"{incomplete_chunk.text} {chunk.text}".strip()
                current_chunk = incomplete_chunk.merge([chunk], text_separator=" ")
                current_chunk.text = text
            else:
                text = chunk.text
                current_chunk = chunk

                # If we're starting on a new sentence in the next chunk (there aren't
                # any partial sentences underway), then add the ignored chunks.
                new_chunks.extend(skipped_chunk_buffer)
                skipped_chunk_buffer = []

            # Process text for complete sentences
            complete_sentences = self._extract_complete_sentences(text)

            # Get the remaining text after removing complete sentences
            remaining_text = text
            for sentence in complete_sentences:
                sentence_start = remaining_text.find(sentence)
                sentence_end = sentence_start + len(sentence)
                remaining_text = (
                    remaining_text[:sentence_start] + remaining_text[sentence_end:]
                ).strip()

            # Add complete sentences as new chunks
            for sentence in complete_sentences:
                new_chunk = current_chunk.model_copy(update={"text": sentence})
                new_chunks.append(new_chunk)

            # Keep track of any remaining incomplete sentence
            if remaining_text:
                incomplete_chunk = current_chunk.model_copy(
                    update={"text": remaining_text}
                )
            else:
                incomplete_chunk = None

            # Add any skipped chunks
            new_chunks.extend(skipped_chunk_buffer)
            skipped_chunk_buffer = []

        # Handle any remaining incomplete sentence at the end
        if incomplete_chunk:
            new_chunks.append(incomplete_chunk)

        new_chunks.extend(skipped_chunk_buffer)

        return new_chunks

    def _extract_complete_sentences(self, text: str) -> list[str]:
        """
        Extract complete sentences from text.

        Handles abbreviations which end in full stops at the end of sentences, by
        temporarily replacing them with placeholders whilst sentence boundaries are
        being identified.
        """

        placeholder_map = {}
        modified_text = text

        decimal_percentage_pattern = r"\d+(\.\d+)+%?"
        version_pattern = r"v?\d+(\.\d+)+"  # handles version numbers like v1.2.3
        number_pattern = rf"{decimal_percentage_pattern}|{version_pattern}"

        matches = re.finditer(number_pattern, modified_text)
        for match in reversed(list(matches)):
            placeholder = f"__NUM_{match.start()}__"
            num_text = modified_text[match.start() : match.end()]
            placeholder_map[placeholder] = num_text
            modified_text = (
                modified_text[: match.start()]
                + placeholder
                + modified_text[match.end() :]
            )

        for i, abbr in enumerate(self.common_abbreviations):
            placeholder = f"__ABBR_{i}__"

            # Create pattern that matches the abbreviation not followed by an uppercase letter
            # (which would indicate a new sentence)
            pattern = f"{abbr}(?![A-Z])"
            matches = re.finditer(pattern, modified_text)
            for match in reversed(
                list(matches)
            ):  # Process in reverse to maintain positions
                span = match.span()
                abbr_text = modified_text[span[0] : span[1]]
                placeholder_map[placeholder] = abbr_text
                modified_text = (
                    modified_text[: span[0]] + placeholder + modified_text[span[1] :]
                )

        # Find sentences in the modified text
        sentences = [
            s.strip()
            for s in self.complete_sentence_pattern.findall(modified_text)
            if s.strip()
        ]

        # Replace placeholders back with original abbreviations
        final_sentences = []
        for sentence in sentences:
            for placeholder, original in placeholder_map.items():
                sentence = sentence.replace(placeholder, original)
            sentence = sentence.replace("\n", " ")
            final_sentences.append(sentence)

        return final_sentences


class SplitTextIntoSentencesPysbd(SplitTextIntoSentencesBasic):
    """
    Version of SplitTextIntoSentences that uses Pysbd to split text into sentences.

    This is the recommended method for splitting text into sentences, as it's performant
    plus being relatively fast. See https://github.com/wikimedia/sentencex?tab=readme-ov-file#performance
    """

    def __init__(
        self,
        chunk_types_to_ignore: list[BlockType] = [
            BlockType.PAGE_HEADER,
            BlockType.PAGE_FOOTER,
            BlockType.FOOT_NOTE,
        ],
    ) -> None:
        super().__init__(chunk_types_to_ignore=chunk_types_to_ignore)
        self.segmenter = pysbd.Segmenter(language="en")

    def _extract_complete_sentences(self, text: str) -> list[str]:
        """
        Extract complete sentences from text using pysbd segmenter.

        For sentence merging to work correctly, this method needs to identify only
        sentences that are complete (ending with proper punctuation) and leave
        incomplete sentences to be merged with subsequent chunks.
        """
        if not text.strip():
            return []

        processed_text = text.replace("\n", " ")

        segments = self.segmenter.segment(processed_text)
        complete_sentences = []

        for segment in segments:
            segment_str = str(segment).strip()

            # Although we're using a powerful sentence splitter here, we still need to
            # check for common abbreviations at the end of sentences, as Pysbd has
            # no context of what happens in the next chunk(s) here.
            if re.search(r"[.!?…]$", segment_str) and not any(
                segment_str.endswith(abbr.replace("\\", ""))
                for abbr in self.common_abbreviations
            ):
                if segment_str in processed_text:
                    complete_sentences.append(segment_str)

        return complete_sentences
