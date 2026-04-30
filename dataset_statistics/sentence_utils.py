"""Sentence segmentation with offsets.

This mirrors the sentence-splitting behavior used for the retrieval dataset so
statistics stay aligned with the sentence-level corpus.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from .modules import SentenceSpan

_ABBREVIATIONS: set[str] = {
    "al",
    "app",
    "art",
    "assn",
    "atl",
    "aug",
    "b",
    "ca",
    "cal",
    "cap",
    "cf",
    "ch",
    "cir",
    "co",
    "comm",
    "corp",
    "d",
    "dec",
    "dept",
    "dist",
    "dr",
    "ed",
    "eds",
    "e.g",
    "et",
    "etc",
    "feb",
    "fig",
    "figs",
    "fla",
    "ft",
    "gov",
    "hon",
    "i.e",
    "id",
    "inc",
    "int",
    "jan",
    "jr",
    "jul",
    "jun",
    "ltd",
    "mar",
    "may",
    "messrs",
    "mfg",
    "mr",
    "mrs",
    "ms",
    "no",
    "nos",
    "nov",
    "oct",
    "op",
    "p",
    "pp",
    "prof",
    "rev",
    "sec",
    "secs",
    "sep",
    "sept",
    "ser",
    "sr",
    "st",
    "subsec",
    "sup",
    "u.s",
    "u.s.c",
    "u.s.c.a",
    "v",
    "vol",
    "vs",
}

_INITIALISM_RE = re.compile(r"^(?:[A-Za-z]\.){2,}$")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\.\-]*$")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(text))


def count_words(text: str) -> int:
    return len(tokenize_words(text))


def _prev_token_with_period(text: str, period_idx: int) -> str:
    window_start = max(0, period_idx - 64)
    window = text[window_start : period_idx + 1]
    match = _TOKEN_RE.search(window)
    return match.group(0) if match else ""


def _is_likely_sentence_boundary(text: str, punct_idx: int) -> bool:
    punct = text[punct_idx]
    if punct not in ".?!":
        return False
    if punct in "?!":
        return True

    next_idx = punct_idx + 1
    while next_idx < len(text) and text[next_idx].isspace():
        next_idx += 1
    if next_idx >= len(text):
        return True

    prev_idx = punct_idx - 1
    while prev_idx >= 0 and text[prev_idx].isspace():
        prev_idx -= 1

    if prev_idx >= 0 and text[prev_idx].isdigit() and text[next_idx].isdigit():
        return False

    token = _prev_token_with_period(text, punct_idx)
    norm = token.strip("()[]{}\"'“”‘’")
    norm_lower = norm[:-1].lower() if norm.endswith(".") else norm.lower()

    if _INITIALISM_RE.match(norm):
        return False
    if norm_lower in _ABBREVIATIONS:
        return False
    if len(norm) == 2 and norm[0].isalpha() and norm.endswith(".") and text[next_idx].isupper():
        return False
    if prev_idx >= 0 and text[prev_idx].isalpha() and text[next_idx].isdigit():
        return False

    return True


def _merge_short_spans(text: str, spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    index = 0
    while index < len(spans):
        start, end = spans[index]
        segment = normalize_text(text[start:end])
        alpha_chars = sum(1 for char in segment if char.isalpha())
        word_count = len(segment.split())

        if (alpha_chars < 8 or word_count < 3) and index + 1 < len(spans):
            _, next_end = spans[index + 1]
            merged.append((start, next_end))
            index += 2
            continue

        merged.append((start, end))
        index += 1

    return merged


def split_sentences_with_offsets(text: str) -> List[SentenceSpan]:
    text = str(text)
    if not text:
        return []

    raw_spans: List[Tuple[int, int]] = []
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in ".?!" and _is_likely_sentence_boundary(text, index):
            end = index + 1
            raw_spans.append((start, end))
            start = end
        elif char == "\n" and index + 1 < len(text) and text[index + 1] == "\n":
            end = index + 1
            raw_spans.append((start, end))
            start = end
        index += 1

    if start < len(text):
        raw_spans.append((start, len(text)))

    normalized_spans: List[Tuple[int, int]] = []
    for span_start, span_end in raw_spans:
        if span_start < span_end and normalize_text(text[span_start:span_end]):
            normalized_spans.append((span_start, span_end))

    normalized_spans = _merge_short_spans(text, normalized_spans)

    sentences: List[SentenceSpan] = []
    for span_start, span_end in normalized_spans:
        sent_text = normalize_text(text[span_start:span_end])
        if not sent_text:
            continue
        sentences.append(
            SentenceSpan(start=int(span_start), end=int(span_end), text=sent_text)
        )
    return sentences


def overlap_length(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    start = max(int(start_a), int(start_b))
    end = min(int(end_a), int(end_b))
    return max(0, end - start)
