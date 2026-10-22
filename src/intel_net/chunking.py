"""Bounded stable chunking: one document at a time, deterministic splits."""
from __future__ import annotations

CHUNKER_VERSION_UTF8 = "utf8bound-v1"
CHUNKER_VERSION_TOK = "tok-v1"
OFFLINE_BYTE_LIMIT = 1000
TOKEN_LIMIT = 1000


def _is_continuation(byte: int) -> bool:
    return (byte & 0xC0) == 0x80


def split_utf8(text: str, limit: int = OFFLINE_BYTE_LIMIT) -> list[str]:
    data = text.encode("utf-8")
    parts, i = [], 0
    while i < len(data):
        end = min(i + limit, len(data))
        # Never end mid-character: if data[end] is a continuation byte,
        # end currently splits a multibyte char -> shrink to boundary.
        while end > i and end < len(data) and _is_continuation(data[end]):
            end -= 1
        # Validate (defensive) and shrink further if still invalid.
        while end > i:
            try:
                data[i:end].decode("utf-8", errors="strict")
                break
            except UnicodeDecodeError:
                end -= 1
        if end == i:
            raise ValueError("single character exceeds byte limit")
        parts.append(data[i:end].decode("utf-8", errors="strict"))
        i = end
    return parts or [""]


def _split_tokens(text: str, tokenize, limit: int = TOKEN_LIMIT) -> list[str]:
    """Lossless recursive split: ''.join(parts)==original, each <=limit tokens."""
    ids = tokenize(text)
    if len(ids) <= limit:
        return [text]
    if len(text) <= 1:
        raise ValueError("single character exceeds model token limit")
    mid = len(text) // 2
    # avoid splitting a surrogate pair (python str is unicode, but keep safe on low/high)
    left, right = text[:mid], text[mid:]
    # guard against no-progress
    if not left or not right:
        return [text]
    out = []
    for half in (left, right):
        if len(tokenize(half)) > limit and len(half) > 1:
            out.extend(_split_tokens(half, tokenize, limit))
        else:
            out.append(half)
    # merge tiny neighbours where possible to keep chunk count bounded
    merged: list[str] = []
    for p in out:
        if merged and len(tokenize(merged[-1] + p)) <= limit:
            merged[-1] = merged[-1] + p
        else:
            merged.append(p)
    # final guarantee: any leftover oversize single piece that can still split
    final: list[str] = []
    for p in merged:
        if len(tokenize(p)) > limit and len(p) > 1:
            final.extend(_split_tokens(p, tokenize, limit))
        else:
            final.append(p)
    assert "".join(final) == text, "token split must be lossless"
    return final


def split_text(text: str, tokenize=None) -> tuple[list[str], str, str]:
    """Return (pieces, chunker_version, token_count_basis)."""
    if tokenize is None:
        return split_utf8(text), CHUNKER_VERSION_UTF8, "utf8_upper_bound"
    parts = _split_tokens(text, tokenize, TOKEN_LIMIT)
    assert "".join(parts) == text
    return parts, CHUNKER_VERSION_TOK, "tokenizer"
