import hashlib

from app.ingestion.models import Chunk, Page

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    """Split on the coarsest separator that works, merging pieces up to `size`."""
    if len(text) <= size:
        return [text]
    if not seps:  # no separators left: hard split
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    if sep not in text:
        return _split(text, size, rest)

    chunks: list[str] = []
    current = ""
    for part in text.split(sep):
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(part) > size:
            chunks.extend(_split(part, size, rest))
            current = ""
        else:
            current = part
    if current:
        chunks.append(current)
    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        tail = prev[-overlap:]
        space = tail.find(" ")
        if space != -1:
            tail = tail[space + 1 :]  # start at a word boundary
        out.append(f"{tail} {cur}")
    return out


def split_text(text: str, size: int, overlap: int) -> list[str]:
    pieces = [p.strip() for p in _split(text.strip(), size, SEPARATORS) if p.strip()]
    return _add_overlap(pieces, overlap)


def chunk_pages(pages: list[Page], size: int, overlap: int) -> list[Chunk]:
    chunks = []
    for page in pages:
        for i, text in enumerate(split_text(page.text, size, overlap)):
            digest = hashlib.sha1(f"{page.source}|{page.page}|{i}|{text}".encode()).hexdigest()[:12]
            chunks.append(Chunk(digest, page.source, page.page, i, text))
    return chunks