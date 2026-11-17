from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    source: str  # file name
    page: int | None  # 1-based for PDFs, None for plain text
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int | None
    index: int  # position within the page
    text: str
