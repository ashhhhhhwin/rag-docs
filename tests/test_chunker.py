from app.ingestion.chunker import chunk_pages, split_text
from app.ingestion.models import Page


def test_short_text_is_one_chunk():
    assert split_text("hello world", size=100, overlap=10) == ["hello world"]


def test_respects_size_without_overlap():
    text = "\n\n".join(["word " * 30 for _ in range(10)])
    assert all(len(c) <= 200 for c in split_text(text, size=200, overlap=0))


def test_overlap_shares_text_between_neighbours():
    text = "\n\n".join(f"Paragraph {i}. " + "filler " * 20 for i in range(5))
    chunks = split_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert chunks[0].split()[-1] in chunks[1]


def test_oversized_token_is_hard_split():
    assert all(len(c) <= 50 for c in split_text("x" * 500, size=50, overlap=0))


def test_chunks_keep_source_and_page_and_stable_ids():
    pages = [Page("doc.pdf", 3, "alpha beta. " * 100)]
    a = chunk_pages(pages, 200, 20)
    b = chunk_pages(pages, 200, 20)
    assert a and all(c.source == "doc.pdf" and c.page == 3 for c in a)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


def test_empty_text_gives_no_chunks():
    assert chunk_pages([Page("a.txt", None, "")], 200, 20) == []
