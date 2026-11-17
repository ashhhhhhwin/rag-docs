import pytest

from app.ingestion.loaders import load_file


def test_loads_text_file(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello there", encoding="utf-8")
    pages = load_file(f)
    assert len(pages) == 1
    assert pages[0].source == "note.txt" and pages[0].page is None


def test_empty_file_gives_no_pages(tmp_path):
    f = tmp_path / "empty.md"
    f.write_text("   \n", encoding="utf-8")
    assert load_file(f) == []


def test_unsupported_type_raises(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b", encoding="utf-8")
    with pytest.raises(ValueError):
        load_file(f)