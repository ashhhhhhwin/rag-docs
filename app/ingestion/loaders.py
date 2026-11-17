from pathlib import Path

from pypdf import PdfReader

from app.ingestion.models import Page

SUPPORTED = {".pdf", ".txt", ".md"}


def load_file(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {suffix}")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [
            Page(path.name, i, (p.extract_text() or "").strip())
            for i, p in enumerate(reader.pages, start=1)
        ]
        return [p for p in pages if p.text]  # skips scanned/empty pages
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [Page(path.name, None, text)] if text else []