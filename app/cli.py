import argparse
import sys
from pathlib import Path

from app.config import get_settings
from app.ingestion.pipeline import ingest_dir
from app.retrieval.embeddings import get_embedder
from app.retrieval.retriever import Retriever
from app.retrieval.store import make_store


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # avoid cp1252 crashes in Windows cmd
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("path", type=Path)
    srch = sub.add_parser("search")
    srch.add_argument("query")
    srch.add_argument("-k", type=int, default=None)
    srch.add_argument("--vector-only", action="store_true")
    args = parser.parse_args()

    s = get_settings()
    store, embedder = make_store(), get_embedder()

    if args.cmd == "ingest":
        for name, n in ingest_dir(args.path, store, embedder, s.chunk_size, s.chunk_overlap):
            print(f"{name}: {n} chunks")
        print(f"Total chunks in store: {store.count()}")
    else:
        retriever = Retriever(store, embedder, hybrid=not args.vector_only, rrf_k=s.rrf_k)
        for i, hit in enumerate(retriever.search(args.query, args.k or s.top_k), start=1):
            c = hit.chunk
            vs = f"{hit.vector_score:.3f}" if hit.vector_score is not None else "n/a"
            print(f"[{i}] score={hit.score:.4f} cosine={vs} {c.source} p.{c.page}")
            print("    " + c.text[:200].replace("\n", " ") + "...\n")


if __name__ == "__main__":
    main()