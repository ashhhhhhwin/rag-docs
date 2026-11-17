from app.ingestion.models import Chunk
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.retriever import Retriever


def _chunk(i: int, text: str, source: str = "a.txt", page: int | None = None) -> Chunk:
    return Chunk(f"id{i}", source, page, 0, text)


CORPUS = [
    _chunk(1, "Metformin is a first line treatment for type 2 diabetes."),
    _chunk(2, "Hypertension is managed with lifestyle changes and ACE inhibitors."),
    _chunk(3, "Asthma inhalers deliver bronchodilators directly to the lungs."),
    _chunk(4, "Vaccination schedules for infants include several doses in year one."),
    _chunk(5, "Reference code ZX4410 identifies the storage requirements section."),
]


def _filled(store, embedder):
    store.upsert(CORPUS, embedder.embed_documents([c.text for c in CORPUS]))
    return store


def test_rrf_rewards_agreement_between_rankers():
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert max(scores, key=scores.get) == "b"


def test_vector_search_finds_relevant_chunk(store, embedder):
    r = Retriever(_filled(store, embedder), embedder, hybrid=False)
    assert r.search("treatment for diabetes", top_k=3)[0].chunk.id == "id1"


def test_hybrid_finds_exact_identifier(store, embedder):
    r = Retriever(_filled(store, embedder), embedder, hybrid=True)
    assert r.search("ZX4410", top_k=3)[0].chunk.id == "id5"


def test_empty_store_returns_nothing(store, embedder):
    assert Retriever(store, embedder).search("anything", top_k=5) == []


def test_page_numbers_round_trip_including_none(store, embedder):
    chunks = [_chunk(1, "alpha text here", page=3), _chunk(2, "beta text here", page=None)]
    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    pages = {c.id: c.page for c in store.all_chunks()}
    assert pages == {"id1": 3, "id2": None}


def test_upsert_is_idempotent(store, embedder):
    _filled(store, embedder)
    _filled(store, embedder)
    assert store.count() == len(CORPUS)


def test_delete_source_removes_only_that_source(store, embedder):
    chunks = [_chunk(1, "one", source="a.txt"), _chunk(2, "two", source="b.txt")]
    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    store.delete_source("a.txt")
    assert store.sources() == {"b.txt": 1}


def test_bm25_index_refreshes_after_new_ingest(store, embedder):
    r = Retriever(_filled(store, embedder), embedder, hybrid=True)
    r.search("ZX4410", top_k=3)  # builds the BM25 index
    new = [_chunk(6, "Quarterly zebra migration report")]
    store.upsert(new, embedder.embed_documents([new[0].text]))
    assert r.search("zebra", top_k=3)[0].chunk.id == "id6"
