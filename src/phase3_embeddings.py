#!/usr/bin/env python3
"""
Phase 3 — Embedding & Retrieval Backbone

  python src/phase3_embeddings.py build
  python src/phase3_embeddings.py query "mechanistic interpretability of transformers"
  python src/phase3_embeddings.py demo
  python src/phase3_embeddings.py interactive
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

CORPUS_PATH = Path("data/corpus_100.json")
CACHE_DIR = Path("data/embeddings")
EMB_CACHE = CACHE_DIR / "corpus_100_embeddings.json"
CHROMA_DIR = Path("data/chroma_corpus")
COLLECTION_NAME = "arxiv_corpus_100"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Corpus not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit("corpus_100.json must be a non-empty JSON array")
    return data


def paper_id(p: dict) -> str:
    if p.get("arxiv_id"):
        return str(p["arxiv_id"])
    if p.get("doi"):
        return f"doi:{p['doi']}"
    if p.get("internal_id"):
        return str(p["internal_id"])
    t = re.sub(r"\s+", " ", (p.get("title") or "")[:60])
    return f"title:{t}"


def text_for_embedding(p: dict) -> str:
    title = (p.get("title") or "").strip()
    abstract = (p.get("abstract") or "").strip()
    if abstract:
        return f"{title}\n{title}\n{abstract}"
    return title


def get_device() -> str:
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_model(model_name: str = DEFAULT_MODEL):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("pip install sentence-transformers chromadb")
    device = get_device()
    print(f"Loading model: {model_name}  (device={device})")
    t0 = time.time()
    model = SentenceTransformer(model_name, device=device)
    print(f"  loaded in {time.time() - t0:.1f}s")
    return model


def embed_texts(model, texts: list[str], batch_size: int = 32) -> list[list[float]]:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [v.tolist() for v in vectors]


def build(args: argparse.Namespace) -> None:
    corpus = load_corpus()
    model = load_model(args.model)
    ids = [paper_id(p) for p in corpus]
    texts = [text_for_embedding(p) for p in corpus]
    print(f"Embedding {len(texts)} papers...")
    t0 = time.time()
    vectors = embed_texts(model, texts, batch_size=args.batch_size)
    print(f"  embedded in {time.time() - t0:.1f}s  dim={len(vectors[0])}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EMB_CACHE.write_text(
        json.dumps({"model": args.model, "dim": len(vectors[0]), "ids": ids, "embeddings": vectors}),
        encoding="utf-8",
    )
    print(f"Cached embeddings -> {EMB_CACHE}")

    import chromadb
    from chromadb.config import Settings

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "model": args.model},
    )
    metadatas, documents = [], []
    for p, text in zip(corpus, texts):
        metadatas.append({
            "arxiv_id": str(p.get("arxiv_id") or ""),
            "title": (p.get("title") or "")[:300],
            "year": str((p.get("published_date") or "")[:4]),
            "doi": str(p.get("doi") or ""),
            "categories": ",".join(p.get("categories") or [])[:200],
            "citation_count": str(p.get("citation_count") if p.get("citation_count") is not None else ""),
        })
        documents.append(text[:2000])
    collection.add(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)
    print(f"Chroma index ready  n={collection.count()}")
    print('Try: python src/phase3_embeddings.py query "mechanistic interpretability circuits"')


def get_collection():
    import chromadb
    from chromadb.config import Settings
    if not CHROMA_DIR.exists():
        raise SystemExit("No index. Run: python src/phase3_embeddings.py build")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    return client.get_collection(COLLECTION_NAME)


def retrieve(query: str, k: int = 10, model_name: str = DEFAULT_MODEL, model=None) -> list[dict[str, Any]]:
    collection = get_collection()
    if model is None:
        model = load_model(model_name)
    qvec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
    res = collection.query(
        query_embeddings=[qvec],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for i, pid in enumerate(res["ids"][0]):
        score = max(0.0, 1.0 - float(res["distances"][0][i]))
        meta = res["metadatas"][0][i] or {}
        out.append({
            "paper_id": pid,
            "score": round(score, 4),
            "title": meta.get("title") or "",
            "arxiv_id": meta.get("arxiv_id") or "",
            "year": meta.get("year") or "",
            "citation_count": meta.get("citation_count") or "",
            "snippet": (res["documents"][0][i] or "")[:240].replace("\n", " "),
        })
    return out


def print_results(query: str, results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(f"Query: {query}")
    print("=" * 70)
    for i, r in enumerate(results, 1):
        aid = r.get("arxiv_id") or r.get("paper_id") or ""
        print(f"{i:2}. [{r['score']:.3f}]  {aid:16}  {r['title'][:58]}")
        if r.get("year") or r.get("citation_count"):
            print(f"     year={r.get('year') or '?'}  cites={r.get('citation_count') or '?'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--batch-size", type=int, default=32)

    q = sub.add_parser("query")
    q.add_argument("query")
    q.add_argument("-k", type=int, default=10)

    d = sub.add_parser("demo")
    d.add_argument("-k", type=int, default=8)

    i = sub.add_parser("interactive")
    i.add_argument("-k", type=int, default=8)

    args = parser.parse_args()
    if args.cmd == "build":
        build(args)
    elif args.cmd == "query":
        print_results(args.query, retrieve(args.query, k=args.k, model_name=args.model))
    elif args.cmd == "demo":
        model = load_model(args.model)
        for qtext in [
            "mechanistic interpretability of large language models and circuit discovery",
            "statistical mechanics of deep learning and generalization",
            "random matrix theory applied to neural network loss landscapes",
            "sparse autoencoders for feature interpretability in transformers",
            "grokking and phase transitions in neural network training",
        ]:
            print_results(qtext, retrieve(qtext, k=args.k, model=model))
            print()
    elif args.cmd == "interactive":
        model = load_model(args.model)
        print("Empty line or q to quit.\n")
        while True:
            try:
                qtext = input("interest> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not qtext or qtext.lower() in {"q", "quit", "exit"}:
                break
            print_results(qtext, retrieve(qtext, k=args.k, model=model))
            print()


if __name__ == "__main__":
    main()