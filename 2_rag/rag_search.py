"""[RAG 공통] 검색 모듈 - 검색의 실체는 행렬곱 한 줄이다.

rag_chat.py와 eval_retrieval.py가 공유하는 검색 로직.
벡터가 정규화되어 있으므로 코사인 유사도 = 내적 = chunk_vecs @ q_vec.
"""
from __future__ import annotations

import sys

import numpy as np

from common import load_config, read_jsonl, resolve


class Retriever:
    def __init__(self):
        cfg = load_config()["rag"]
        index_path = resolve(cfg["index_file"])
        if not index_path.exists():
            sys.exit(f"[오류] 인덱스가 없습니다: {index_path}\n먼저 python scripts/embed_chunks.py 를 실행하세요.")

        data = np.load(index_path, allow_pickle=False)
        self.vectors = data["vectors"]          # (조각 수, 차원)
        self.chunk_ids = list(data["chunk_ids"])
        self.chunks = {c["chunk_id"]: c for c in read_jsonl(resolve(cfg["chunks_file"]))}
        self.top_k = cfg["top_k"]

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(cfg["embedding_model"])

    def search(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """질문과 가장 가까운 조각들의 (chunk_id, 유사도) 목록."""
        k = top_k or self.top_k
        q_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.vectors @ q_vec               # <- 검색의 전부 (코사인 유사도)
        order = np.argsort(scores)[::-1][:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in order]

    def get_chunk(self, chunk_id: str) -> dict:
        return self.chunks[chunk_id]
