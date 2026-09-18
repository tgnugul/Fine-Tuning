"""[RAG 1/3] 조문 조각 → 임베딩 인덱스 구축.

파싱해둔 조문(chunks.jsonl)을 임베딩 모델로 벡터화해서 저장한다.
이 벡터들이 "의미 좌표"이고, 검색은 질문 좌표와의 거리 계산일 뿐이다.

사용법 (맥, 최초 1회 + 문서가 바뀔 때마다):
  pip install sentence-transformers   # 최초 1회
  python scripts/embed_chunks.py
"""
from __future__ import annotations

import sys

import numpy as np

from common import load_config, read_jsonl, resolve


def chunk_to_text(c: dict) -> str:
    """검색용 텍스트: 제목을 앞에 붙여 '무엇에 관한 조문인지'를 강조."""
    head = f"{c['article']}({c['title']})" if c.get("article") else c.get("title", "")
    return f"{head}\n{c['text']}"


def main() -> None:
    cfg = load_config()["rag"]
    chunks = read_jsonl(resolve(cfg["chunks_file"]))
    if not chunks:
        sys.exit("[오류] chunks 파일이 비어 있습니다. parse_doc.py를 먼저 실행하세요.")
    # 하위 조각이 있는 부모 조문은 임베딩에서 제외 (중복 검색 방지, 원문은 보존)
    chunks = [c for c in chunks if not c.get("has_subs")]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("[오류] sentence-transformers가 필요합니다: pip install sentence-transformers")

    print(f"임베딩 모델 로드: {cfg['embedding_model']} (첫 실행이면 다운로드에 수 분)")
    model = SentenceTransformer(cfg["embedding_model"])

    texts = [chunk_to_text(c) for c in chunks]
    print(f"조각 {len(texts)}개 임베딩 중...")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    out = resolve(cfg["index_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        vectors=np.asarray(vectors, dtype=np.float32),
        chunk_ids=np.array([c["chunk_id"] for c in chunks]),
    )
    print(f"인덱스 저장: {out} (벡터 {vectors.shape[0]}개 × {vectors.shape[1]}차원)")
    print("다음 단계: python scripts/eval_retrieval.py 또는 python scripts/build_rag_dataset.py")


if __name__ == "__main__":
    main()
