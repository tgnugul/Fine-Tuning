"""[RAG 3/3] 검색 정확도 평가 - RAG 평가의 첫 번째 축.

RAG 평가는 반드시 둘로 쪼갠다:
  ① 검색(retrieval): 맞는 조문을 가져왔는가 ← 이 스크립트 (hit@k)
  ② 생성(generation): 가져온 근거로 충실히 답했는가 ← 답변 평가에서

합성 Q&A의 chunk_id(근거 라벨)가 그대로 정답지가 된다.

사용법 (맥):
  python scripts/eval_retrieval.py
  → eval/results/retrieval_report.md
"""
from __future__ import annotations

from collections import Counter

from common import read_jsonl, resolve
from rag_search import Retriever

QA_FILES = ["data/synth/qa_verified.jsonl", "data/synth/qa_augmented_verified.jsonl"]
K_LIST = [1, 3, 5]


def main() -> None:
    retriever = Retriever()

    qa = []
    for f in QA_FILES:
        p = resolve(f)
        if p.exists():
            qa.extend(r for r in read_jsonl(p) if r.get("chunk_id"))
    print(f"라벨 있는 질문 {len(qa)}건으로 검색 평가")

    hits = {k: 0 for k in K_LIST}
    miss_by_chunk: Counter[str] = Counter()
    miss_examples: list[dict] = []

    def parent_of(cid: str) -> str:
        c = retriever.chunks.get(cid, {})
        return c.get("parent", cid)

    for r in qa:
        # 하위 조각이 검색되면 부모 조문으로 환산해서 채점 (라벨은 조문 단위)
        results = []
        for cid, _ in retriever.search(r["question"], max(K_LIST) * 2):
            p = parent_of(cid)
            if p not in results:
                results.append(p)
        results = results[:max(K_LIST)]
        for k in K_LIST:
            if r["chunk_id"] in results[:k]:
                hits[k] += 1
        if r["chunk_id"] not in results[:max(K_LIST)]:
            miss_by_chunk[r["chunk_id"]] += 1
            if len(miss_examples) < 10:
                miss_examples.append({"q": r["question"], "gold": r["chunk_id"], "got": results[:3]})

    n = len(qa)
    lines = [
        "# 검색(retrieval) 정확도 리포트",
        "",
        f"- 평가 질문: {n}건 (합성 Q&A의 chunk_id 라벨 사용)",
        "",
        "| 지표 | 값 |",
        "|---|---|",
    ] + [f"| hit@{k} | {hits[k]/n:.1%} ({hits[k]}/{n}) |" for k in K_LIST] + [
        "",
        "## top-5에도 정답이 없었던 질문의 정답 조문 분포",
        "",
    ]
    for cid, cnt in miss_by_chunk.most_common(10):
        lines.append(f"- {cid}: {cnt}건")
    if not miss_by_chunk:
        lines.append("- (없음 - 전 질문 top-5 적중)")
    lines += ["", "## 실패 예시 (최대 10건)", ""]
    for m in miss_examples:
        lines.append(f"- Q: {m['q']}  \n  정답: {m['gold']} / 검색됨: {', '.join(m['got'])}")

    out = resolve("eval/results/retrieval_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for k in K_LIST:
        print(f"  hit@{k}: {hits[k]/n:.1%}")
    print(f"리포트: {out}")
    print("해석 기준: hit@3가 90% 미만이면 생성 이전에 검색부터 고쳐야 한다 (RAG 평가 분리 원칙)")


if __name__ == "__main__":
    main()
