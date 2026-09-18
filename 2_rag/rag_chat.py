"""[RAG] 검색 증강 답변 - 모델이 외운 기억 대신 눈앞의 원문을 읽고 답한다.

어댑터ㆍ모델 선택:
  --base                          원본 모델 + RAG
  (기본값)                         config의 training.adapter_dir + RAG
  --adapter <경로>                 지정 어댑터 (예: adapters/run5)
  --model <경로> --adapter <경로>  융합 모델 위의 어댑터 (예: fused_run5 + run6_dpo)

사용법 (맥):
  python scripts/rag_chat.py -q "질문" --show-context
  python scripts/rag_chat.py --model fused_run5 --adapter adapters/run6_dpo -q "질문"
"""
from __future__ import annotations

import argparse
import sys

from common import load_config, resolve
from rag_search import Retriever

RAG_SYSTEM = """당신은 조직 내부 규정에 답하는 한국어 업무 어시스턴트입니다.
반드시 아래 [근거] 조문의 내용만 사용하여 답변하세요.
근거에 없는 내용은 지어내지 말고, "제공된 규정에서 확인되지 않습니다"라고 답하세요.
답변에는 근거가 된 조문 번호를 표기하세요."""


def build_prompt(question: str, contexts: list[dict]) -> str:
    blocks = []
    for c in contexts:
        head = f"{c['article']}({c['title']})" if c.get("article") else c["chunk_id"]
        blocks.append(f"--- {head} ---\n{c['text']}")
    return "[근거]\n" + "\n\n".join(blocks) + f"\n\n[질문]\n{question}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question", help="단일 질문 (생략 시 대화 모드)")
    ap.add_argument("--base", action="store_true", help="어댑터 없이 원본 모델 사용")
    ap.add_argument("--adapter", help="사용할 어댑터 경로")
    ap.add_argument("--model", help="베이스 모델 경로/이름 재지정 (예: fused_run5)")
    ap.add_argument("--show-context", action="store_true", help="검색된 근거 조문 출력")
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    retriever = Retriever()

    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        sys.exit("[오류] mlx-lm이 필요합니다 (맥에서 실행).")

    model_name = args.model or cfg["model"]["name"]
    if args.base:
        adapter = None
    elif args.adapter:
        adapter = str(resolve(args.adapter))
    else:
        adapter = str(resolve(cfg["training"]["adapter_dir"]))
    print(f"모델 로드: {model_name}" + (" (원본)" if adapter is None else f" + {adapter}"))
    model, tokenizer = load(model_name, adapter_path=adapter)
    sampler = make_sampler(temp=0.0)

    def answer(question: str) -> None:
        hits = retriever.search(question, args.top_k)
        contexts = [retriever.get_chunk(cid) for cid, _ in hits]
        if args.show_context:
            for (cid, score), c in zip(hits, contexts):
                print(f"  [검색] {cid} (유사도 {score:.3f}) {c['title']}")
        messages = [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": build_prompt(question, contexts)},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        text = generate(model, tokenizer, prompt=prompt, max_tokens=512, sampler=sampler)
        print(text.strip())

    if args.question:
        answer(args.question)
    else:
        print("대화 모드 (빈 입력으로 종료)")
        while True:
            q = input("\n>> ").strip()
            if not q:
                break
            answer(q)


if __name__ == "__main__":
    main()
