"""[4단계-③] 자기개선 루프 1/2: 생성 - 모델이 스스로 학습 재료 후보를 만든다.

STaR / rejection sampling의 소규모 재현:
  모델이 새 질문에 답을 여러 개 생성 → (다음 스크립트) 검증기가 걸러냄 → 합격만 학습 재료.

설계 원칙:
  - 질문 숫자는 학습 숫자와도 홀드아웃(시험지)과도 겹치지 않는다 → 시험지 오염 방지
  - temp 0.7로 질문당 3개 샘플 → 다양한 궤적 중 합격작을 건진다 (temp 0이면 3개가 똑같음)
  - 생성 시 사용한 검색 근거를 그대로 저장 → 합격작을 SFT 데이터로 만들 때 재사용

사용법 (맥, fused_run6 필요):
  python scripts/self_improve_generate.py
  재측정: python scripts/self_improve_generate.py --adapter adapters/run7 \
            --out data/self_improve/candidates_run7.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys

from common import resolve
from make_augmented_qa import yeonga_band
from build_cot_dataset import longserve
from rag_search import Retriever
from rag_chat import RAG_SYSTEM, build_prompt

TEMP = 0.7

# ---- 새 질문 세트: 학습 숫자ㆍ홀드아웃 숫자 모두 회피 ----
QUESTIONS: list[dict] = []


def add(kind: str, q: str, hint: str, **truth) -> None:
    QUESTIONS.append({"kind": kind, "q": q, "hint": hint, "truth": truth})


# 연가 구간 판정 (학습: 0y7m,1y3m,... / 홀드아웃: 4y7m,5y2m 과 겹치지 않는 숫자)
for y, m in [(0, 3), (1, 9), (2, 8), (3, 10), (4, 11), (5, 9), (6, 6), (7, 2), (9, 0), (11, 0)]:
    band, days = yeonga_band(y + m / 12)
    period = f"만 {y}년 {m}개월" if m else f"만 {y}년"
    add("yeonga", f"재직기간이 {period}인데 연가가 며칠인가요?", "연가 일수 재직기간",
        band=band, days=days)

# 병가 진단서 문턱 (학습: 2,3,5,7,10 / 홀드아웃: 4) - 6은 경계값(초과 아님)
for req in [1, 6, 8, 9, 12]:
    add("byeongga", f"병가를 {req}일 쓰려고 하는데 진단서를 내야 하나요?", "병가 진단서",
        req=req, need=req > 6)

# 지각 누계 나눗셈 (학습: 8,12,16,24 / 홀드아웃: 20)
for hours in [10, 18, 28, 32, 40]:
    d, rem = divmod(hours, 8)
    add("jigak", f"개인 사유 지각과 조퇴가 누계 {hours}시간이면 연가에서 며칠 깎이나요?",
        "지각 조퇴 연가", hours=hours, d=d, rem=rem)

# 장기재직 구간 (학습: 5,8,15,20,25 / 홀드아웃: 12) - 3은 대상 미달 경계
for y in [3, 7, 11, 18, 23, 30]:
    band, days = longserve(y)
    add("longserve", f"만 {y}년 재직했는데 장기재직휴가는 며칠인가요?", "장기재직휴가",
        years=y, band=band, days=days)

# N년차 중의성 경우 구분 - 두 해석의 일수가 달라지는 년차만
for n in [3, 6]:
    b1, d1 = yeonga_band(n - 1)   # 만 (n-1)년 해석
    b2, d2 = yeonga_band(n)       # 만 n년 해석
    add("casesplit", f"저 올해로 {n}년차인데 연가 며칠이에요?", "연가 일수 재직기간",
        d1=d1, d2=d2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fused_run6", help="베이스 모델 경로 (기본 fused_run6)")
    ap.add_argument("--adapter", default=None, help="어댑터 경로 (예: adapters/run7)")
    ap.add_argument("--samples", type=int, default=3, help="질문당 샘플 수 (기본 3)")
    ap.add_argument("--out", default="data/self_improve/candidates.jsonl")
    args = ap.parse_args()

    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        sys.exit("[오류] mlx-lm이 필요합니다 (맥에서 실행).")

    if not resolve(args.model).exists():
        sys.exit(f"[오류] {args.model}이 없습니다.")
    adapter = str(resolve(args.adapter)) if args.adapter else None

    retriever = Retriever()
    print(f"모델 로드: {args.model}" + (f" + {adapter}" if adapter else ""))
    model, tokenizer = load(str(resolve(args.model)), adapter_path=adapter)
    sampler = make_sampler(temp=TEMP, top_p=0.95)

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(QUESTIONS) * args.samples
    done = 0
    with open(out, "w", encoding="utf-8") as f:
        for item in QUESTIONS:
            hits = retriever.search(f"{item['q']} {item['hint']}", 3)
            contexts = [retriever.get_chunk(cid) for cid, _ in hits]
            messages = [
                {"role": "system", "content": RAG_SYSTEM},
                {"role": "user", "content": build_prompt(item["q"], contexts)},
            ]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            for s in range(args.samples):
                text = generate(model, tokenizer, prompt=prompt, max_tokens=512, sampler=sampler)
                f.write(json.dumps({**item, "sample_no": s, "contexts": contexts,
                                    "answer": text.strip()}, ensure_ascii=False) + "\n")
                done += 1
                print(f"  [{done}/{n_total}] {item['kind']} - {item['q'][:24]}...")

    print(f"\n후보 {n_total}건 → {out}")
    print("다음 단계: python scripts/self_improve_filter.py --candidates " + str(args.out)
          + (" --tag run7" if args.adapter else ""))


if __name__ == "__main__":
    main()
