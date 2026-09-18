"""[4단계-②] DPO 선호 쌍 생성기 - "이렇게 하지 마라"를 가르치는 데이터.

SFT는 좋은 예시만 보여주지만, DPO는 (선호, 비선호) 쌍으로
"무엇이 더 나은가"의 방향을 가르친다.

우리의 재료: 비선호(rejected)는 지어낸 것이 아니라 **실제 관측된 실패 유형의 재현**이다.
  r1. 풀이 생략      - run5가 일부 문항에서 보인 습관
  r2. 한 칸 밀림     - run4b가 보인 인접 구간 오답
  r3. 자기모순       - 원본+RAG가 보인 "답변 + 거절 문구 이어붙이기"
  r4. 잡탕 인용      - 다른 조문 번호를 인용

주의: mlx-tune의 DPOTrainer는 prompt+응답을 그냥 이어붙일 뿐 챗 템플릿을
적용하지 않는다 (소스 확인). 따라서 여기서 챗 템플릿을 미리 렌더링해
prompt 문자열에 넣는다 - 학습 형식 = 추론 형식.

사용법 (맥, 임베딩 인덱스 필요):
  python scripts/build_dpo_pairs.py
  → data/dpo/pairs.jsonl
"""
from __future__ import annotations

import json
import random

from common import load_config, resolve
from make_augmented_qa import yeonga_band
from build_rag_dataset import get_retriever
from build_cot_dataset import longserve
from rag_chat import RAG_SYSTEM, build_prompt

PAIRS: list[dict] = []


def cot_yeonga(period: str, band: str, days: int) -> str:
    return (
        f"[풀이]\n1. 질문의 재직기간은 {period}이다.\n"
        f"2. 근거 제15조의 재직기간별 연가 일수 표에서 {period}은 '{band}' 구간에 해당한다.\n"
        f"3. 해당 구간의 연가 일수는 {days}일이다.\n"
        f"[답변] 복무규정 제15조에 따르면 재직기간 {band}인 공무원의 연가 일수는 {days}일입니다."
    )


def add(question: str, chosen: str, rejected: str, hint: str) -> None:
    PAIRS.append({"q": question, "chosen": chosen, "rejected": rejected, "hint": hint})


# ---- 연가: r1(풀이 생략) / r2(한 칸 밀림) / r3(자기모순) / r4(잡탕 인용) 순환 ----
YEARS = [(0, 7), (1, 3), (2, 0), (3, 2), (4, 0), (5, 6), (6, 1), (8, 0), (10, 0)]
for i, (y, m) in enumerate(YEARS):
    yrs = y + m / 12
    band, days = yeonga_band(yrs)
    wrong_band, wrong_days = yeonga_band(max(0.5, yrs - 1)) if yeonga_band(max(0.5, yrs - 1))[1] != days else yeonga_band(yrs + 1)
    period = f"만 {y}년 {m}개월" if m else f"만 {y}년"
    q = f"재직기간이 {period}인데 연가가 며칠인가요?"
    chosen = cot_yeonga(period, band, days)
    kind = i % 4
    if kind == 0:      # r1 풀이 생략
        rejected = f"{days}일입니다."
    elif kind == 1:    # r2 한 칸 밀림
        rejected = cot_yeonga(period, wrong_band, wrong_days)
    elif kind == 2:    # r3 자기모순
        rejected = (f"복무규정 제15조에 따르면 재직기간 {band}인 공무원의 연가 일수는 {days}일입니다. "
                    f"제공된 규정에서 확인되지 않습니다.")
    else:              # r4 잡탕 인용
        rejected = chosen.replace("제15조", "제16조")
    add(q, chosen, rejected, "연가 일수 재직기간")

# ---- 병가 진단서: 판정 뒤집기 / 풀이 생략 ----
for i, req in enumerate([2, 3, 5, 7, 10]):
    need = req > 6
    chosen = (
        f"[풀이]\n1. 신청하려는 병가는 {req}일이고, 올해 사용한 병가는 없다 → 연간 누계 {req}일.\n"
        f"2. 근거 제18조: 병가 일수가 연간 6일을 초과하는 경우 의사의 진단서를 첨부하여야 한다.\n"
        f"3. {req}일은 6일을 {'초과하므로 진단서가 필요하다' if need else '초과하지 않으므로 진단서가 필요 없다'}.\n"
        f"[답변] 복무규정 제18조에 따라 연간 누계 {req}일의 병가는 진단서가 {'필요합니다' if need else '필요하지 않습니다'}."
    )
    if i % 2 == 0:     # 판정 뒤집기 (문턱 비교 오류)
        rejected = (f"복무규정 제18조에 따라 연간 누계 {req}일의 병가는 진단서가 "
                    f"{'필요하지 않습니다' if need else '필요합니다'}.")
    else:              # 풀이 생략
        rejected = f"진단서가 {'필요합니다' if need else '필요하지 않습니다'}."
    add(f"병가를 {req}일 쓰려고 하는데 진단서를 내야 하나요?", chosen, rejected, "병가 진단서")

# ---- 지각 누계: 나눗셈 오류 / 풀이 생략 ----
for i, hours in enumerate([8, 12, 16, 24]):
    d, rem = divmod(hours, 8)
    rem_txt = f" 나머지 {rem}시간은 아직 연가 1일에 미달한다." if rem else ""
    chosen = (
        f"[풀이]\n1. 질병ㆍ부상 외 사유의 지각ㆍ조퇴ㆍ외출 누계가 {hours}시간이다.\n"
        f"2. 근거 제17조: 누계 8시간을 연가 1일로 계산한다.\n"
        f"3. {hours}시간은 8시간 × {d}회이므로 연가 {d}일에 해당한다.{rem_txt}\n"
        f"[답변] 복무규정 제17조에 따라 누계 {hours}시간은 연가 {d}일로 계산되어 연가 일수에서 공제됩니다."
    )
    if i % 2 == 0:
        rejected = chosen.replace(f"연가 {d}일", f"연가 {d + 1}일")   # 계산 오류
    else:
        rejected = f"연가 {d}일이 공제됩니다."
    add(f"개인 사유 지각과 조퇴가 누계 {hours}시간이면 연가에서 며칠 깎이나요?", chosen, rejected, "지각 조퇴 연가")

# ---- 장기재직: 한 칸 밀림 / 자기모순 ----
for i, y in enumerate([5, 8, 15, 20, 25]):
    band, days = longserve(y)
    wb, wd = longserve(y - 4) if longserve(y - 4)[1] not in (None, days) else longserve(y + 6)
    chosen = (
        f"[풀이]\n1. 질문의 재직기간은 만 {y}년이다.\n"
        f"2. 근거 제20조의 장기재직휴가 구분에서 만 {y}년은 '{band}'에 해당한다.\n"
        f"3. 해당 구간의 장기재직휴가는 {days}일이다.\n"
        f"[답변] 복무규정 제20조에 따라 재직기간 {band}인 공무원의 장기재직휴가는 {days}일입니다."
    )
    if i % 2 == 0 and wb:
        rejected = chosen.replace(band, wb).replace(f"{days}일", f"{wd}일")
    else:
        rejected = (f"복무규정 제20조에 따라 장기재직휴가는 {days}일입니다. "
                    f"제공된 규정에서 확인되지 않습니다.")
    add(f"만 {y}년 재직했는데 장기재직휴가는 며칠인가요?", chosen, rejected, "장기재직휴가")


def main() -> None:
    cfg = load_config()
    retriever, mode = get_retriever()
    if mode == "mock":
        print("[경고] 모의 검색 - 실제 쌍은 맥에서 인덱스 구축 후 빌드할 것")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    eos = tok.eos_token or ""

    rows = []
    for p in PAIRS:
        hits = retriever.search(f"{p['q']} {p['hint']}", 3)
        contexts = [retriever.get_chunk(cid) for cid, _ in hits]
        messages = [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": build_prompt(p["q"], contexts)},
        ]
        # 챗 템플릿을 미리 렌더링 (mlx-tune은 raw 이어붙이기만 하므로)
        prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        rows.append({"prompt": prompt_str, "chosen": p["chosen"] + eos, "rejected": p["rejected"] + eos})

    random.Random(cfg["prepare"]["seed"]).shuffle(rows)
    out = resolve("data/dpo/pairs.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"선호 쌍 {len(rows)}건 → {out}")
    print("다음 단계: python scripts/train_dpo.py")


if __name__ == "__main__":
    main()
