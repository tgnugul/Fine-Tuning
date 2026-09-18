"""[4단계] CoT(풀이 과정) 데이터 생성기 - "표를 읽고 구간을 도출하는 사고" 증류.

run4b의 진단: SFT 시연 찍어내기는 행동은 심지만 추론은 못 심는다.
처방: 풀이 과정을 잘게 쪼갠 CoT 시연을 학습시킨다. 각 걸음은 쉬워서
작은 모델도 밟을 수 있고, 절차 자체가 학습 대상이 된다.

설계 원칙:
  1. 풀이의 모든 수치는 코드가 계산해서 채운다 → 풀이 정확성 100% 보장
     (모델 붕괴 방지의 '외부 검증기' 원칙을 생성 단계에 적용)
  2. 숫자 홀드아웃: 학습에 쓰는 숫자 조합과 평가용 숫자 조합을 분리
     → "외운 것인지, 절차를 배운 것인지"를 측정 가능하게
  3. 형식은 3단계와 동일한 RAG 형식 (학습 형식 = 추론 형식)

대상 절차 5종:
  연가 표(만 N년 / N년차 중의성), 병가 진단서 판정(연 6일 문턱),
  지각ㆍ조퇴 누계(8시간=연가 1일), 장기재직휴가 구간, 유산ㆍ사산휴가 주수 구간

사용법 (맥, build_rag_dataset.py 실행 후):
  python scripts/build_cot_dataset.py
  → data/processed_rag/ 의 train/valid에 CoT 예시를 병합 (train_rag.sh 그대로 사용)
  → eval/testsets/cot_holdout.md 에 홀드아웃 평가 문항 생성
"""
from __future__ import annotations

import random

from common import load_config, read_jsonl, resolve, write_jsonl
from make_augmented_qa import yeonga_band
from build_rag_dataset import get_retriever, to_example

# =========================================================
# 절차(판정 함수)들 - 풀이의 수치는 전부 여기서 계산된다
# =========================================================

def longserve(years: float):
    if years < 5:
        return None, None
    if years < 10:
        return "5년 이상 10년 미만", 3
    if years < 20:
        return "10년 이상 20년 미만", 5
    return "20년 이상", 7


def miscarriage(weeks: int):
    if weeks <= 15:
        return "임신기간 15주 이내", 10
    if weeks <= 21:
        return "16주 이상 21주 이내", 30
    if weeks <= 27:
        return "22주 이상 27주 이내", 60
    return "28주 이상", 90


COT_EXAMPLES: list[dict] = []  # (question, cot_answer, search_hint)


def add(question: str, answer: str, hint: str) -> None:
    COT_EXAMPLES.append({"q": question, "a": answer, "hint": hint})


# ---- 1. 연가 표: 만 N년 M개월 (학습 값 - 홀드아웃: 4년7개월, 5년2개월) ----
YEONGA_TRAIN = [(0, 7), (1, 3), (2, 0), (3, 2), (4, 0), (5, 6), (6, 1), (8, 0), (10, 0)]
Y_TPL = [
    "재직기간이 만 {y}년 {m}개월인데 연가가 며칠인가요?",
    "만 {y}년 {m}개월 근무했습니다. 올해 연가 일수를 알려주세요.",
    "제 재직기간은 만 {y}년 {m}개월입니다. 연가는 얼마나 되나요?",
]
for y, m in YEONGA_TRAIN:
    band, days = yeonga_band(y + m / 12)
    period = f"만 {y}년 {m}개월" if m else f"만 {y}년"
    cot = (
        f"[풀이]\n"
        f"1. 질문의 재직기간은 {period}이다.\n"
        f"2. 근거 제15조의 재직기간별 연가 일수 표에서 {period}은 '{band}' 구간에 해당한다.\n"
        f"3. 해당 구간의 연가 일수는 {days}일이다.\n"
        f"[답변] 복무규정 제15조에 따르면 재직기간 {band}인 공무원의 연가 일수는 {days}일입니다."
    )
    for tpl in Y_TPL:
        add(tpl.format(y=y, m=m), cot, "연가 일수 재직기간")

# ---- 2. N년차 중의성 + 풀이 ----
for y in [1, 3, 4, 5, 6, 7]:
    lo_band, lo_days = yeonga_band(y - 0.5)
    hi_band, hi_days = yeonga_band(y)
    if lo_days == hi_days:
        concl = f"두 경우 모두 '{lo_band}' 안에 있어 연가 일수는 {lo_days}일로 같다."
        final = f"어느 해석이든 복무규정 제15조에 따라 연가 일수는 {lo_days}일입니다."
    else:
        concl = f"만 {y-1}년대이면 '{lo_band}' 구간으로 {lo_days}일, 만 {y}년 이상이면 '{hi_band}' 구간으로 {hi_days}일이다."
        final = (f"재직기간이 만 {y-1}년대이면 {lo_days}일, 만 {y}년 이상이면 {hi_days}일입니다. "
                 f"본인의 재직기간(만 기준)을 확인해 주세요.")
    cot = (
        f"[풀이]\n"
        f"1. '{y}년차'는 재직기간이 만 {y-1}년대일 수도, 만 {y}년을 넘었을 수도 있는 중의적 표현이다.\n"
        f"2. 근거 제15조의 표에서 두 경우를 각각 확인한다: {concl}\n"
        f"[답변] {final}"
    )
    for tpl in ["올해로 {y}년차인데 연가가 며칠인가요?", "{y}년차 공무원 연가 일수 알려주세요."]:
        add(tpl.format(y=y), cot, "연가 일수 재직기간")

# ---- 3. 병가 진단서 판정 (문턱 6일 / 홀드아웃: 4일, 누계 5+4일) ----
for req in [2, 3, 5, 7, 10]:
    need = req > 6
    cot = (
        f"[풀이]\n"
        f"1. 신청하려는 병가는 {req}일이고, 올해 사용한 병가는 없다 → 연간 누계 {req}일.\n"
        f"2. 근거 제18조: 병가 일수가 연간 6일을 초과하는 경우 의사의 진단서를 첨부하여야 한다.\n"
        f"3. {req}일은 6일을 {'초과하므로 진단서가 필요하다' if need else '초과하지 않으므로 진단서가 필요 없다'}.\n"
        f"[답변] 복무규정 제18조에 따라 연간 누계 {req}일의 병가는 진단서가 {'필요합니다' if need else '필요하지 않습니다'}."
    )
    add(f"병가를 {req}일 쓰려고 하는데 진단서를 내야 하나요?", cot, "병가 진단서")
    add(f"몸이 아파서 {req}일 쉬려고 합니다. 진단서가 필요한가요?", cot, "병가 진단서")
for used, req in [(3, 2), (4, 4), (5, 3), (2, 3)]:
    total = used + req
    need = total > 6
    cot = (
        f"[풀이]\n"
        f"1. 올해 이미 사용한 병가 {used}일 + 신청 {req}일 = 연간 누계 {total}일.\n"
        f"2. 근거 제18조: 연간 6일을 초과하는 병가는 의사의 진단서를 첨부하여야 한다.\n"
        f"3. 누계 {total}일은 6일을 {'초과하므로 진단서가 필요하다' if need else '초과하지 않으므로 진단서 없이 가능하다'}.\n"
        f"[답변] 복무규정 제18조에 따라 누계 {total}일이 되므로 진단서가 {'필요합니다' if need else '필요하지 않습니다'}."
    )
    add(f"올해 병가를 {used}일 썼는데 {req}일 더 쓰려면 진단서가 필요한가요?", cot, "병가 진단서")

# ---- 4. 지각ㆍ조퇴 누계 (8시간 = 연가 1일 / 홀드아웃: 20시간) ----
for hours in [8, 12, 16, 24]:
    d, rem = divmod(hours, 8)
    rem_txt = f" 나머지 {rem}시간은 아직 연가 1일에 미달한다." if rem else ""
    cot = (
        f"[풀이]\n"
        f"1. 질병ㆍ부상 외 사유의 지각ㆍ조퇴ㆍ외출 누계가 {hours}시간이다.\n"
        f"2. 근거 제17조: 누계 8시간을 연가 1일로 계산한다.\n"
        f"3. {hours}시간 ÷ 8시간 = {d}일.{rem_txt}\n"
        f"[답변] 복무규정 제17조에 따라 누계 {hours}시간은 연가 {d}일로 계산되어 연가 일수에서 공제됩니다."
    )
    add(f"개인 사유 지각과 조퇴가 누계 {hours}시간이면 연가에서 며칠 깎이나요?", cot, "지각 조퇴 연가")
    add(f"외출 누계가 {hours}시간인데 연가 공제가 어떻게 되나요?", cot, "지각 조퇴 연가")

# ---- 5. 장기재직휴가 (홀드아웃: 만 12년) ----
for y in [5, 8, 15, 20, 25]:
    band, days = longserve(y)
    cot = (
        f"[풀이]\n"
        f"1. 질문의 재직기간은 만 {y}년이다.\n"
        f"2. 근거 제20조의 장기재직휴가 구분에서 만 {y}년은 '{band}'에 해당한다.\n"
        f"3. 해당 구간의 장기재직휴가는 {days}일이다.\n"
        f"[답변] 복무규정 제20조에 따라 재직기간 {band}인 공무원의 장기재직휴가는 {days}일입니다."
    )
    add(f"만 {y}년 재직했는데 장기재직휴가는 며칠인가요?", cot, "장기재직휴가")

# ---- 6. 유산ㆍ사산휴가 주수 구간 (홀드아웃: 23주) ----
for w in [10, 18, 25, 30]:
    band, days = miscarriage(w)
    cot = (
        f"[풀이]\n"
        f"1. 임신기간은 {w}주이다.\n"
        f"2. 근거 제20조의 유산ㆍ사산휴가 구분에서 {w}주는 '{band}' 구간에 해당한다.\n"
        f"3. 해당 구간의 휴가는 유산하거나 사산한 날부터 {days}일까지이다.\n"
        f"[답변] 복무규정 제20조에 따라 임신 {w}주에 유산ㆍ사산한 경우 {days}일까지의 휴가를 사용할 수 있습니다."
    )
    add(f"임신 {w}주에 유산한 경우 휴가가 며칠인가요?", cot, "유산 사산 휴가")


# =========================================================
# 병합 및 저장
# =========================================================

HOLDOUT_MD = """# CoT 홀드아웃 평가 문항 (학습에 없는 숫자 조합)

run5 학습 후 아래를 rag_chat.py --adapter adapters/run5 로 시험한다.
채점 기준: [풀이] 단계가 나오는가 / 구간 판정이 맞는가 / 최종 수치가 맞는가.

| 문항 | 정답 |
|---|---|
| 재직기간이 만 4년 7개월인데 연가가 며칠인가요? | 4년 이상 5년 미만 → 17일 |
| 만 5년 2개월 근무했습니다. 연가 일수를 알려주세요. | 5년 이상 6년 미만 → 20일 |
| 병가를 4일 쓰려고 하는데 진단서를 내야 하나요? | 누계 4일 ≤ 6일 → 불필요 |
| 올해 병가를 5일 썼는데 4일 더 쓰려면 진단서가 필요한가요? | 누계 9일 > 6일 → 필요 |
| 개인 사유 지각 누계가 20시간이면 연가에서 며칠 깎이나요? | 20÷8 → 2일 공제 (4시간 잔여) |
| 만 12년 재직했는데 장기재직휴가는 며칠인가요? | 10년 이상 20년 미만 → 5일 |
| 임신 23주에 유산한 경우 휴가가 며칠인가요? | 22~27주 → 60일 |
"""


def main() -> None:
    cfg = load_config()
    rcfg = cfg["rag_training"]
    seed = cfg["prepare"]["seed"]
    retriever, mode = get_retriever()
    if mode == "mock":
        print("[경고] 모의 검색 - 실제 데이터는 맥에서 인덱스 구축 후 빌드할 것")

    out_dir = resolve(rcfg["output_dir"])
    base_train = read_jsonl(out_dir / "train.jsonl")
    base_valid = read_jsonl(out_dir / "valid.jsonl")
    print(f"기존 결합 데이터: train {len(base_train)} / valid {len(base_valid)}")

    cot_msgs = []
    for ex in COT_EXAMPLES:
        # 검색은 질문 + 힌트로 (숫자만 다른 질문들이 같은 조각을 안정적으로 가져오도록)
        hits = retriever.search(f"{ex['q']} {ex['hint']}", 3)
        contexts = [retriever.get_chunk(cid) for cid, _ in hits]
        cot_msgs.append(to_example(ex["q"], contexts, ex["a"], "cot"))

    rng = random.Random(seed)
    rng.shuffle(cot_msgs)
    n_val = max(2, len(cot_msgs) // 15)
    strip = lambda rows: [{"messages": r["messages"]} for r in rows]

    train = base_train + strip(cot_msgs[n_val:])
    valid = base_valid + strip(cot_msgs[:n_val])
    rng.shuffle(train)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)

    holdout_path = resolve("eval/testsets/cot_holdout.md")
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_path.write_text(HOLDOUT_MD, encoding="utf-8")

    print(f"CoT 예시 {len(cot_msgs)}건 생성ㆍ병합 → train {len(train)} / valid {len(valid)}")
    print(f"홀드아웃 평가 문항: {holdout_path}")
    print("다음 단계: config의 rag_training.adapter_dir을 adapters/run5 로 바꾸고 bash scripts/train_rag.sh")


if __name__ == "__main__":
    main()
