"""[run3] 사실(fact) 단위 데이터 증강 생성기.

run2의 진단: 사실 하나당 1~2개 표현으로는 지식이 형성되지 않고 표면 암기만 된다.
처방: 핵심 사실마다 질문 표현을 프로그램으로 대량 전개한다.

설계 원칙:
  1. 사실을 구조화된 데이터(표, 규칙)로 정의하고 질문 템플릿과 곱한다
  2. 질문에는 다양한 사용자 상황(재직 7년차 등)을 넣되,
     답변은 규정의 구간+수치만 말한다 → 근거 검수(숫자 대조) 통과 보장
  3. run2가 실패한 지점(학습 표현과 다른 질문)을 정면 공략

사용법:
  python scripts/make_augmented_qa.py
  → data/synth/qa_augmented.jsonl (chunk_id/question/answer)

주의(아카이브 메모): 연가(제15조) 파트는 맥 원본 그대로이고,
그 아래 병가·지각·장기재직 파트는 동일 패턴으로 재구성한 것이다.
정본은 맥 ~/qwen-domain-ft/scripts/make_augmented_qa.py.
"""
from __future__ import annotations

from common import ROOT, write_jsonl

QA: list[dict] = []


def add(chunk: str, question: str, answer: str) -> None:
    QA.append({"chunk_id": f"복무규정#{chunk}", "question": question, "answer": answer})


# =========================================================
# 1. 연가 일수 표 (제15조) - 연차 × 질문 표현 전개
# =========================================================
def yeonga_band(years: float) -> tuple[str, int]:
    if years < 1:
        return "1개월 이상 1년 미만", 11
    if years < 3:
        return "1년 이상 3년 미만", 15
    if years < 4:
        return "3년 이상 4년 미만", 16
    if years < 5:
        return "4년 이상 5년 미만", 17
    if years < 6:
        return "5년 이상 6년 미만", 20
    return "6년 이상", 21


YEAR_QUESTIONS = [
    "재직 {y}년차인데 연가가 며칠인가요?",
    "공무원 된 지 {y}년째입니다. 올해 연가 일수가 궁금해요.",
    "{y}년 일한 공무원은 연차를 며칠 받나요?",
    "입사한 지 {y}년 됐는데 제 연가는 총 며칠인가요?",
    "올해로 {y}년차 공무원인데 쓸 수 있는 연가가 얼마나 되나요?",
    "재직기간 {y}년이면 연가가 며칠 나오는지 알려줘.",
]

for y in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
    band, days = yeonga_band(y)
    ans = f"복무규정 제15조에 따르면 재직기간 {band}인 공무원의 연가 일수는 {days}일입니다."
    for i, tpl in enumerate(YEAR_QUESTIONS):
        # 연차마다 템플릿 절반씩 사용 (조합 폭발 방지 + 표현 다양성 유지)
        if (y + i) % 2 == 0:
            add("제15조", tpl.format(y=y), ans)

# 신규자(1년 미만) 별도
for q in [
    "이번에 임용된 신규 공무원인데 연가가 있나요?",
    "들어온 지 6개월밖에 안 됐는데 연차를 쓸 수 있나요?",
    "1년 미만 재직자도 연가를 받을 수 있는지 궁금합니다.",
]:
    add("제15조", q, "네. 복무규정 제15조의 재직기간별 연가 일수에 따르면 재직기간 1개월 이상 1년 미만인 공무원의 연가 일수는 11일입니다.")

# =========================================================
# 2. 병가 진단서 (제18조) - 연간 누계 6일 초과 시 진단서
#    (재구성 파트: 일수 × 표현으로 전개)
# =========================================================
BYEONGGA_QUESTIONS = [
    "병가를 {d}일 쓰려고 하는데 진단서를 내야 하나요?",
    "몸이 아파서 {d}일 쉬려는데 진단서가 필요한가요?",
    "연간 병가 {d}일이면 진단서 첨부 대상인가요?",
]
for d in [2, 3, 5, 6, 7, 8, 10]:
    need = d > 6
    ans = (f"복무규정 제18조에 따라 연간 누계 {d}일의 병가는 진단서가 "
           f"{'필요합니다' if need else '필요하지 않습니다'}.")
    for i, tpl in enumerate(BYEONGGA_QUESTIONS):
        if (d + i) % 2 == 0:
            add("제18조", tpl.format(d=d), ans)

# =========================================================
# 3. 지각·조퇴·외출 누계 (제17조) - 누계 8시간 = 연가 1일
#    (재구성 파트)
# =========================================================
JIGAK_QUESTIONS = [
    "개인 사유 지각과 조퇴가 누계 {h}시간이면 연가에서 며칠 깎이나요?",
    "외출 누계가 {h}시간인데 연가 공제가 어떻게 되나요?",
]
for h in [8, 16, 24, 32]:
    d = h // 8
    ans = f"복무규정 제17조에 따라 누계 {h}시간은 연가 {d}일로 계산되어 연가 일수에서 공제됩니다."
    for tpl in JIGAK_QUESTIONS:
        add("제17조", tpl.format(h=h), ans)


if __name__ == "__main__":
    out = ROOT / "data" / "synth" / "qa_augmented.jsonl"
    write_jsonl(out, QA)
    print(f"증강 Q&A {len(QA)}건 생성 → {out}")
    print("다음 단계: python scripts/verify_qa.py --qa data/synth/qa_augmented.jsonl "
          "--output data/synth/qa_augmented_verified.jsonl")
