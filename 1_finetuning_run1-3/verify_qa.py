"""[합성 3/3] 생성된 Q&A의 근거(grounding) 검수.

모델 붕괴(이론 노트 Q2)를 막는 핵심 장치. 생성된 Q&A가
근거 조각(chunk)에 실제로 붙어 있는지 기계적으로 검사한다.

입력:
  - data/synth/chunks.jsonl          (parse_doc.py 산출물)
  - Q&A 파일: 한 줄에 {"chunk_id", "question", "answer"}

검사 항목:
  1. chunk_id 존재: 근거 없는 Q&A는 즉시 탈락
  2. 숫자 충실성: 답변 속 모든 숫자(일수·시간·비율 등)가 근거 조각에
     실제로 존재해야 함 → 규정 데이터에서 환각이 가장 잘 생기는 지점
  3. 내용어 겹침: 답변의 한국어 내용어 중 일정 비율 이상이 근거 조각에
     등장해야 함 (기본 40%) → 조각과 무관한 창작 답변 차단
  4. 기본 품질: 길이, 질문 중복

통과분은 {"question","answer"} 포맷으로 저장되어
기존 파이프라인(validate_data.py → prepare_data.py)에 그대로 들어간다.

사용법:
  python scripts/verify_qa.py --qa data/synth/qa_generated.jsonl
  → data/synth/qa_verified.jsonl + 검수 리포트
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter

from common import read_jsonl, resolve, write_jsonl

RE_NUM = re.compile(r"\d+")
RE_KO_WORD = re.compile(r"[가-힣]{2,}")

# 조각에 없어도 허용하는 일반 표현 (내용어 겹침 계산에서 제외)
STOPWORDS = {
    "있습니다", "합니다", "됩니다", "입니다", "경우", "해당", "관련", "규정",
    "따라", "위해", "대한", "또는", "그리고", "하지만", "하는", "있는", "없는",
    "수행", "가능", "필요", "때문", "다음", "이상", "이하", "이내", "기준",
}


def norm_digits(text: str) -> set[str]:
    return set(RE_NUM.findall(text))


def content_words(text: str) -> set[str]:
    return {w for w in RE_KO_WORD.findall(text) if w not in STOPWORDS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/synth/chunks.jsonl")
    ap.add_argument("--qa", required=True, help="생성된 Q&A JSONL (chunk_id/question/answer)")
    ap.add_argument("--output", default="data/synth/qa_verified.jsonl")
    ap.add_argument("--report", default="data/synth/synth_report.md")
    ap.add_argument("--min-overlap", type=float, default=0.4, help="내용어 겹침 최소 비율")
    args = ap.parse_args()

    chunks = {c["chunk_id"]: c for c in read_jsonl(resolve(args.chunks))}
    qa_rows = read_jsonl(resolve(args.qa))
    if not chunks or not qa_rows:
        sys.exit("[오류] chunks 또는 qa 파일이 비어 있습니다.")

    passed, rejected = [], []
    reasons = Counter()
    seen_q = set()

    for row in qa_rows:
        q, a, cid = row.get("question", "").strip(), row.get("answer", "").strip(), row.get("chunk_id", "")

        def reject(reason: str):
            reasons[reason] += 1
            rejected.append({**row, "reject_reason": reason})

        chunk = chunks.get(cid)
        if chunk is None:
            reject("근거 조각 없음 (chunk_id 불일치)")
            continue
        if len(q) < 8 or len(a) < 15:
            reject("질문/답변이 너무 짧음")
            continue
        if q in seen_q:
            reject("질문 중복")
            continue

        src = chunk["text"]
        # 2) 숫자 충실성
        missing_nums = norm_digits(a) - norm_digits(src) - norm_digits(chunk.get("article", ""))
        if missing_nums:
            reject(f"근거에 없는 숫자 포함: {sorted(missing_nums)[:5]}")
            continue
        # 3) 내용어 겹침
        words = content_words(a)
        if words:
            overlap = len({w for w in words if w in src}) / len(words)
            if overlap < args.min_overlap:
                reject(f"근거 겹침 부족 ({overlap:.0%})")
                continue

        seen_q.add(q)
        passed.append({"question": q, "answer": a, "chunk_id": cid})

    out = resolve(args.output)
    write_jsonl(out, passed)
    rej_path = resolve("data/synth/qa_rejected.jsonl")
    write_jsonl(rej_path, rejected)

    total = len(qa_rows)
    lines = [
        "# 합성 Q&A 검수 리포트",
        "",
        f"- 생성: **{total}건** / 통과: **{len(passed)}건** ({len(passed)/total:.0%}) / 탈락: **{len(rejected)}건**",
        f"- 근거 조각 수: {len(chunks)} / 조각당 평균 통과 Q&A: {len(passed)/max(len(chunks),1):.1f}건",
        "",
        "## 탈락 사유",
        "",
    ] + [f"- {r}: {c}건" for r, c in reasons.most_common()]
    lines += ["", f"탈락 상세: {rej_path}"]
    report = resolve(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"통과 {len(passed)}/{total}건 → {out}")
    for r, c in reasons.most_common():
        print(f"  - {r}: {c}건")
    print(f"리포트: {report}")
    print("다음 단계: 통과분을 data/raw/dataset.jsonl 로 병합 후 validate_data.py 진행")


if __name__ == "__main__":
    main()
