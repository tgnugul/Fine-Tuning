"""2단계: Pydantic 기반 데이터 검증 및 필터링.

data/raw/dataset.jsonl 을 읽어 스키마·품질 검증을 통과한 데이터만
data/validated/dataset.jsonl 로 저장하고, 어떤 데이터가 왜 탈락했는지
validation_report.md 리포트를 생성합니다.

핵심 검증 항목 (configs/config.yaml 의 validation 섹션에서 조정):
  1. 스키마: question/answer 가 문자열이고 비어있지 않은가
  2. 길이: 너무 짧거나 긴 질문/답변 제거
  3. 언어 오염: 한글 비율이 낮거나 한자·가나가 섞인 데이터 제거
     → 지난 학습에서 중국어가 섞여 나온 문제의 재발 방지 핵심
  4. 중복: 동일 질문 중복 제거 (과적합·암기 방지)
  5. 저품질: URL만 있는 답변 제거

사용법:
  python scripts/validate_data.py
"""
from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter

from pydantic import BaseModel, ValidationError, field_validator

from common import load_config, read_jsonl, resolve, write_jsonl

CFG = load_config()
V = CFG["validation"]

# ---------- 문자 분류 유틸 ----------

RE_URL = re.compile(r"https?://\S+")
RE_WS = re.compile(r"\s+")

HANGUL_RANGES = (
    (0xAC00, 0xD7A3),  # 한글 음절
    (0x1100, 0x11FF),  # 자모
    (0x3130, 0x318F),  # 호환 자모
)
CJK_OTHER_RANGES = (
    (0x4E00, 0x9FFF),  # CJK 한자
    (0x3400, 0x4DBF),  # 한자 확장 A
    (0xF900, 0xFAFF),  # 호환 한자
    (0x3040, 0x30FF),  # 히라가나 + 가타카나
)


def _in_ranges(ch: str, ranges) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ranges)


def char_stats(text: str) -> tuple[float, float]:
    """(한글 비율, 한글 외 CJK 비율) — 공백 제외 문자 기준."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0, 0.0
    hangul = sum(1 for c in chars if _in_ranges(c, HANGUL_RANGES))
    cjk_other = sum(1 for c in chars if _in_ranges(c, CJK_OTHER_RANGES))
    return hangul / len(chars), cjk_other / len(chars)


# ---------- Pydantic 스키마 ----------

class QAPair(BaseModel):
    """학습 데이터 한 건의 스키마. 위반 시 ValidationError로 탈락."""

    question: str
    answer: str

    @field_validator("question", "answer", mode="before")
    @classmethod
    def must_be_str(cls, v):
        if not isinstance(v, str):
            raise ValueError(f"문자열이 아님 (타입: {type(v).__name__})")
        return v

    @field_validator("question", "answer")
    @classmethod
    def normalize(cls, v: str) -> str:
        # 유니코드 정규화(NFC) + 공백 정리
        v = unicodedata.normalize("NFC", v)
        v = RE_WS.sub(" ", v).strip()
        if not v:
            raise ValueError("빈 값")
        return v

    @field_validator("question")
    @classmethod
    def question_length(cls, v: str) -> str:
        if len(v) < V["min_question_chars"]:
            raise ValueError(f"질문이 너무 짧음 ({len(v)}자)")
        if len(v) > V["max_question_chars"]:
            raise ValueError(f"질문이 너무 김 ({len(v)}자)")
        return v

    @field_validator("answer")
    @classmethod
    def answer_length(cls, v: str) -> str:
        if len(v) < V["min_answer_chars"]:
            raise ValueError(f"답변이 너무 짧음 ({len(v)}자)")
        if len(v) > V["max_answer_chars"]:
            raise ValueError(f"답변이 너무 김 ({len(v)}자)")
        return v

    @field_validator("answer")
    @classmethod
    def not_url_only(cls, v: str) -> str:
        if V.get("drop_url_only_answers", True):
            stripped = RE_URL.sub("", v).strip(" .,:;-")
            if len(stripped) < 5:
                raise ValueError("답변이 URL 뿐임")
        return v

    @field_validator("answer")
    @classmethod
    def language_check(cls, v: str) -> str:
        hangul_ratio, cjk_other_ratio = char_stats(v)
        if hangul_ratio < V["min_korean_ratio"]:
            raise ValueError(f"한글 비율 미달 ({hangul_ratio:.0%})")
        if cjk_other_ratio > V["max_cjk_other_ratio"]:
            raise ValueError(f"한자/가나 비율 초과 ({cjk_other_ratio:.0%}) - 언어 오염 의심")
        return v


# ---------- 메인 ----------

def main() -> None:
    raw_path = resolve(CFG["data"]["raw_file"])
    if not raw_path.exists():
        sys.exit(f"[오류] 원본 데이터가 없습니다: {raw_path}\n먼저 python scripts/download_data.py 를 실행하세요.")

    rows = read_jsonl(raw_path)
    print(f"원본 데이터: {len(rows):,}건")

    passed: list[dict] = []
    reject_reasons: Counter[str] = Counter()
    reject_samples: dict[str, list[str]] = {}
    seen_questions: set[str] = set()

    for row in rows:
        try:
            qa = QAPair(**{k: row.get(k) for k in ("question", "answer")})
        except ValidationError as e:
            reason = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())
            reject_reasons[reason] += 1
            if len(reject_samples.setdefault(reason, [])) < 3:
                snippet = str(row.get("question", ""))[:60] or str(row.get("answer", ""))[:60]
                reject_samples[reason].append(snippet)
            continue

        if V.get("drop_duplicates", True):
            key = qa.question
            if key in seen_questions:
                reject_reasons["중복 질문"] += 1
                if len(reject_samples.setdefault("중복 질문", [])) < 3:
                    reject_samples["중복 질문"].append(qa.question[:60])
                continue
            seen_questions.add(key)

        out_row = {"question": qa.question, "answer": qa.answer}
        if "source" in row:          # 혼합 데이터의 출처 태그 보존 (통계용)
            out_row["source"] = row["source"]
        passed.append(out_row)

    out_path = resolve(CFG["data"]["validated_file"])
    write_jsonl(out_path, passed)

    # ---------- 리포트 ----------
    total, ok = len(rows), len(passed)
    lines = [
        "# 데이터 검증 리포트",
        "",
        f"- 원본: **{total:,}건**",
        f"- 통과: **{ok:,}건** ({ok / total:.1%})" if total else "- 통과: 0건",
        f"- 탈락: **{total - ok:,}건**",
        "",
        "## 탈락 사유별 집계",
        "",
        "| 사유 | 건수 | 예시 (앞 60자) |",
        "|---|---|---|",
    ]
    for reason, count in reject_reasons.most_common():
        examples = " / ".join(reject_samples.get(reason, []))
        lines.append(f"| {reason} | {count:,} | {examples} |")
    if not reject_reasons:
        lines.append("| (탈락 없음) | 0 | |")

    report_path = resolve(CFG["data"]["report_file"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"통과: {ok:,}건 → {out_path}")
    print(f"탈락: {total - ok:,}건 (사유별 상세: {report_path})")
    for reason, count in reject_reasons.most_common():
        print(f"  - {reason}: {count:,}건")
    print("다음 단계: python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
