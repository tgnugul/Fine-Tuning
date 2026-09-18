"""[합성 1/3] 문서 → 조각(chunk) 분해.

공개 규정/법령/매뉴얼 문서를 Q&A 생성의 근거 단위로 쪼갠다.
각 조각이 "진리 앵커"가 된다 - 이후 생성되는 모든 Q&A는
자신의 근거 조각(chunk_id)을 달고 다니며, 검수 단계에서
답변이 근거 조각에 실제로 붙어 있는지 검사받는다.

지원 입력: .txt / .md  (PDF·HWP는 텍스트로 변환 후 사용)

파싱 전략:
  1. 법령 모드(자동 감지): "제N조(제목)" 패턴이 3개 이상이면
     조문 단위로 분해. "제N장" 장 제목도 함께 기록.
  2. 일반 모드(폴백): 빈 줄 기준 문단을 약 max_chars 크기로 묶음.

사용법:
  python scripts/parse_doc.py --input data/source_docs/복무규정.txt
  → data/synth/chunks.jsonl 생성
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from common import resolve, write_jsonl

# 조문 헤더: 제63조, 제24조의2 (제목) 등. 줄 시작(들여쓰기 허용) 기준.
RE_ARTICLE = re.compile(r"^[ \t]*제(\d+)조(의\d+)?\s*\(([^)]+)\)", re.MULTILINE)
RE_CHAPTER = re.compile(r"^[ \t]*제(\d+)장\s+(.+)$", re.MULTILINE)
RE_DELETED = re.compile(r"삭제\s*<[^>]*>")

# 웹 복사본 노이즈 제거 패턴 (법령정보센터 기준)
CLEAN_LINE_PATTERNS = [
    re.compile(r"^\s*조문체계도버튼.*$"),          # 내비게이션 찌꺼기
    re.compile(r"^\s*\[(전문개정|본조신설|제목개정)[^\]]*\]\s*$"),  # 개정 이력 줄
]
RE_INLINE_AMEND = re.compile(r"<(개정|신설)[^>]*>")   # 문장 속 <개정 2013...> 표기


def clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if any(p.match(line) for p in CLEAN_LINE_PATTERNS):
            continue
        lines.append(RE_INLINE_AMEND.sub("", line).rstrip())
    return "\n".join(lines)


def parse_law(text: str, source: str) -> list[dict]:
    """법령 모드: 조문 단위로 분해."""
    # 장 제목 위치 기록
    chapters: list[tuple[int, str]] = [
        (m.start(), f"제{m.group(1)}장 {m.group(2).strip()}") for m in RE_CHAPTER.finditer(text)
    ]

    def chapter_of(pos: int) -> str:
        cur = ""
        for start, title in chapters:
            if start <= pos:
                cur = title
            else:
                break
        return cur

    matches = list(RE_ARTICLE.finditer(text))
    chunks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        # 부칙 등이 붙으면 잘라냄
        body = re.split(r"^\s*부\s*칙", body, maxsplit=1, flags=re.MULTILINE)[0].strip()
        article_no = f"제{m.group(1)}조" + (m.group(2) or "")
        if RE_DELETED.search(body) and len(body) < 80:
            continue  # 삭제된 조문은 제외
        chunks.append({
            "chunk_id": f"{Path(source).stem}#{article_no}",
            "source": source,
            "chapter": chapter_of(m.start()),
            "article": article_no,
            "title": m.group(3).strip(),
            "text": body,
        })
    return chunks


def parse_generic(text: str, source: str, max_chars: int = 900) -> list[dict]:
    """일반 모드: 문단을 max_chars 근처로 묶어서 분해."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return [
        {
            "chunk_id": f"{Path(source).stem}#c{i:03d}",
            "source": source,
            "chapter": "",
            "article": "",
            "title": "",
            "text": c,
        }
        for i, c in enumerate(chunks, 1)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="원문 텍스트 파일 (.txt/.md)")
    ap.add_argument("--output", default="data/synth/chunks.jsonl")
    ap.add_argument("--max-chars", type=int, default=900, help="일반 모드 조각 크기")
    args = ap.parse_args()

    path = Path(args.input).expanduser()
    if not path.exists():
        sys.exit(f"[오류] 파일 없음: {path}")
    text = clean_text(path.read_text(encoding="utf-8"))

    # 본문 뒤에 붙는 부칙(개정 이력 모음) 전체를 절단 - 조문 번호가 재시작되어
    # 그대로 두면 중복 chunk_id와 잡동사니 조각이 대량 생성됨
    m = re.search(r"^\s*부\s*칙", text, re.MULTILINE)
    if m:
        text = text[:m.start()]
        print(f"[정제] 부칙 이후 절단 (본문 {len(text):,}자만 사용)")

    law_hits = len(RE_ARTICLE.findall(text))
    if law_hits >= 3:
        chunks = parse_law(text, path.name)
        mode = f"법령 모드 (조문 {law_hits}개 감지)"
    else:
        chunks = parse_generic(text, path.name, args.max_chars)
        mode = "일반 모드 (문단 묶음)"

    out = resolve(args.output)
    write_jsonl(out, chunks)

    lens = [len(c["text"]) for c in chunks]
    print(f"파싱 완료: {mode}")
    print(f"조각 수: {len(chunks)} / 길이: 평균 {sum(lens)//max(len(lens),1)}자, 최대 {max(lens, default=0)}자")
    print(f"저장: {out}")
    print("다음 단계: 조각별 Q&A 생성 (생성 결과 포맷: {chunk_id, question, answer} JSONL)")


if __name__ == "__main__":
    main()
