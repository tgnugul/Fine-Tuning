"""[4단계-③] 자기개선 루프 2/2: 검증ㆍ데이터화 - 합격작만 다음 세대의 교재가 된다.

핵심 원칙: **답만 맞으면 통과가 아니다. 궤적도 검사한다.**
  지각 문항의 "2.5일" 사례처럼 답은 맞는데 풀이가 틀린 샘플이 통과되면
  그 결함이 다음 세대에 유전된다 (모델 붕괴의 축소판).
  → 검증기는 정답 + 형식 + 인용 조문 + 결함 패턴 + 언어 비율을 모두 본다.

사용법:
  기준선 채점(데이터 생성까지):  python scripts/self_improve_filter.py
  run7 재측정(리포트만):        python scripts/self_improve_filter.py \
                                  --candidates data/self_improve/candidates_run7.jsonl --tag run7
학습 (run7):
  mlx_lm lora --model fused_run6 --train --data data/self_improve \\
    --adapter-path adapters/run7 --iters 30 --batch-size 4 \\
    --learning-rate 1e-5 --steps-per-eval 5 --save-every 10 --num-layers 16
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

from common import resolve
from rag_chat import RAG_SYSTEM, build_prompt

ALL_YEONGA_DAYS = {11, 15, 16, 17, 20, 21}


def hangul_ratio(t: str) -> float:
    chars = [c for c in t if not c.isspace()]
    if not chars:
        return 0.0
    return sum("가" <= c <= "힣" for c in chars) / len(chars)


def check(row: dict) -> str | None:
    """실패 사유를 반환. None이면 합격."""
    a = row["answer"]
    t = row["truth"]
    kind = row["kind"]
    ans_part = a.split("[답변]")[-1]

    # --- 공통: 언어ㆍ거절ㆍ결함 패턴 ---
    if hangul_ratio(a) < 0.2:
        return "한글 비율 미달"
    if len(re.findall(r"[A-Za-z]", a)) / max(len(a), 1) > 0.10:
        return "영어 이탈"
    if "확인되지 않" in a and kind != "longserve":
        return "부당 거절"        # 전 문항이 근거로 답 가능한 질문
    if re.search(r"\d\.5\s*일", a):
        return "궤적 결함(.5일)"   # 나눗셈을 반올림 없이 서술한 결함 유전 차단

    # --- 절차별: 정답 + 인용 + 형식 ---
    if kind == "yeonga":
        if "[풀이]" not in a or "[답변]" not in a:
            return "형식 이탈"
        if "15조" not in a:
            return "인용 누락/오류"
        if t["band"] not in a:
            return "구간 문자열 불일치"
        if f"{t['days']}일" not in ans_part:
            return "일수 오답"
        others = {f"{d}일" for d in ALL_YEONGA_DAYS - {t["days"]}}
        if any(o in ans_part for o in others):
            return "오답 일수 혼입"

    elif kind == "byeongga":
        if "[풀이]" not in a or "[답변]" not in a:
            return "형식 이탈"
        if "18조" not in a:
            return "인용 누락/오류"
        negative = ("필요하지 않" in ans_part) or ("필요 없" in ans_part) or ("않아도" in ans_part)
        if t["need"] and (negative or "필요" not in ans_part):
            return "판정 오답"
        if not t["need"] and not negative:
            return "판정 오답"

    elif kind == "jigak":
        if "[풀이]" not in a or "[답변]" not in a:
            return "형식 이탈"
        if "17조" not in a:
            return "인용 누락/오류"
        if f"연가 {t['d']}일" not in a:
            return "일수 오답"
        if f"{t['d'] + 1}일" in ans_part:
            return "오답 일수 혼입"

    elif kind == "longserve":
        if t["days"] is None:
            # 만 5년 미만 - "대상 아님"을 규정 근거로 말해야 정답
            if re.search(r"장기재직휴가는\s*\d+\s*일", a):
                return "판정 오답(미달인데 일수 부여)"
            if not any(w in a for w in ("미만", "아니", "않", "없")):
                return "판정 오답"
        else:
            if "[풀이]" not in a or "[답변]" not in a:
                return "형식 이탈"
            if "20조" not in a:
                return "인용 누락/오류"
            if t["band"] not in a:
                return "구간 문자열 불일치"
            if f"{t['days']}일" not in ans_part:
                return "일수 오답"

    elif kind == "casesplit":
        # 두 해석의 일수를 모두 제시해야 정답 (한쪽 단정은 실패)
        if f"{t['d1']}일" not in a or f"{t['d2']}일" not in a:
            return "경우 구분 실종"
        if "15조" not in a:
            return "인용 누락/오류"

    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/self_improve/candidates.jsonl")
    ap.add_argument("--tag", default=None,
                    help="설정 시 report_<tag>.md만 생성 (train/valid를 덮어쓰지 않음)")
    args = ap.parse_args()

    src = resolve(args.candidates)
    if not src.exists():
        raise SystemExit("[오류] 후보가 없습니다. 먼저 python scripts/self_improve_generate.py")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]

    reasons = Counter()
    by_kind = defaultdict(lambda: [0, 0])          # kind → [합격, 전체]
    accepted: dict[str, dict] = {}                 # 질문당 합격작 1건 (중복 과대표집 방지)
    for row in rows:
        by_kind[row["kind"]][1] += 1
        fail = check(row)
        if fail:
            reasons[f"{row['kind']}: {fail}"] += 1
            continue
        by_kind[row["kind"]][0] += 1
        accepted.setdefault(row["q"], row)

    out_dir = resolve("data/self_improve")
    n_valid = 0
    if not args.tag:
        exs = [{"messages": [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": build_prompt(r["q"], r["contexts"])},
            {"role": "assistant", "content": r["answer"]},
        ]} for r in accepted.values()]
        n_valid = 4 if len(exs) >= 12 else 2
        for name, part in [("valid.jsonl", exs[:n_valid]), ("train.jsonl", exs[n_valid:])]:
            with open(out_dir / name, "w", encoding="utf-8") as f:
                for e in part:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # --- 합격률 지도: 어느 절차가 아직 약한가 ---
    n_q = len({r["q"] for r in rows})
    n_ok = sum(v[0] for v in by_kind.values())
    title = f" ({args.tag})" if args.tag else ""
    lines = [f"# 자기개선 루프: 검증 리포트{title}\n",
             f"- 후보 {len(rows)}건(질문 {n_q}개) → 샘플 합격 {n_ok}건 ({n_ok / len(rows):.0%}), "
             f"질문 커버 {len(accepted)}/{n_q}\n",
             "\n## 절차별 샘플 합격률 (= 모델이 약한 절차의 지도)\n",
             "| 절차 | 합격/전체 | 합격률 |", "|---|---|---|"]
    for kind, (ok, total) in sorted(by_kind.items()):
        lines.append(f"| {kind} | {ok}/{total} | {ok / total:.0%} |")
    lines.append("\n## 탈락 사유 분포\n")
    for reason, n in reasons.most_common():
        lines.append(f"- {reason}: {n}건")
    report = out_dir / (f"report_{args.tag}.md" if args.tag else "report.md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"샘플 합격 {n_ok}/{len(rows)} ({n_ok / len(rows):.0%})")
    print(f"리포트: {report}")
    if not args.tag:
        print(f"학습 데이터: train {len(accepted) - n_valid} / valid {n_valid}")


if __name__ == "__main__":
    main()
