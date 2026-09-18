"""[1-5] 정량 평가 - 원본 vs 학습 모델을 같은 문항으로 비교.

loss가 아니라 '행동'으로 판정한다는 원칙의 도구.
표준 문항마다 (a) 반드시 들어갈 키워드 (b) 중국어 혼입 여부 (c) 거절 정확성을
기계적으로 채점해, 원본과 어댑터의 점수를 나란히 낸다.

run3에서 이 평가로 키워드 적중률 12.5% → 75%, 중국어 2건 → 0건을 확인했다.

사용법 (맥):
  python scripts/evaluate.py --base            # 원본 모델
  python scripts/evaluate.py                   # config의 어댑터
  python scripts/evaluate.py --adapter adapters/run3

주의(아카이브 메모): 이 파일은 맥 원본을 보지 못한 상태에서 설계 설명대로
재구성한 것이다. 정본은 맥 ~/qwen-domain-ft/scripts/evaluate.py.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime

from common import load_config, read_jsonl, resolve

RE_CJK = re.compile(r"[一-鿿぀-ヿ]")   # 한자 + 가나 (중국어/일본어 혼입 감지)

# 표준 평가 문항: must(모두 포함) / any(하나 이상) / refuse(거절이어야 정답) / no_cjk
RUBRIC = [
    {"q": "재직 7년차 공무원의 연가는 며칠인가요?", "must": ["21일", "제15조"]},
    {"q": "병가를 8일 쓰려는데 진단서가 필요한가요?", "must": ["제18조"], "any": ["필요합니다", "첨부"]},
    {"q": "주차장 등록은 어떻게 하나요?", "refuse": True},
    {"q": "감기에 걸렸을 때 먹으면 좋은 음식 알려줘.", "no_cjk": True},
]
REFUSE_MARKERS = ["확인되지", "다루지 않", "규정되어 있지 않", "담당 부서", "관련 부서", "어렵습니다"]


def score(ans: str, rule: dict) -> tuple[bool, str]:
    if rule.get("no_cjk"):
        return (RE_CJK.search(ans) is None), ("중국어 혼입" if RE_CJK.search(ans) else "정답")
    if rule.get("refuse"):
        ok = any(m in ans for m in REFUSE_MARKERS)
        return ok, ("정답(거절)" if ok else "거절 실패(환각 의심)")
    for tok in rule.get("must", []):
        if tok not in ans:
            return False, f"필수 '{tok}' 누락"
    anys = rule.get("any")
    if anys and not any(t in ans for t in anys):
        return False, f"핵심 표현 없음: {anys}"
    return True, "정답"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true", help="어댑터 없이 원본 모델")
    ap.add_argument("--adapter", help="어댑터 경로 (예: adapters/run3)")
    args = ap.parse_args()

    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        sys.exit("[오류] mlx-lm이 필요합니다 (맥에서 실행).")

    cfg = load_config()
    if args.base:
        adapter, tag = None, "base"
    elif args.adapter:
        adapter, tag = str(resolve(args.adapter)), args.adapter.rstrip("/").split("/")[-1]
    else:
        adapter, tag = str(resolve(cfg["training"]["adapter_dir"])), "adapter"

    print(f"모델 로드: {cfg['model']['name']}" + (" (원본)" if adapter is None else f" + {tag}"))
    model, tokenizer = load(cfg["model"]["name"], adapter_path=adapter)
    sampler = make_sampler(temp=0.0)
    system_prompt = cfg["system_prompt"].strip()

    correct = 0
    lines = [f"# 평가 리포트 ({tag}, {datetime.now():%Y-%m-%d %H:%M})", ""]
    for rule in RUBRIC:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": rule["q"]}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        ans = generate(model, tokenizer, prompt=prompt, max_tokens=200, sampler=sampler).strip()
        ok, why = score(ans, rule)
        correct += ok
        mark = "✓" if ok else "✗"
        print(f"{mark} {rule['q']}  → {why}")
        lines += [f"### {mark} {rule['q']}", f"- 판정: {why}", f"- 답변: {ans}", ""]

    n = len(RUBRIC)
    print(f"\n총점: {correct}/{n} = {correct/n:.0%}")
    lines.insert(2, f"**총점: {correct}/{n} ({correct/n:.0%})**\n")
    out = resolve(f"eval/results/eval_{tag}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"리포트: {out}")


if __name__ == "__main__":
    main()
