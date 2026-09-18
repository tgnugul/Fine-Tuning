"""[4단계-①] KMMLU 벤치마크 채점기 - 표준 벤치마크로 전후 비교 + 부작용 검사.

목적:
  1. "능력 변화를 표준 점수로 증명하는 법" 체험 (모델 발표 때 벤치마크 표가 나오는 이유)
  2. 부작용 검사: 도메인 CoT 학습(run5)이 일반 능력을 깎지 않았는지
     → 기대: 원본 ≈ run5. 크게 떨어지면 망각의 신호

방식: KMMLU 객관식을 zero-shot으로 제시, temp=0에서 첫 A~D 글자를 답으로 채점.

사용법 (맥):
  python scripts/eval_benchmark.py --base            # 원본 모델
  python scripts/eval_benchmark.py --adapter adapters/run5
  python scripts/eval_benchmark.py --subjects Law Management --n 30
  → eval/results/benchmark_<tag>.md
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime

from common import load_config, resolve

DEFAULT_SUBJECTS = ["Law", "Management", "Korean-History"]
CHOICES = "ABCD"


def build_prompt(row: dict) -> str:
    return (
        "다음 문제의 정답을 A, B, C, D 중 하나로만 답하시오.\n\n"
        f"문제: {row['question']}\n"
        f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n"
        "정답:"
    )


def parse_choice(text: str) -> str | None:
    m = re.search(r"[ABCD]", text.upper())
    return m.group(0) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true", help="어댑터 없이 원본 모델")
    ap.add_argument("--adapter", help="어댑터 경로 (예: adapters/run5)")
    ap.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--n", type=int, default=50, help="과목당 문항 수")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError as e:
        sys.exit(f"[오류] 필요한 패키지가 없습니다: {e}")

    cfg = load_config()
    if args.base:
        adapter, tag = None, "base"
    elif args.adapter:
        adapter, tag = str(resolve(args.adapter)), args.adapter.rstrip("/").split("/")[-1]
    else:
        adapter, tag = str(resolve(cfg["training"]["adapter_dir"])), "default_adapter"

    print(f"모델 로드: {cfg['model']['name']}" + (" (원본)" if adapter is None else f" + {adapter}"))
    model, tokenizer = load(cfg["model"]["name"], adapter_path=adapter)
    sampler = make_sampler(temp=0.0)

    results: dict[str, tuple[int, int]] = {}   # subject -> (정답 수, 전체)
    for subject in args.subjects:
        try:
            ds = load_dataset("HAERAE-HUB/KMMLU", subject, split="test")
        except Exception as e:
            print(f"[건너뜀] {subject}: 로드 실패 ({type(e).__name__}). "
                  f"과목명을 확인하세요 (예: Law, Management, Korean-History, Economics 등)")
            continue
        n = min(args.n, len(ds))
        correct = 0
        for i in range(n):
            row = ds[i]
            messages = [{"role": "user", "content": build_prompt(row)}]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            out = generate(model, tokenizer, prompt=prompt, max_tokens=5, sampler=sampler)
            pred = parse_choice(out)
            gold = CHOICES[int(row["answer"]) - 1]   # KMMLU answer: 1~4
            correct += int(pred == gold)
            if (i + 1) % 10 == 0:
                print(f"  {subject}: {i+1}/{n} (누적 정확도 {correct/(i+1):.1%})")
        results[subject] = (correct, n)
        print(f"{subject}: {correct}/{n} = {correct/n:.1%}")

    if not results:
        sys.exit("[오류] 채점된 과목이 없습니다.")

    total_c = sum(c for c, _ in results.values())
    total_n = sum(n for _, n in results.values())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = [
        f"# KMMLU 벤치마크 리포트 ({tag}, {stamp})",
        "",
        f"- 모델: {cfg['model']['name']}" + (" (원본)" if adapter is None else f" + {tag}"),
        f"- 방식: zero-shot 객관식, temp=0, 과목당 최대 {args.n}문항",
        "",
        "| 과목 | 정확도 |",
        "|---|---|",
    ] + [f"| {s} | {c}/{n} ({c/n:.1%}) |" for s, (c, n) in results.items()] + [
        f"| **전체** | **{total_c}/{total_n} ({total_c/total_n:.1%})** |",
        "",
        "해석: 무작위 기대치는 25%. 원본과 어댑터의 점수 차가 크면(±5%p 이상) 원인 분석 필요.",
    ]
    out_path = resolve(f"eval/results/benchmark_{tag}_{stamp}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n전체: {total_c}/{total_n} = {total_c/total_n:.1%}")
    print(f"리포트: {out_path}")


if __name__ == "__main__":
    main()
