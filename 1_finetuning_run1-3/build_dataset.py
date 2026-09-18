"""[1-4] 실전 데이터셋 빌더 - run1 처방을 반영한 혼합 설계.

세 종류의 데이터를 정해진 비율로 섞어 최종 원본 데이터셋을 만든다:

  1. 도메인 데이터 (주재료): 복무규정 합성 Q&A (+ AI Hub 변환분이 있으면 함께)
  2. 거절 예시 (처방 1): 규정에 없는 질문에 물러서는 행동 학습
  3. replay 일반 데이터 (처방 2): KoAlpaca에서 소량 샘플링 → 일반 능력 망각 방지

비율은 configs/config.yaml 의 data_mix 섹션에서 조절:
  data_mix:
    refusal_ratio: 0.05   # 도메인 대비 5%
    replay_ratio: 0.08    # 도메인 대비 8%

출력: data/raw/dataset.jsonl (이후 validate → prepare → train 순서 그대로)

사용법:
  python scripts/build_dataset.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from common import load_config, read_jsonl, resolve, write_jsonl
from make_refusal_data import make_rows as make_refusal_rows

# 도메인 데이터로 취급할 파일들 (존재하는 것만 사용)
DOMAIN_FILES = [
    ("data/synth/qa_verified.jsonl", "synth"),            # 복무규정 합성 (검수 통과분)
    ("data/synth/qa_augmented_verified.jsonl", "aug"),    # run3: 사실 단위 증강 (검수 통과분)
    ("data/external/aihub.jsonl", "aihub"),               # AI Hub 변환분 (선택)
]


def load_domain() -> list[dict]:
    rows = []
    for rel, tag in DOMAIN_FILES:
        p = resolve(rel)
        if p.exists():
            part = read_jsonl(p)
            for r in part:
                rows.append({"question": r["question"], "answer": r["answer"], "source": tag})
            print(f"  도메인[{tag}]: {len(part):,}건 ({rel})")
        else:
            print(f"  도메인[{tag}]: 없음 - 건너뜀 ({rel})")
    return rows


def load_replay(n: int, seed: int) -> list[dict]:
    """KoAlpaca에서 n건 샘플링. 오프라인이거나 미설치면 건너뜀."""
    if n <= 0:
        return []
    try:
        from datasets import load_dataset
        ds = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
    except Exception as e:
        print(f"  replay: KoAlpaca 로드 실패({type(e).__name__}) - replay 없이 진행")
        return []
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    rows = [
        {"question": ds[i]["instruction"], "answer": ds[i]["output"], "source": "replay"}
        for i in idx
    ]
    print(f"  replay: {len(rows):,}건 샘플링 (KoAlpaca)")
    return rows


def main() -> None:
    cfg = load_config()
    mix = cfg.get("data_mix", {"refusal_ratio": 0.05, "replay_ratio": 0.08})
    seed = cfg["prepare"]["seed"]

    print("데이터 로드:")
    domain = load_domain()
    if not domain:
        sys.exit("[오류] 도메인 데이터가 없습니다. 먼저 합성 파이프라인(parse_doc → 생성 → verify_qa)을 완료하세요.")

    n_refusal = max(1, int(len(domain) * mix["refusal_ratio"]))
    n_replay = int(len(domain) * mix["replay_ratio"])

    refusal_all = make_refusal_rows()
    rng = random.Random(seed)
    rng.shuffle(refusal_all)
    refusal = refusal_all[:n_refusal]
    print(f"  거절 예시: {len(refusal)}건 (목표 {n_refusal}건, 보유 {len(refusal_all)}건)")

    replay = load_replay(n_replay, seed)

    dataset = domain + refusal + replay
    rng.shuffle(dataset)

    out = resolve(cfg["data"]["raw_file"])
    write_jsonl(out, dataset)

    total = len(dataset)
    print(f"\n최종 데이터셋: {total:,}건 → {out}")
    for tag in ("synth", "aihub", "refusal", "replay"):
        n = sum(1 for r in dataset if r["source"] == tag)
        if n:
            print(f"  - {tag}: {n:,}건 ({n / total:.1%})")
    print("다음 단계: python scripts/validate_data.py")


if __name__ == "__main__":
    main()
