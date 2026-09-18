"""3단계: Qwen 챗 템플릿 변환 + 학습/검증/테스트 분리.

검증 통과 데이터를 mlx-lm이 바로 학습할 수 있는 chat 포맷으로 변환합니다.
지난 학습 실패의 직접 원인이었던 부분을 여기서 해결합니다:
raw Q/A 텍스트를 그대로 학습시키는 대신, Qwen이 사전학습 때 사용한
system/user/assistant 대화 구조(messages)로 감싸서 학습시킵니다.
mlx-lm이 학습 시 이 messages를 Qwen 토크나이저의 공식 챗 템플릿
(<|im_start|>...<|im_end|>)으로 자동 렌더링합니다.

출력 (data/processed/):
  - train.jsonl  학습용
  - valid.jsonl  학습 중 손실 모니터링용 (과적합 감시)
  - holdout.jsonl 학습에 전혀 쓰지 않는 홀드아웃 - 학습 후 평가 전용

각 줄 포맷:
  {"messages": [
      {"role": "system", "content": "<시스템 프롬프트>"},
      {"role": "user", "content": "<질문>"},
      {"role": "assistant", "content": "<답변>"}]}

사용법:
  python scripts/prepare_data.py
"""
from __future__ import annotations

import random
import sys

from common import load_config, read_jsonl, resolve, write_jsonl


def to_chat(row: dict, system_prompt: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]
    }


def main() -> None:
    cfg = load_config()
    p = cfg["prepare"]
    system_prompt = cfg["system_prompt"].strip()

    src = resolve(cfg["data"]["validated_file"])
    if not src.exists():
        sys.exit(f"[오류] 검증된 데이터가 없습니다: {src}\n먼저 python scripts/validate_data.py 를 실행하세요.")

    rows = read_jsonl(src)
    print(f"검증 통과 데이터: {len(rows):,}건")

    # 재현 가능한 셔플
    rng = random.Random(p["seed"])
    rng.shuffle(rows)

    # 소규모 리허설 모드
    max_samples = int(p.get("max_samples") or 0)
    if max_samples > 0:
        rows = rows[:max_samples]
        print(f"[소규모 모드] {len(rows):,}건만 사용 (config: prepare.max_samples)")

    n = len(rows)
    if n < 20:
        sys.exit(f"[오류] 데이터가 너무 적습니다 ({n}건). 최소 20건 이상 필요합니다.")

    n_valid = max(1, int(n * p["valid_ratio"]))
    n_test = max(1, int(n * p["test_ratio"]))
    n_train = n - n_valid - n_test

    test_rows = rows[:n_test]
    valid_rows = rows[n_test:n_test + n_valid]
    train_rows = rows[n_test + n_valid:]

    out_dir = resolve(p["output_dir"])
    # 과거 버전이 남긴 test.jsonl 제거 - mlx-lm이 학습 데이터로 읽다가 포맷 오류를 냄
    (out_dir / "test.jsonl").unlink(missing_ok=True)
    write_jsonl(out_dir / "train.jsonl", [to_chat(r, system_prompt) for r in train_rows])
    write_jsonl(out_dir / "valid.jsonl", [to_chat(r, system_prompt) for r in valid_rows])
    # 홀드아웃은 평가 스크립트가 질문/정답을 그대로 쓰도록 raw 형태 유지.
    # 주의: 파일명을 test.jsonl로 하면 mlx-lm이 학습 데이터로 읽으려다 포맷 오류를 냄.
    write_jsonl(out_dir / "holdout.jsonl", test_rows)

    print(f"train  : {n_train:,}건 → {out_dir / 'train.jsonl'}")
    print(f"valid  : {n_valid:,}건 → {out_dir / 'valid.jsonl'}  (학습 중 loss 모니터링)")
    print(f"holdout: {n_test:,}건 → {out_dir / 'holdout.jsonl'} (학습 후 평가 전용, 학습에 미사용)")
    print("다음 단계: bash scripts/train.sh")


if __name__ == "__main__":
    main()
