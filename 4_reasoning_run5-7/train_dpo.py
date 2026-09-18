"""[4단계-②] DPO 학습 v2 (mlx-tune) - 학습 후 병합 저장으로 어댑터 호환성 문제 우회.

v1 대비 변경: 학습 직후 save_pretrained_merged로 통짜 모델(fused_run6) 저장.
  - mlx-tune 어댑터 형식과 mlx-lm 로더의 비호환을 원천 회피
  - merged_16bit 사용: DPO는 학습률 5e-7의 미세 조정이라 4bit 재양자화 시
    변화가 반올림에 소실될 수 있음 → 16bit 병합으로 보존 (~15GB)

전제:
  pip install mlx-tune
  fused_run5 존재 (mlx_lm fuse --adapter-path adapters/run5 --save-path fused_run5 --dequantize)
  data/dpo/pairs.jsonl 존재 (build_dpo_pairs.py)

사용법:
  python scripts/train_dpo.py
  → fused_run6/ (병합 완료 모델)
검증:
  python scripts/rag_chat.py --model fused_run6 --base -q "..."
  (--base는 "추가 어댑터 없음" 의미로 사용)
"""
from __future__ import annotations

import json
import sys

from common import resolve


def main() -> None:
    try:
        from mlx_tune import FastLanguageModel, DPOTrainer, DPOConfig
    except ImportError:
        sys.exit("[오류] mlx-tune이 필요합니다: pip install mlx-tune")

    fused = resolve("fused_run5")
    if not fused.exists():
        sys.exit("[오류] fused_run5가 없습니다. 먼저:\n"
                 "  mlx_lm fuse --model mlx-community/Qwen2.5-7B-Instruct-4bit "
                 "--adapter-path adapters/run5 --save-path fused_run5 --dequantize")

    pairs_path = resolve("data/dpo/pairs.jsonl")
    if not pairs_path.exists():
        sys.exit("[오류] 선호 쌍이 없습니다. 먼저 python scripts/build_dpo_pairs.py")
    data = [json.loads(l) for l in open(pairs_path, encoding="utf-8")]
    print(f"선호 쌍 {len(data)}건 로드")

    model, tokenizer = FastLanguageModel.from_pretrained(str(fused), max_seq_length=2048)
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
    )

    config = DPOConfig(
        beta=0.1,             # KL 페널티: 기준 모델에서 멀어지는 정도의 제동
        loss_type="sigmoid",  # 표준 DPO 손실
        learning_rate=5e-7,   # 선호학습은 매우 낮은 학습률이 정석
        max_steps=60,
        logging_steps=5,
        output_dir=str(resolve("adapters/run6_dpo")),
    )
    trainer = DPOTrainer(model=model, train_dataset=data, tokenizer=tokenizer, args=config)
    result = trainer.train()
    print(f"학습 상태: {result.get('status')}")

    # 어댑터를 모델에 병합해 통짜로 저장 (mlx-lm에서 바로 로드 가능한 HF 형식)
    out = str(resolve("fused_run6"))
    model.save_pretrained_merged(out, tokenizer, save_method="merged_16bit")
    print(f"\n병합 모델 저장: {out}")
    print("검증: python scripts/rag_chat.py --model fused_run6 --base -q \"만 12년 재직했는데 장기재직휴가는 며칠인가요?\"")


if __name__ == "__main__":
    main()
