#!/usr/bin/env bash
# =============================================================
# 3~4단계: RAG 형식 데이터로 LoRA 학습 (결합 run4b, CoT run5 공용)
#
# train.sh와 같은 방식이지만, 데이터 폴더가 data/processed_rag 이고
# 어댑터 경로는 config의 rag_training 섹션을 읽는다.
# run4b / run5 를 돌릴 때 config의 rag_training.adapter_dir 만 바꿔 재사용한다.
#
# 실행:
#   bash scripts/train_rag.sh
#
# 주의(아카이브 메모): train.sh 패턴대로 재구성한 것. 정본은 맥.
# =============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
import yaml, pathlib
cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
r = cfg["rag_training"]

data_dir = pathlib.Path(r["output_dir"])
n_valid = sum(1 for _ in open(data_dir / "valid.jsonl", encoding="utf-8"))
batch = min(r.get("batch_size", 4), max(1, n_valid))

lora_cfg = {
    "model": cfg["model"]["name"],
    "train": True,
    "fine_tune_type": "lora",
    "data": r["output_dir"],              # data/processed_rag
    "seed": cfg["prepare"]["seed"],
    "num_layers": r.get("num_layers", 16),
    "batch_size": batch,
    "iters": r["iters"],                  # run4b=150, run5=180 등
    "learning_rate": r.get("learning_rate", 1.0e-05),
    "steps_per_eval": r.get("steps_per_eval", 25),
    "save_every": r.get("save_every", 25),
    "adapter_path": r["adapter_dir"],     # adapters/run4b, adapters/run5 ...
    "max_seq_length": r.get("max_seq_length", 2048),
    "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
}
pathlib.Path("configs/lora_rag.generated.yaml").write_text(
    yaml.safe_dump(lora_cfg, allow_unicode=True), encoding="utf-8")
print(f"LoRA 설정 생성 → configs/lora_rag.generated.yaml  (iters={lora_cfg['iters']}, adapter={lora_cfg['adapter_path']})")
PY

echo "=== RAG 학습 시작 ==="
if command -v mlx_lm >/dev/null 2>&1; then
  mlx_lm lora -c configs/lora_rag.generated.yaml
else
  mlx_lm.lora -c configs/lora_rag.generated.yaml
fi
echo "=== 완료 ==="
echo "검증: python scripts/rag_chat.py --adapter <adapter_dir> -q \"질문\" --show-context"
