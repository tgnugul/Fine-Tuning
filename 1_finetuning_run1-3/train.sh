#!/usr/bin/env bash
# =============================================================
# 4단계: MLX LoRA 학습 (맥 스튜디오 / Apple Silicon 전용)
#
# 사전 준비 (최초 1회):
#   pip install mlx-lm
#
# 실행:
#   bash scripts/train.sh
#
# configs/config.yaml 의 training 섹션 값을 읽어
# configs/lora_config.generated.yaml 을 만든 뒤 mlx_lm 으로 학습합니다.
# 학습 로그에서 반드시 볼 것:
#   - "Val loss" 가 계속 내려가는지 (올라가기 시작하면 과적합 → iters 축소)
# =============================================================
set -euo pipefail
cd "$(dirname "$0")/.."   # 프로젝트 루트로 이동

python3 - <<'PY'
import yaml, pathlib

cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
t = cfg["training"]

# 데이터가 batch_size보다 적으면 자동 축소 (소규모 리허설 대비)
data_dir = pathlib.Path(cfg["prepare"]["output_dir"])
n_valid = sum(1 for _ in open(data_dir / "valid.jsonl", encoding="utf-8"))
batch = min(t["batch_size"], max(1, n_valid))
if batch != t["batch_size"]:
    print(f"[안내] 검증셋이 {n_valid}건뿐이라 batch_size를 {t['batch_size']}→{batch}로 자동 축소")

lora_cfg = {
    "model": cfg["model"]["name"],
    "train": True,
    "fine_tune_type": "lora",
    "data": cfg["prepare"]["output_dir"],       # train.jsonl / valid.jsonl 이 있는 폴더
    "seed": cfg["prepare"]["seed"],
    "num_layers": t["num_layers"],
    "batch_size": batch,
    "iters": t["iters"],
    "learning_rate": t["learning_rate"],
    "steps_per_report": t["steps_per_report"],
    "steps_per_eval": t["steps_per_eval"],      # 이 간격마다 valid loss 출력
    "save_every": t["save_every"],
    "adapter_path": t["adapter_dir"],
    "max_seq_length": t["max_seq_length"],
    "val_batches": 25,
    "grad_checkpoint": False,                   # 메모리 넉넉하면 False가 더 빠름 (부족하면 True)
    "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
}
out = pathlib.Path("configs/lora_config.generated.yaml")
out.write_text(yaml.safe_dump(lora_cfg, allow_unicode=True), encoding="utf-8")
print(f"LoRA 설정 생성 → {out}")
PY

mkdir -p "$(python3 -c "import yaml;print(yaml.safe_load(open('configs/config.yaml'))['training']['adapter_dir'])")"

echo "=== 학습 시작 ==="
# mlx-lm 신버전은 'mlx_lm lora', 구버전은 'mlx_lm.lora' 명령을 사용
if command -v mlx_lm >/dev/null 2>&1; then
  mlx_lm lora -c configs/lora_config.generated.yaml
else
  mlx_lm.lora -c configs/lora_config.generated.yaml
fi

echo ""
echo "=== 학습 완료 ==="
echo "어댑터 저장 위치: $(python3 -c "import yaml;print(yaml.safe_load(open('configs/config.yaml'))['training']['adapter_dir'])")"
echo "다음 단계:"
echo "  대화 테스트  : bash scripts/chat.sh"
echo "  정량 평가    : python scripts/evaluate.py"
echo "  모델 합치기  : bash scripts/fuse.sh (배포용, 선택)"
