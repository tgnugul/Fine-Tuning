#!/usr/bin/env bash
# 대화 테스트 - 학습한 어댑터로 바로 말 걸어보기 (1단계 산출물 확인용).
#
# 사용:
#   bash scripts/chat.sh                 # config의 어댑터
#   bash scripts/chat.sh adapters/run3   # 지정 어댑터
set -euo pipefail
cd "$(dirname "$0")/.."
ADAPTER="${1:-$(python3 -c "import yaml;print(yaml.safe_load(open('configs/config.yaml'))['training']['adapter_dir'])")}"
MODEL="mlx-community/Qwen2.5-7B-Instruct-4bit"

echo "대화 모드 (Ctrl-C로 종료). 모델=$MODEL + $ADAPTER"
mlx_lm chat --model "$MODEL" --adapter-path "$ADAPTER"
