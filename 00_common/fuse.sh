#!/usr/bin/env bash
# 배포용 모델 병합 - 어댑터를 원본 가중치에 합쳐 통짜 모델로 저장.
#
# 중요: 4단계 DPO 베이스로 쓸 fused_run5는 반드시 --dequantize(16bit)로 만든다.
#   4bit 재양자화는 LoRA의 미세한 가중치 변화를 반올림에 소실시켜
#   학습된 행동(풀이 형식ㆍ언어ㆍ거절 정책)을 망가뜨린다 (트러블슈팅 #10).
#
# 사용:
#   bash scripts/fuse.sh adapters/run5 fused_run5
set -euo pipefail
cd "$(dirname "$0")/.."
ADAPTER="${1:-adapters/run5}"
OUT="${2:-fused_run5}"
MODEL="mlx-community/Qwen2.5-7B-Instruct-4bit"

echo "융합: $MODEL + $ADAPTER → $OUT (16bit, --dequantize)"
mlx_lm fuse --model "$MODEL" --adapter-path "$ADAPTER" --save-path "$OUT" --dequantize
echo "완료. 용량 확인: ls -lh $OUT  (16bit면 ~15GB)"
