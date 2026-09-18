"""파이프라인 공통 유틸리티: 설정 로드, JSONL 읽기/쓰기."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

# 프로젝트 루트 (scripts/의 상위 폴더)
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel_path: str) -> Path:
    """config에 적힌 상대 경로를 프로젝트 루트 기준 절대 경로로 변환."""
    p = Path(rel_path)
    return p if p.is_absolute() else ROOT / p


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
