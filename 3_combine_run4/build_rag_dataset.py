"""[3단계] 결합 학습 데이터 빌더 v2 - run4 실패(경우 구분 미발현)의 처방 반영.

run4의 교훈: 행동에도 밀도와 일관성이 필요하다.
  - v1의 실수: "N년차 → 즉답" 예시 수십 건 vs 경계 년차 경우구분 예시 15건의 행동 충돌
  - v2의 정책: "N년차/N년째"류 중의적 표현은 즉답 예시에서 전면 제외하고,
    전부 유형 3(해석 명시)으로 통일. 유형 3을 15건 → 54건으로 증량.

원리: 학습 형식 = 추론 형식. 시스템 프롬프트와 근거 블록은 rag_chat.py에서 그대로 가져옴.
근거는 실제 검색기로 구성 (정답 + 방해 조각 - 실전과 같은 소음 환경).

사용법 (맥, 인덱스 구축 후):
  python scripts/build_rag_dataset.py
  → data/processed_rag/train.jsonl / valid.jsonl
"""
from __future__ import annotations

import random
import re

from common import load_config, read_jsonl, resolve, write_jsonl
from make_refusal_data import make_rows as refusal_rows
from make_augmented_qa import yeonga_band
from rag_chat import RAG_SYSTEM, build_prompt

QA_FILES = ["data/synth/qa_verified.jsonl", "data/synth/qa_augmented_verified.jsonl"]
REFUSAL_ANSWER = "제공된 규정에서 확인되지 않는 내용입니다. 정확한 사항은 관련 부서에 확인해 주시기 바랍니다."

# "N년차/N년째"류 표현: 만 나이식 해석이 갈리는 중의적 표현 → 즉답 예시에서 전면 제외
RE_YEONCHA = re.compile(r"\d+\s*년\s*차|\d+년째|올해로\s*\d+년")


def get_retriever():
    try:
        from rag_search import Retriever
        return Retriever(), "real"
    except SystemExit:
        pass
    except ImportError:
        pass

    class MockRetriever:  # 스모크 테스트 전용
        def __init__(self):
            cfg = load_config()["rag"]
            all_chunks = read_jsonl(resolve(cfg["chunks_file"]))
            self.chunks = {c["chunk_id"]: c for c in all_chunks}
            self.units = [c for c in all_chunks if not c.get("has_subs")]

        def search(self, query: str, top_k: int = 3):
            q_words = set(re.findall(r"[가-힣]{2,}", query))
            scored = []
            for c in self.units:
                words = set(re.findall(r"[가-힣]{2,}", c["text"]))
                scored.append((c["chunk_id"], len(q_words & words) / (len(q_words) or 1)))
            scored.sort(key=lambda x: -x[1])
            return scored[:top_k]

        def get_chunk(self, cid):
            return self.chunks[cid]

    return MockRetriever(), "mock"


def to_example(question: str, contexts: list[dict], answer: str, kind: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": build_prompt(question, contexts)},
            {"role": "assistant", "content": answer},
        ],
        "kind": kind,
    }


def main() -> None:
    cfg = load_config()
    rcfg = cfg["rag_training"]
    seed = cfg["prepare"]["seed"]
    retriever, mode = get_retriever()
    if mode == "mock":
        print("[경고] 임베딩 인덱스가 없어 모의 검색으로 빌드합니다 - 스모크 테스트 전용.")

    def parent_of(cid: str) -> str:
        return retriever.get_chunk(cid).get("parent", cid)

    examples: list[dict] = []
    stats = {"retrieval_miss_skip": 0, "yeoncha_skip": 0}

    # ---- 유형 1: 근거 인용 (년차류 중의 표현 제외) ----
    for f in QA_FILES:
        p = resolve(f)
        if not p.exists():
            continue
        for r in read_jsonl(p):
            q, gold = r["question"], r.get("chunk_id", "")
            if RE_YEONCHA.search(q):
                stats["yeoncha_skip"] += 1       # 년차류는 전부 유형 3이 담당 (정책 일관성)
                continue
            hits = retriever.search(q, 3)
            contexts = [retriever.get_chunk(cid) for cid, _ in hits]
            if gold not in {parent_of(cid) for cid, _ in hits}:
                stats["retrieval_miss_skip"] += 1
                continue
            examples.append(to_example(q, contexts, r["answer"], "grounded"))

    # ---- 유형 2: 거절 ----
    for r in refusal_rows():
        hits = retriever.search(r["question"], 3)
        contexts = [retriever.get_chunk(cid) for cid, _ in hits]
        examples.append(to_example(r["question"], contexts, REFUSAL_ANSWER, "refusal"))

    # ---- 유형 3: "N년차" 해석 명시 (전 년차 × 다양한 표현으로 증량) ----
    templates = [
        "올해로 {y}년차인데 연가가 며칠인가요?",
        "{y}년차 공무원 연가 일수 알려주세요.",
        "저 {y}년차인데 올해 연차 며칠 쓸 수 있어요?",
        "공무원 된 지 {y}년차입니다. 연가가 얼마나 되나요?",
        "이제 {y}년차인데 쓸 수 있는 연가가 궁금합니다.",
        "{y}년째 근무 중인데 연가 일수가 어떻게 되나요?",
    ]
    for y in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
        lo_band, lo_days = yeonga_band(y - 0.5)   # "N년차 = 만 N-1년대" 해석
        hi_band, hi_days = yeonga_band(y)         # "N년차 = 만 N년" 해석
        if lo_days == hi_days:
            answer = (
                f"'{y}년차'가 만 {y-1}년대인지 만 {y}년인지에 따라 재직기간 구간이 달라질 수 있으나, "
                f"복무규정 제15조에 따르면 두 경우 모두 연가 일수는 {lo_days}일로 같습니다."
            )
        else:
            answer = (
                f"'{y}년차'는 재직기간이 만 {y-1}년대인지 만 {y}년을 넘었는지에 따라 연가 일수가 달라집니다. "
                f"복무규정 제15조에 따르면 재직기간 {lo_band}이면 {lo_days}일, {hi_band}이면 {hi_days}일입니다. "
                f"본인의 재직기간(만 기준)을 확인하시면 정확한 일수를 알 수 있습니다."
            )
        for tpl in templates:
            q = tpl.format(y=y)
            hits = retriever.search(q, 3)
            contexts = [retriever.get_chunk(cid) for cid, _ in hits]
            examples.append(to_example(q, contexts, answer, "casesplit"))

    # ---- 유형 4: 안정성 replay ----
    n_replay = int(len(examples) * rcfg.get("replay_ratio", 0.10))
    try:
        from datasets import load_dataset
        ds = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
        rng = random.Random(seed)
        for i in rng.sample(range(len(ds)), min(n_replay, len(ds))):
            examples.append({
                "messages": [
                    {"role": "system", "content": cfg["system_prompt"].strip()},
                    {"role": "user", "content": ds[i]["instruction"]},
                    {"role": "assistant", "content": ds[i]["output"]},
                ],
                "kind": "replay",
            })
        print(f"  replay: {n_replay}건 혼합")
    except Exception as e:
        print(f"  replay: 로드 실패({type(e).__name__}) - 없이 진행")

    # ---- 분리 및 저장 ----
    rng = random.Random(seed)
    rng.shuffle(examples)
    n_valid = max(4, int(len(examples) * rcfg["valid_ratio"]))
    valid, train = examples[:n_valid], examples[n_valid:]

    out_dir = resolve(rcfg["output_dir"])
    strip = lambda rows: [{"messages": r["messages"]} for r in rows]
    write_jsonl(out_dir / "train.jsonl", strip(train))
    write_jsonl(out_dir / "valid.jsonl", strip(valid))

    from collections import Counter
    kinds = Counter(r["kind"] for r in examples)
    print(f"\n결합 데이터셋 v2: 총 {len(examples)}건 (train {len(train)} / valid {len(valid)}) [{mode} 검색]")
    for k, v in kinds.most_common():
        print(f"  - {k}: {v}건 ({v/len(examples):.0%})")
    print(f"  - 검색 실패 제외: {stats['retrieval_miss_skip']}건 / 년차류 대체: {stats['yeoncha_skip']}건")
    print(f"저장: {out_dir}")
    print("다음 단계: bash scripts/train_rag.sh  (학습 전 config의 rag_training.adapter_dir을 run4b로!)")


if __name__ == "__main__":
    main()
