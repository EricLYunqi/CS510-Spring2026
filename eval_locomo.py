"""
Minimal LoCoMo evaluation example for an agent memory framework.

This is a starting point — replace `SimpleMemory` with your real memory module
and keep the rest of the eval loop.

Setup:
  1. pip install nano-vllm sentence-transformers numpy tqdm transformers nltk

  2. Get the LoCoMo data:
       git clone https://github.com/snap-research/locomo
       # the file we need is locomo/data/locomo10.json

  3. python eval_locomo.py
"""

from __future__ import annotations

import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from tqdm import tqdm


# ---------- Config ----------
LOCOMO_PATH = os.getenv("LOCOMO_PATH", "locomo/data/locomo10.json")
MODEL       = os.getenv("MODEL", "Qwen/Qwen3-4B")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
TOP_K       = 8
MAX_TOKENS  = 256

pipe     = pipeline("text-generation", model=MODEL, device="mps")
embedder = SentenceTransformer(EMBED_MODEL)


# ---------- Memory: a minimal vector store ----------
# Replace this whole class with your framework's memory module.
class SimpleMemory:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.metas: list[dict[str, Any]] = []
        self.vecs: np.ndarray | None = None

    def write(self, text: str, meta: dict[str, Any]) -> None:
        self.texts.append(text)
        self.metas.append(meta)
        v = embedder.encode([text], normalize_embeddings=True)
        self.vecs = v if self.vecs is None else np.vstack([self.vecs, v])

    def recall(self, query: str, k: int = TOP_K) -> list[dict[str, Any]]:
        if self.vecs is None or not self.texts:
            return []
        q = embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.vecs @ q
        top = np.argsort(-scores)[:k]
        return [{"text": self.texts[i], "meta": self.metas[i], "score": float(scores[i])}
                for i in top]


# ---------- Conversation -> memory ----------
def ingest_conversation(memory: SimpleMemory, sample: dict[str, Any]) -> None:
    """Walk each session in chronological order and write each turn to memory.
    A real memory framework would also summarize, consolidate, prune, etc."""
    conv = sample["conversation"]
    session_keys = sorted(
        (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
        key=lambda k: int(k.split("_")[1]),
    )
    for sk in session_keys:
        ts = conv.get(f"{sk}_date_time", "")
        for turn in conv[sk]:
            text = f"[{ts}] {turn['speaker']}: {turn['text']}"
            memory.write(text, meta={"session": sk,
                                      "dia_id":  turn["dia_id"],
                                      "speaker": turn["speaker"]})


# ---------- Answering ----------
ANSWER_SYSTEM = (
    "You answer questions about a long-running conversation between two people. "
    "You will be given excerpts retrieved from the conversation memory. "
    "Answer concisely based ONLY on the excerpts. "
    "If the answer is not present, say \"I don't know\"."
)

def answer(memory: SimpleMemory, question: str) -> dict[str, Any]:
    hits = memory.recall(question, k=TOP_K)
    context = "\n".join(f"- {h['text']}" for h in hits)
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user",   "content": f"Memory excerpts:\n{context}\n\nQuestion: {question}"},
    ]
    result = pipe(messages, max_new_tokens=MAX_TOKENS, do_sample=False)
    return {"prediction": result[0]["generated_text"][-1]["content"].strip(),
            "retrieved":  [h["meta"] for h in hits]}


# ---------- LoCoMo official scoring ----------
# Ported verbatim from snap-research/locomo task_eval/evaluation.py:
#   - normalize_answer
#   - f1_score (Porter-stemmed token F1)
#   - f1       (multi-answer, comma-split)
#   - eval_question_answering's per-category dispatch
_ps = PorterStemmer()

def _normalize_answer(s: str) -> str:
    s = s.replace(',', "")
    s = s.lower()
    s = re.sub(r'\b(a|an|the|and)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = ' '.join(s.split())
    return s

def _f1_token(prediction: str, ground_truth: str) -> float:
    pred_toks = [_ps.stem(w) for w in _normalize_answer(prediction).split()]
    gt_toks   = [_ps.stem(w) for w in _normalize_answer(ground_truth).split()]
    common = Counter(pred_toks) & Counter(gt_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall    = num_same / len(gt_toks)
    return (2 * precision * recall) / (precision + recall)

def _f1_multi(prediction: str, ground_truth: str) -> float:
    """Multi-answer F1 used by LoCoMo for category 1 (multi-hop):
    splits both prediction and gold on ',', then averages over gts the
    max f1_score across predictions."""
    preds = [p.strip() for p in prediction.split(',')]
    gts   = [g.strip() for g in ground_truth.split(',')]
    return float(np.mean([max(_f1_token(p, g) for p in preds) for g in gts]))

def locomo_score(prediction: str, gold: Any, category: int) -> float:
    """Per-category metric exactly as in
    task_eval/evaluation.py::eval_question_answering."""
    # Cat 5 (adversarial): no gold needed; check for the magic phrases.
    if category == 5:
        out = prediction.lower()
        return 1.0 if ('no information available' in out
                       or 'not mentioned' in out) else 0.0

    answer = str(gold)
    # Cat 3 (temporal): only the first ';'-separated answer is graded.
    if category == 3:
        answer = answer.split(';')[0].strip()

    if category == 1:                       # multi-hop -> multi-answer F1
        return _f1_multi(prediction, answer)
    elif category in (2, 3, 4):             # single-hop, temporal, open-domain -> F1
        return _f1_token(prediction, answer)
    else:
        raise ValueError(f"Unknown LoCoMo category: {category}")


# ---------- Retrieval recall (bonus metric) ----------
def retrieval_recall(retrieved_metas: list[dict[str, Any]], evidence_ids: list[str]) -> float:
    if not evidence_ids:
        return float("nan")
    retrieved_ids = {m["dia_id"] for m in retrieved_metas}
    return len(set(evidence_ids) & retrieved_ids) / len(evidence_ids)


# ---------- Eval loop ----------
def evaluate(samples: list[dict[str, Any]], n_conversations: int = 1) -> list[dict[str, Any]]:
    results = []
    for sample in samples[:n_conversations]:
        memory = SimpleMemory()
        ingest_conversation(memory, sample)

        for qa in tqdm(sample["qa"], desc=f"QA on {sample['sample_id']}"):
            q    = qa["question"]
            gold = qa.get("answer")
            cat  = qa.get("category")
            evidence = qa.get("evidence", []) or []

            # Cat 5 (adversarial) has no gold answer; all other categories require one.
            if cat != 5 and gold is None:
                continue

            out   = answer(memory, q)
            score = locomo_score(out["prediction"], gold, cat)
            rrec  = retrieval_recall(out["retrieved"], evidence)

            results.append({
                "sample_id":         sample["sample_id"],
                "question":          q,
                "gold":              gold,
                "pred":              out["prediction"],
                "category":          cat,
                "score":             score,
                "retrieval_recall":  rrec,
            })

    if not results:
        print("No results."); return results

    acc = float(np.mean([r["score"] for r in results]))
    rr  = float(np.nanmean([r["retrieval_recall"] for r in results]))
    print(f"\nF1:                {acc:.3f}  ({len(results)} questions)")
    print(f"Retrieval recall:  {rr:.3f}\n")

    cats: dict[Any, list[float]] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r["score"])
    for c, scores in sorted(cats.items(), key=lambda x: str(x[0])):
        print(f"  category {c}: {np.mean(scores):.3f}  (n={len(scores)})")

    return results


if __name__ == "__main__":
    data = json.loads(Path(LOCOMO_PATH).read_text())
    results = evaluate(data, n_conversations=1)  # start with 1, scale up later
    Path("results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} results to results.json")