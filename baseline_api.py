"""
LoCoMo baseline evaluation using commercial API models (vector-retrieval memory).

Usage:
    python baseline_api.py --model claude-haiku-4-5
    python baseline_api.py --model gpt-4.1-mini
    python baseline_api.py --model deepseek-v4-flash
    python baseline_api.py --model kimi-k2.5

Environment variables:
    ANTHROPIC_API_KEY           for claude-haiku-4-5
    OPENAI_API_KEY              for gpt-4.1-mini
    OPENAI_BASE_URL             optional Azure endpoint override for gpt-4.1-mini
    DEEPSEEK_API_KEY            for deepseek-v4-flash
    DEEPSEEK_BASE_URL           base URL for Deepseek endpoint
    KIMI_API_KEY                for kimi-k2.5
    KIMI_BASE_URL               base URL for Kimi endpoint
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv(override=True)


# ---------- Model configs ----------
# model_id is the actual API model name / Azure deployment name.
# Adjust model_id entries if your Azure deployment names differ.
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "claude-haiku-4-5": {
        "provider":    "anthropic",
        "model_id":    "claude-haiku-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url_env": "ANTHROPIC_BASE_URL"
    },
    "gpt-5-mini": {
        "provider":     "openai",
        "model_id":     "gpt-5-mini",
        "api_key_env":  "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "api_version":  "OPENAI_API_VERSION"
    },
    "deepseek-v4-flash": {
        "provider":    "openai",
        "model_id":    "DeepSeek-V4-Flash",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
    },
    "kimi-k2.5": {
        "provider":    "openai",
        "model_id":    "Kimi-K2.5",
        "api_key_env": "KIMI_API_KEY",
        "base_url_env": "KIMI_BASE_URL",
    },
}

TOP_K      = 14
MAX_TOKENS = 256

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoCoMo baseline eval with commercial APIs")
    parser.add_argument(
        "--model", required=True, choices=list(MODEL_CONFIGS),
        help="Model to use for generation",
    )
    parser.add_argument(
        "--locomo-path", default="locomo/data/locomo10.json",
        help="Path to LoCoMo JSON dataset",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: results_baseline_<model>.json)",
    )
    return parser.parse_args()


# ---------- API client ----------
def build_client(cfg: dict[str, Any]) -> Any:
    api_key  = os.environ[cfg["api_key_env"]]
    base_url = os.environ[cfg["base_url_env"]]
    model_id = cfg["model_id"]
    
    if cfg["provider"] == "anthropic":
        import anthropic
        return anthropic.AnthropicFoundry(api_key=api_key, base_url=base_url)
    else:
        import openai
        if model_id == "gpt-5-mini":
            return openai.AzureOpenAI(api_key=api_key, azure_endpoint=base_url, api_version=os.environ[cfg["api_version"]])
        else:
            return openai.OpenAI(api_key=api_key, base_url=base_url)


def call_api(
    client: Any,
    provider: str,
    model_id: str,
    messages: list[dict[str, str]],
    max_retries: int = 6,
) -> str:
    for attempt in range(max_retries):
        try:
            if provider == "anthropic":
                system    = next((m["content"] for m in messages if m["role"] == "system"), "")
                user_msgs = [m for m in messages if m["role"] != "system"]
                resp = client.messages.create(
                    model=model_id,
                    max_tokens=MAX_TOKENS,
                    system=system,
                    messages=user_msgs,
                )
                return resp.content[0].text.strip()
            else:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_completion_tokens=MAX_TOKENS,
                )
                return resp.choices[0].message.content.strip()
        except Exception as exc:
            if attempt == max_retries - 1:
                return ""
            wait = 2 ** attempt
            print(f"\n[retry {attempt+1}/{max_retries}] {exc}  — sleeping {wait}s")
            time.sleep(wait)
    
    return ""


# ---------- Memory: minimal vector store ----------
class SimpleMemory:
    def __init__(self, embedder: SentenceTransformer) -> None:
        self.embedder = embedder
        self.texts: list[str] = []
        self.metas: list[dict[str, Any]] = []
        self.vecs: np.ndarray | None = None

    def write(self, text: str, meta: dict[str, Any]) -> None:
        self.texts.append(text)
        self.metas.append(meta)
        v = self.embedder.encode([text], normalize_embeddings=True)
        self.vecs = v if self.vecs is None else np.vstack([self.vecs, v])

    def recall(self, query: str, k: int = TOP_K) -> list[dict[str, Any]]:
        if self.vecs is None or not self.texts:
            return []
        q      = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.vecs @ q
        top    = np.argsort(-scores)[:k]
        return [{"text": self.texts[i], "meta": self.metas[i], "score": float(scores[i])}
                for i in top]


# ---------- Conversation -> memory ----------
def ingest_conversation(memory: SimpleMemory, sample: dict[str, Any]) -> None:
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


def answer(
    memory: SimpleMemory,
    question: str,
    client: Any,
    provider: str,
    model_id: str,
) -> dict[str, Any]:
    hits    = memory.recall(question, k=TOP_K)
    context = "\n".join(f"- {h['text']}" for h in hits)
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user",   "content": f"Memory excerpts:\n{context}\n\nQuestion: {question}"},
    ]
    prediction = call_api(client, provider, model_id, messages)
    return {"prediction": prediction, "retrieved": [h["meta"] for h in hits]}


# ---------- LoCoMo official scoring ----------
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
    common    = Counter(pred_toks) & Counter(gt_toks)
    num_same  = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall    = num_same / len(gt_toks)
    return (2 * precision * recall) / (precision + recall)

def _f1_multi(prediction: str, ground_truth: str) -> float:
    preds = [p.strip() for p in prediction.split(',')]
    gts   = [g.strip() for g in ground_truth.split(',')]
    return float(np.mean([max(_f1_token(p, g) for p in preds) for g in gts]))

def locomo_score(prediction: str, gold: Any, category: int) -> float:
    if category == 5:
        out = prediction.lower()
        return 1.0 if ('no information available' in out
                       or 'not mentioned' in out) else 0.0

    ans = str(gold)
    if category == 3:
        ans = ans.split(';')[0].strip()

    if category == 1:
        return _f1_multi(prediction, ans)
    elif category in (2, 3, 4):
        return _f1_token(prediction, ans)
    else:
        raise ValueError(f"Unknown LoCoMo category: {category}")


# ---------- Retrieval recall ----------
def retrieval_recall(retrieved_metas: list[dict[str, Any]], evidence_ids: list[str]) -> float:
    if not evidence_ids:
        return float("nan")
    retrieved_ids = {m["dia_id"] for m in retrieved_metas}
    return len(set(evidence_ids) & retrieved_ids) / len(evidence_ids)


# ---------- Eval loop ----------
def evaluate(
    samples: list[dict[str, Any]],
    embedder: SentenceTransformer,
    client: Any,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    provider = cfg["provider"]
    model_id = cfg["model_id"]
    results  = []

    for sample in samples:
        memory = SimpleMemory(embedder)
        ingest_conversation(memory, sample)

        for qa in tqdm(sample["qa"], desc=f"QA on {sample['sample_id']}"):
            q        = qa["question"]
            gold     = qa.get("answer")
            cat      = qa.get("category")
            evidence = qa.get("evidence", []) or []

            if cat != 5 and gold is None:
                continue

            out   = answer(memory, q, client, provider, model_id)
            score = locomo_score(out["prediction"], gold, cat)
            rrec  = retrieval_recall(out["retrieved"], evidence)

            results.append({
                "sample_id":        sample["sample_id"],
                "question":         q,
                "gold":             gold,
                "pred":             out["prediction"],
                "category":         cat,
                "score":            score,
                "retrieval_recall": rrec,
            })

    if not results:
        print("No results.")
        return results

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
    args    = parse_args()
    cfg     = MODEL_CONFIGS[args.model]
    output  = args.output or f"results_baseline_14_{args.model}.json"

    print(f"Model:   {args.model}  ({cfg['model_id']})")
    print(f"Dataset: {args.locomo_path}")
    print(f"Output:  {output}\n")

    embedder = SentenceTransformer(EMBED_MODEL)
    client   = build_client(cfg)

    data    = json.loads(Path(args.locomo_path).read_text())
    results = evaluate(data, embedder, client, cfg)

    Path(output).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} results to {output}")
