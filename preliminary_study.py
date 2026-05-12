"""Count retrieval frequency of each turn / session across LoCoMo."""

from __future__ import annotations
import json, os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

LOCOMO_PATH = os.getenv("LOCOMO_PATH", "locomo/data/locomo10.json")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
TOP_K       = 8

os.makedirs("./results/prelim", exist_ok=True)

embedder = SentenceTransformer(EMBED_MODEL)


class SimpleMemory:
    def __init__(self):
        self.metas: list[dict] = []
        self.vecs: np.ndarray | None = None

    def write(self, text, meta):
        self.metas.append(meta)
        v = embedder.encode([text], normalize_embeddings=True)
        self.vecs = v if self.vecs is None else np.vstack([self.vecs, v])

    def recall(self, query, k=TOP_K):
        q = embedder.encode([query], normalize_embeddings=True)[0]
        top = np.argsort(-(self.vecs @ q))[:k]
        return [self.metas[i] for i in top]


def ingest(memory, sample):
    conv = sample["conversation"]
    keys = sorted(
        (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
        key=lambda k: int(k.split("_")[1]),
    )
    for sk in keys:
        ts = conv.get(f"{sk}_date_time", "")
        for turn in conv[sk]:
            memory.write(f"[{ts}] {turn['speaker']}: {turn['text']}",
                         {"session": sk, "dia_id": turn["dia_id"]})


def main():
    data = json.loads(Path(LOCOMO_PATH).read_text())
    turn_hits, session_hits = Counter(), Counter()
    total_turns = total_questions = 0

    for sample in tqdm(data, desc="conversations"):
        sid = sample["sample_id"]
        memory = SimpleMemory()
        ingest(memory, sample)
        total_turns += len(memory.metas)
        for qa in sample["qa"]:
            for meta in memory.recall(qa["question"]):
                turn_hits[(sid, meta["dia_id"])]   += 1
                session_hits[(sid, meta["session"])] += 1
            total_questions += 1

    print(f"\n{total_questions} questions, {total_turns} turns, top-K={TOP_K}")
    print(f"Turns ever retrieved: {len(turn_hits)}/{total_turns} ({len(turn_hits)/total_turns:.1%})")
    print(f"Mean retrievals per turn: {sum(turn_hits.values())/total_turns:.2f}")

    _plot_session_hits(session_hits)
    _plot_turn_hits(turn_hits)
    _plot_all_session_hits(session_hits)
    _plot_all_turn_hits(turn_hits)


_TOP_N   = 10
_COLORS  = plt.cm.tab10.colors   # 10 distinct colors, one per rank
_GAP     = 2                      # blank slots between conversation groups


def _grouped_bar(ax, per_conv: dict[str, Counter], label_fn):
    """Draw a grouped bar chart: one group per conversation, bars = top-_TOP_N items."""
    group_size = _TOP_N + _GAP
    conversations = sorted(per_conv.keys())
    group_centers = []

    for i, sid in enumerate(conversations):
        top = per_conv[sid].most_common(_TOP_N)
        for j, (key, n) in enumerate(top):
            x = i * group_size + j
            ax.bar(x, n, color=_COLORS[j], width=0.85)
            ax.text(x, 0, label_fn(key),
                    ha="center", va="bottom", fontsize=5,
                    rotation=90, color="white", clip_on=True)
        group_centers.append(i * group_size + (len(top) - 1) / 2)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(conversations, rotation=30, ha="right", fontsize=8)
    ax.margins(x=0.01)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=_COLORS[j], label=f"Rank {j + 1}") for j in range(_TOP_N)]
    ax.legend(handles=legend, loc="upper right", fontsize=7, ncol=2, title="Rank within conv.")


def _plot_session_hits(session_hits: Counter):
    per_conv: dict[str, Counter] = defaultdict(Counter)
    for (sid, sk), n in session_hits.items():
        per_conv[sid][sk] = n

    fig, ax = plt.subplots(figsize=(24, 6))
    _grouped_bar(ax, per_conv, lambda sk: sk.replace("session_", "s"))
    ax.set_ylabel("Retrieval hits")
    ax.set_title(f"Top-{_TOP_N} session retrieval hits per conversation")
    fig.tight_layout()
    fig.savefig("./results/prelim/session_hits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved session_hits.png")


def _plot_turn_hits(turn_hits: Counter):
    counts = np.array(sorted(turn_hits.values(), reverse=True), dtype=float)
    ranks  = np.arange(1, len(counts) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(ranks, counts, ".", markersize=2)
    axes[0].set_xlabel("Rank"); axes[0].set_ylabel("Retrieval hits")
    axes[0].set_title("Linear–Linear")

    axes[1].plot(ranks, counts, ".", markersize=2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Rank (log)"); axes[1].set_ylabel("Retrieval hits")
    axes[1].set_title("Log–Linear")

    axes[2].plot(ranks, counts, ".", markersize=2)
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("Rank (log)"); axes[2].set_ylabel("Retrieval hits (log)")
    axes[2].set_title("Log–Log")

    for ax in axes:
        ax.grid(True, which="both", ls="--", alpha=0.35)

    fig.suptitle(f"Turn retrieval hits — Rank–Frequency (N={len(counts)} turns)", y=1.02)
    fig.tight_layout()
    fig.savefig("./results/prelim/turn_hits_zipf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved turn_hits_zipf.png")


def _plot_all_session_hits(session_hits: Counter):
    per_conv: dict[str, Counter] = defaultdict(Counter)
    for (sid, sk), n in session_hits.items():
        per_conv[sid][sk] = n

    out_dir = Path("./results/prelim/session_hits")
    out_dir.mkdir(exist_ok=True)

    for sid, counter in per_conv.items():
        ordered = counter.most_common()
        labels = [sk.replace("session_", "s") for sk, _ in ordered]
        counts = [n for _, n in ordered]

        fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.7), 4))
        ax.bar(range(len(labels)), counts, color=_COLORS[:len(labels)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Retrieval hits")
        ax.set_title(f"Session hits — {sid}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{sid}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(per_conv)} figures → {out_dir}/")


def _plot_all_turn_hits(turn_hits: Counter):
    per_conv: dict[str, Counter] = defaultdict(Counter)
    for (sid, dia_id), n in turn_hits.items():
        per_conv[sid][dia_id] = n

    out_dir = Path("./results/prelim/turn_hits")
    out_dir.mkdir(exist_ok=True)

    for sid, counter in per_conv.items():
        counts = np.array(sorted(counter.values(), reverse=True), dtype=float)
        ranks  = np.arange(1, len(counts) + 1)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.loglog(ranks, counts, "o-", markersize=4, linewidth=1)
        ax.set_xlabel("Rank"); ax.set_ylabel("Retrieval hits")
        ax.set_title(f"Turn hits — {sid} (log–log)")
        ax.grid(True, which="both", ls="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / f"{sid}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(per_conv)} figures → {out_dir}/")


if __name__ == "__main__":
    main()