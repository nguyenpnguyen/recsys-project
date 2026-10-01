"""Precompute strong lightweight news embeddings (BGE-small-en-v1.5) for MIND.

LLMEncCA and Supermodel read ``MindData.llm_emb``. V10 uses real MIND news and
precomputes one shared embedding from title + abstract.

    # sentence-transformers is expected to be available in the Kaggle image:
    python llm_embed.py --news /data/MINDsmall_train/news.tsv --out news_emb.npz \
        --model BAAI/bge-small-en-v1.5            # ~33M, very light (default)

Then in your own script:

    import numpy as np
    from data import MindData
    z = np.load("news_emb.npz", allow_pickle=True)
    emb = {nid: v for nid, v in zip(z["ids"], z["vecs"])}
    data = MindData.from_mind(train_dir, dev_dir, llm_embeddings=emb)

The V10 run fixes the encoder to BAAI/bge-small-en-v1.5.
"""
from __future__ import annotations

import argparse

import numpy as np


def read_titles(paths: list[str]) -> tuple[list[str], list[str]]:
    # Keep one deterministic row per news id when train and dev overlap.
    by_id = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                title, abstract = p[3], (p[4] if len(p) > 4 else "")
                by_id[p[0]] = (title + ". " + abstract).strip()
    ids = list(by_id)
    return ids, [by_id[nid] for nid in ids]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", nargs="+", required=True, help="one or more MIND news.tsv files")
    ap.add_argument("--out", default="news_emb.npz")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5",
                    help="V10 fixed encoder: BAAI/bge-small-en-v1.5")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args(argv)

    from sentence_transformers import SentenceTransformer

    ids, texts = read_titles(args.news)
    enc = SentenceTransformer(args.model, trust_remote_code=args.trust_remote_code)
    vecs = enc.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                      normalize_embeddings=True)
    np.savez(args.out, ids=np.array(ids), vecs=np.asarray(vecs, np.float32), model_name=np.array(args.model))
    print(f"Saved {len(ids)} embeddings of dim {vecs.shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()
