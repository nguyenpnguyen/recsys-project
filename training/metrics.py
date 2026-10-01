"""Ranking metrics for news recommendation, computed per-impression.

Pure NumPy so there is no dependency beyond numpy. Definitions match the
Microsoft Recommenders / MIND leaderboard conventions (MRR, nDCG@k) plus a
tie-aware AUC.
"""
from __future__ import annotations

import numpy as np


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Area under ROC curve for one impression (tie-aware, rank based).

    Returns None if the impression has no positive or no negative (undefined),
    so the caller can skip it in the mean.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    # Average ranks (1..n), ties share the mean rank -> Mann-Whitney U form.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg = (start + cum + 1) / 2.0  # average 1-based rank per distinct value
    ranks = avg[inv]
    sum_ranks_pos = ranks[labels == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mrr_score(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    y = np.asarray(labels)[order]
    rr = y / (np.arange(len(y)) + 1)
    denom = y.sum()
    return float(rr.sum() / denom) if denom > 0 else 0.0


def _dcg(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = (2 ** np.asarray(labels)[order] - 1).astype(np.float64)
    discounts = np.log2(np.arange(len(gains)) + 2)
    return float((gains / discounts).sum())


def ndcg_score(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    labels = np.asarray(labels)
    ideal = _dcg(labels, labels, k)
    if ideal == 0:
        return 0.0
    return _dcg(labels, scores, k) / ideal


def aggregate(impressions: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Mean metrics over a list of (labels, scores) impressions."""
    aucs, mrrs, n5, n10 = [], [], [], []
    for labels, scores in impressions:
        a = auc_score(labels, scores)
        if a is not None:
            aucs.append(a)
        mrrs.append(mrr_score(labels, scores))
        n5.append(ndcg_score(labels, scores, 5))
        n10.append(ndcg_score(labels, scores, 10))
    return {
        "auc": float(np.mean(aucs)) if aucs else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "ndcg@5": float(np.mean(n5)) if n5 else 0.0,
        "ndcg@10": float(np.mean(n10)) if n10 else 0.0,
    }
