"""Self-contained MIND-small data reader with chronological validation split."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch

PAD = 0


@dataclass
class Batch:
    hist_title: torch.Tensor
    hist_cat: torch.Tensor
    hist_idx: torch.Tensor
    hist_mask: torch.Tensor
    cand_title: torch.Tensor
    cand_cat: torch.Tensor
    cand_idx: torch.Tensor
    cand_mask: torch.Tensor
    user_idx: torch.Tensor
    labels: torch.Tensor

    def to(self, device):
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name).to(device))
        return self


@dataclass
class MindData:
    news_title: np.ndarray
    news_cat: np.ndarray
    n_news: int
    n_users: int
    vocab_size: int
    n_cat: int
    title_len: int
    max_hist: int
    llm_emb: np.ndarray
    train_core: list = field(default_factory=list)
    validation: list = field(default_factory=list)
    test: list = field(default_factory=list)
    train_core_impressions: int = 0
    validation_impressions: int = 0
    test_impressions: int = 0
    train_days: list[str] = field(default_factory=list)
    validation_days: list[str] = field(default_factory=list)
    test_days: list[str] = field(default_factory=list)
    split_info: dict = field(default_factory=dict)
    _edges: list = field(default_factory=list)

    def _pad_hist(self, hist):
        hist = list(hist)[-self.max_hist:]
        mask = [True] * len(hist)
        while len(hist) < self.max_hist:
            hist.append(PAD)
            mask.append(False)
        return hist, mask

    def _gather(self, idxs):
        return self.news_title[idxs], self.news_cat[idxs]

    def collate_train(self, rows):
        H, L = self.max_hist, self.title_len
        C, B = 1 + len(rows[0][3]), len(rows)
        ht = np.zeros((B, H, L), np.int64)
        hc = np.zeros((B, H), np.int64)
        hi = np.zeros((B, H), np.int64)
        hm = np.zeros((B, H), bool)
        ct = np.zeros((B, C, L), np.int64)
        cc = np.zeros((B, C), np.int64)
        ci = np.zeros((B, C), np.int64)
        ui = np.zeros(B, np.int64)
        for b, (user, hist, pos, negs) in enumerate(rows):
            h, m = self._pad_hist(hist)
            hi[b], hm[b] = h, m
            ht[b], hc[b] = self._gather(h)
            cands = [pos] + list(negs)
            ci[b], ct[b], cc[b] = cands, *self._gather(cands)
            ui[b] = user
        return _to_batch(ht, hc, hi, hm, ct, cc, ci,
                         np.ones((B, C), bool), ui,
                         np.pad(np.ones((B, 1), np.float32), ((0, 0), (0, C - 1))))

    def collate_eval(self, rows):
        H, L = self.max_hist, self.title_len
        C, B = max(len(_eval_parts(r)[3]) for r in rows), len(rows)
        ht = np.zeros((B, H, L), np.int64)
        hc = np.zeros((B, H), np.int64)
        hi = np.zeros((B, H), np.int64)
        hm = np.zeros((B, H), bool)
        ct = np.zeros((B, C, L), np.int64)
        cc = np.zeros((B, C), np.int64)
        ci = np.zeros((B, C), np.int64)
        cm = np.zeros((B, C), bool)
        labels = np.zeros((B, C), np.float32)
        ui = np.zeros(B, np.int64)
        for b, row in enumerate(rows):
            _, user, hist, cands, labs = _eval_parts(row)
            h, m = self._pad_hist(hist)
            hi[b], hm[b] = h, m
            ht[b], hc[b] = self._gather(h)
            n = len(cands)
            ci[b, :n], cm[b, :n], labels[b, :n] = cands, True, labs
            if n:
                ct[b, :n], cc[b, :n] = self._gather(cands)
            ui[b] = user
        return _to_batch(ht, hc, hi, hm, ct, cc, ci, cm, ui, labels)

    def build_adjacency(self):
        U, N = self.n_users, self.n_news
        rows, cols = [], []
        for u, n in self._edges:
            rows += [u, U + n]
            cols += [U + n, u]
        if not rows:
            rows, cols = [0], [0]
        idx = np.array([rows, cols], dtype=np.int64)
        vals = np.ones(idx.shape[1], np.float32)
        size = U + N
        deg = np.zeros(size, np.float64)
        np.add.at(deg, idx[0], vals)
        inv = np.zeros(size, np.float64)
        inv[deg > 0] = deg[deg > 0] ** -0.5
        norm = (inv[idx[0]] * inv[idx[1]]).astype(np.float32)
        return torch.sparse_coo_tensor(torch.from_numpy(idx), torch.from_numpy(norm), (size, size)).coalesce()

    @classmethod
    def from_mind(cls, train_dir, dev_dir, title_len=20, max_hist=50,
                  n_neg=4, min_word_freq=3, llm_dim=64,
                  llm_embeddings=None, dev_ratio=0.1, seed=0):
        import os
        from collections import Counter

        rng = np.random.default_rng(seed)

        def read_news(path):
            out = {}
            with open(path, encoding="utf-8") as f:
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if len(p) >= 5:
                        out[p[0]] = (p[1], p[3], p[4])
            return out

        news_rows = read_news(os.path.join(train_dir, "news.tsv"))
        if dev_dir and os.path.exists(os.path.join(dev_dir, "news.tsv")):
            news_rows.update(read_news(os.path.join(dev_dir, "news.tsv")))

        wf = Counter()
        for _, title, abstract in news_rows.values():
            wf.update((title + " " + abstract).lower().split())
        vocab = {"<pad>": PAD}
        for word, count in wf.items():
            if count >= min_word_freq:
                vocab[word] = len(vocab)
        cats = sorted({cat for cat, _, _ in news_rows.values()})
        cat_map = {cat: i + 1 for i, cat in enumerate(cats)}
        nid_map = {"<pad>": PAD}
        for nid in news_rows:
            nid_map[nid] = len(nid_map)
        n_news = len(nid_map)
        news_title = np.zeros((n_news, title_len), np.int64)
        news_cat = np.zeros(n_news, np.int64)
        for nid, (cat, title, abstract) in news_rows.items():
            words = (title + " " + abstract).lower().split()[:title_len]
            gi = nid_map[nid]
            news_title[gi, :len(words)] = [vocab.get(w, PAD) for w in words]
            news_cat[gi] = cat_map[cat]

        if llm_embeddings is None:
            llm_emb = np.zeros((n_news, llm_dim), np.float32)
        else:
            if not llm_embeddings:
                raise ValueError("empty BGE embedding map")
            dims = {np.asarray(v).reshape(-1).size for v in llm_embeddings.values()}
            if len(dims) != 1:
                raise ValueError("inconsistent embedding dimensions")
            dim = dims.pop()
            llm_emb = np.zeros((n_news, dim), np.float32)
            for nid, gi in nid_map.items():
                if nid != "<pad>" and nid in llm_embeddings:
                    llm_emb[gi] = np.asarray(llm_embeddings[nid], np.float32).reshape(-1)
            coverage = float((np.linalg.norm(llm_emb[1:], axis=1) > 0).mean())
            if coverage < 0.999 or not np.isfinite(llm_emb).all():
                raise ValueError(f"invalid BGE coverage={coverage:.4%}")
            print(f"  embedding coverage: {coverage:.3%} (PAD excluded)")
        llm_emb = (llm_emb / (np.linalg.norm(llm_emb, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

        uid_map = {}
        def uidx(uid):
            if uid not in uid_map:
                uid_map[uid] = len(uid_map)
            return uid_map[uid]

        def day_of(value):
            value = value.strip()
            for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(value, fmt).date().isoformat()
                except ValueError:
                    pass
            return value.split(" ")[0]

        def read_behaviors(path):
            out = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if len(p) < 5:
                        continue
                    impr_id, user_raw, timestamp, history_raw, impressions = p[:5]
                    hist = [nid_map[n] for n in history_raw.split() if n in nid_map] if history_raw else []
                    cands, labs = [], []
                    for item in impressions.split():
                        nid, lab = item.rsplit("-", 1)
                        if nid in nid_map:
                            cands.append(nid_map[nid])
                            labs.append(int(lab))
                    out.append((str(impr_id), uidx(user_raw), hist, cands, labs, day_of(timestamp)))
            return out

        train_raw = read_behaviors(os.path.join(train_dir, "behaviors.tsv"))
        if not dev_dir or os.path.abspath(dev_dir) == os.path.abspath(train_dir):
            raise ValueError("V10 requires separate MINDsmall_train and MINDsmall_dev")
        test_raw = read_behaviors(os.path.join(dev_dir, "behaviors.tsv"))
        if not train_raw or not test_raw:
            raise ValueError("MIND behaviors files are empty")

        days = sorted({row[5] for row in train_raw})
        target = max(1, int(np.ceil(len(train_raw) * dev_ratio)))
        val_days, count = [], 0
        for day in reversed(days):
            val_days.append(day)
            count += sum(row[5] == day for row in train_raw)
            if count >= target:
                break
        val_days = sorted(val_days)
        train_days = [d for d in days if d not in set(val_days)]
        if not train_days:
            raise ValueError("day split leaves no train_core day")
        train_raw_core = [r for r in train_raw if r[5] not in set(val_days)]
        val_raw = [r for r in train_raw if r[5] in set(val_days)]
        train_core_beh = [(u, h, c, labs) for _, u, h, c, labs, _ in train_raw_core]
        validation = [(i, u, h, c, labs) for i, u, h, c, labs, _ in val_raw]
        test = [(i, u, h, c, labs) for i, u, h, c, labs, _ in test_raw]
        if max(r[5] for r in train_raw_core) >= min(r[5] for r in val_raw):
            raise AssertionError("chronological train/validation ordering violated")

        train_core, edges = [], []
        for user, hist, cands, labs in train_core_beh:
            edges.extend((user, n) for n in hist)
            pos = [c for c, lab in zip(cands, labs) if lab == 1]
            neg = [c for c, lab in zip(cands, labs) if lab == 0]
            if not pos or not neg:
                continue
            for p in pos:
                sampled = list(rng.choice(neg, size=n_neg, replace=len(neg) < n_neg))
                train_core.append((user, hist, p, sampled))
                edges.append((user, p))
        split_info = {
            "train_days": train_days,
            "validation_days": val_days,
            "test_days": sorted({r[5] for r in test_raw}),
            "train_day_min": min(train_days), "train_day_max": max(train_days),
            "validation_day_min": min(val_days), "validation_day_max": max(val_days),
            "target_validation_ratio": float(dev_ratio),
            "actual_validation_ratio": len(validation) / len(train_raw),
            "raw_train_impressions": len(train_raw),
            "train_core_impressions": len(train_core_beh),
            "validation_impressions": len(validation),
            "test_impressions": len(test),
        }
        print(f"  chronological split: {len(train_core_beh):,} train_core / "
              f"{len(validation):,} validation / {len(test):,} official test impressions")
        return cls(news_title, news_cat, n_news, len(uid_map), len(vocab), len(cat_map) + 1,
                   title_len, max_hist, llm_emb, train_core, validation, test,
                   len(train_core_beh), len(validation), len(test), train_days, val_days,
                   split_info["test_days"], split_info, edges)


def _eval_parts(row):
    if len(row) == 5:
        return row
    user, hist, cands, labs = row
    return "", user, hist, cands, labs


def _to_batch(*arrays):
    ht, hc, hi, hm, ct, cc, ci, cm, ui, labels = arrays
    return Batch(*(torch.from_numpy(x) for x in (ht, hc, hi, hm, ct, cc, ci, cm, ui, labels)))


def iter_batches(rows, batch_size, collate, shuffle=False, seed=0):
    order = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield collate([rows[i] for i in order[start:start + batch_size]])
