"""Small news recommendation model zoo used by V10."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import PAD, Batch


class AdditiveAttention(nn.Module):
    def __init__(self, dim, hidden=200):
        super().__init__()
        self.proj, self.query = nn.Linear(dim, hidden), nn.Linear(hidden, 1, bias=False)

    def forward(self, x, mask=None):
        a = self.query(torch.tanh(self.proj(x))).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~mask, -1e9)
        return torch.einsum("...n,...nd->...d", torch.softmax(a, -1), x)


class FastformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh = heads, dim // heads
        self.Wq = nn.Linear(dim, dim); self.Wk = nn.Linear(dim, dim); self.Wv = nn.Linear(dim, dim)
        self.q_att = nn.Linear(self.dh, 1); self.k_att = nn.Linear(self.dh, 1)
        self.o = nn.Linear(dim, dim); self.norm = nn.LayerNorm(dim); self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        def split(t): return t.view(B, N, self.h, self.dh)
        q, k, v = split(self.Wq(x)), split(self.Wk(x)), split(self.Wv(x))
        if mask is None: mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        wq = self.q_att(torch.tanh(q)).squeeze(-1).masked_fill(~mask[:, :, None], -1e9)
        qg = (torch.softmax(wq, 1).unsqueeze(-1) * q).sum(1, keepdim=True)
        u = q * qg
        wk = self.k_att(torch.tanh(k)).squeeze(-1).masked_fill(~mask[:, :, None], -1e9)
        kg = (torch.softmax(wk, 1).unsqueeze(-1) * k).sum(1, keepdim=True)
        out = self.o((u * kg).reshape(B, N, D))
        return self.norm(x + self.drop(out))


class SelfAttnNewsEncoder(nn.Module):
    def __init__(self, d, cfg):
        super().__init__(); D = cfg.dim
        self.word_emb = nn.Embedding(d.vocab_size, D, padding_idx=PAD)
        self.mha = nn.MultiheadAttention(D, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.pool, self.drop = AdditiveAttention(D), nn.Dropout(cfg.dropout)

    def forward(self, title):
        e = self.drop(self.word_emb(title)); o, _ = self.mha(e, e, e)
        return self.pool(self.drop(o))


def _safe_kpm(mask):
    kpm = ~mask
    return kpm.masked_fill(kpm.all(dim=1, keepdim=True), False)


class NRMS(nn.Module):
    def __init__(self, d, cfg):
        super().__init__(); D = cfg.dim
        self.news = SelfAttnNewsEncoder(d, cfg)
        self.user_mha = nn.MultiheadAttention(D, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.user_pool = AdditiveAttention(D)

    def score(self, b):
        B, H, L = b.hist_title.shape; C = b.cand_title.shape[1]
        hist = self.news(b.hist_title.reshape(B * H, L)).view(B, H, -1)
        cand = self.news(b.cand_title.reshape(B * C, L)).view(B, C, -1)
        o, _ = self.user_mha(hist, hist, hist, key_padding_mask=_safe_kpm(b.hist_mask))
        return torch.einsum("bd,bcd->bc", self.user_pool(o, b.hist_mask), cand)


class NAML(nn.Module):
    def __init__(self, d, cfg):
        super().__init__(); D = cfg.dim
        self.word_emb = nn.Embedding(d.vocab_size, D, padding_idx=PAD)
        self.cat_emb = nn.Embedding(d.n_cat, D, padding_idx=PAD)
        self.conv = nn.Conv1d(D, D, 3, padding=1); self.title_pool = AdditiveAttention(D)
        self.cat_dense = nn.Linear(D, D); self.view_pool = AdditiveAttention(D); self.user_pool = AdditiveAttention(D)
        self.drop = nn.Dropout(cfg.dropout)

    def _news(self, title, cat):
        e = self.drop(self.word_emb(title)).transpose(1, 2)
        title_vec = self.title_pool(self.drop(torch.relu(self.conv(e)).transpose(1, 2)))
        cat_vec = torch.relu(self.cat_dense(self.cat_emb(cat)))
        return self.view_pool(torch.stack([title_vec, cat_vec], 1))

    def score(self, b):
        B, H, L = b.hist_title.shape; C = b.cand_title.shape[1]
        hist = self._news(b.hist_title.reshape(B * H, L), b.hist_cat.reshape(B * H)).view(B, H, -1)
        cand = self._news(b.cand_title.reshape(B * C, L), b.cand_cat.reshape(B * C)).view(B, C, -1)
        return torch.einsum("bd,bcd->bc", self.user_pool(hist, b.hist_mask), cand)


class Fastformer(nn.Module):
    def __init__(self, d, cfg):
        super().__init__(); D = cfg.dim
        self.word_emb = nn.Embedding(d.vocab_size, D, padding_idx=PAD)
        self.news_ff, self.user_ff = FastformerBlock(D, cfg.heads, cfg.dropout), FastformerBlock(D, cfg.heads, cfg.dropout)
        self.news_pool, self.user_pool = AdditiveAttention(D), AdditiveAttention(D)
        self.drop = nn.Dropout(cfg.dropout)

    def _news(self, title):
        return self.news_pool(self.news_ff(self.drop(self.word_emb(title))))

    def score(self, b):
        B, H, L = b.hist_title.shape; C = b.cand_title.shape[1]
        hist = self._news(b.hist_title.reshape(B * H, L)).view(B, H, -1)
        cand = self._news(b.cand_title.reshape(B * C, L)).view(B, C, -1)
        return torch.einsum("bd,bcd->bc", self.user_pool(self.user_ff(hist, b.hist_mask), b.hist_mask), cand)


class LightGCN(nn.Module):
    def __init__(self, d, cfg):
        super().__init__(); D = cfg.dim; self.n_users, self.layers = d.n_users, cfg.gcn_layers
        self.user_emb, self.news_emb = nn.Embedding(d.n_users, D), nn.Embedding(d.n_news, D)
        self.register_buffer("adj", d.build_adjacency())

    def _propagate(self):
        x = torch.cat([self.user_emb.weight, self.news_emb.weight]); agg = x
        for _ in range(self.layers): x = torch.sparse.mm(self.adj, x); agg = agg + x
        agg /= self.layers + 1
        return agg[:self.n_users], agg[self.n_users:]

    def score(self, b):
        u, n = self._propagate()
        return torch.einsum("bd,bcd->bc", u[b.user_idx], n[b.cand_idx])


class LLMNewsEncoder(nn.Module):
    def __init__(self, d, cfg):
        super().__init__(); emb = torch.from_numpy(d.llm_emb)
        self.register_buffer("emb", emb)
        self.head = nn.Sequential(nn.Linear(emb.shape[1], cfg.dim), nn.GELU(), nn.LayerNorm(cfg.dim), nn.Dropout(cfg.dropout), nn.Linear(cfg.dim, cfg.dim))

    def forward(self, idx): return self.head(self.emb[idx])


class LLMEncCA(nn.Module):
    def __init__(self, d, cfg):
        super().__init__(); self.news = LLMNewsEncoder(d, cfg); self.cand_proj = nn.Linear(cfg.dim, cfg.dim, bias=False); self.scale = cfg.dim ** 0.5

    def score(self, b):
        hist, cand = self.news(b.hist_idx), self.news(b.cand_idx)
        att = torch.einsum("bcd,bhd->bch", self.cand_proj(cand), hist) / self.scale
        att = torch.softmax(att.masked_fill(~b.hist_mask[:, None, :], -1e9), -1)
        user = torch.einsum("bch,bhd->bcd", att, hist)
        return (user * cand).sum(-1)


def _znorm(x, mask):
    """Standardise scores within each impression, using only real candidates."""
    n = mask.sum(-1, keepdim=True).clamp(min=1)
    mu = (x * mask).sum(-1, keepdim=True) / n
    sd = (((x - mu) * mask) ** 2).sum(-1, keepdim=True).div(n).sqrt()
    return (x - mu) / (sd + 1e-6)


class Popularity(nn.Module):
    """No training: online smoothed log-CTR of each candidate (non-personalised)."""
    def __init__(self, d, cfg): super().__init__()

    def score(self, b): return b.cand_pop


class BGEZeroShot(nn.Module):
    """No training: dot product of candidate BGE with the mean BGE of the clicked history."""
    def __init__(self, d, cfg):
        super().__init__(); self.register_buffer("emb", torch.from_numpy(d.llm_emb), persistent=False)

    def score(self, b):
        m = b.hist_mask.unsqueeze(-1).float()
        user = (self.emb[b.hist_idx] * m).sum(1) / m.sum(1).clamp(min=1)
        return torch.einsum("bd,bcd->bc", user, self.emb[b.cand_idx])


class BGEZeroShotPop(BGEZeroShot):
    """No training: equal-weight sum of standardised BGE zero-shot and popularity scores."""
    def score(self, b):
        m = b.cand_mask.float()
        return _znorm(super().score(b), m) + _znorm(b.cand_pop, m)


class CAUM(nn.Module):
    """Candidate-aware user attention baseline."""
    def __init__(self, d, cfg):
        super().__init__(); D = cfg.dim
        self.news = SelfAttnNewsEncoder(d, cfg)
        self.hist_mha = nn.MultiheadAttention(D, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.inter = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, 1))

    def score(self, b):
        B, H, L = b.hist_title.shape; C = b.cand_title.shape[1]
        hist = self.news(b.hist_title.reshape(B * H, L)).view(B, H, -1)
        cand = self.news(b.cand_title.reshape(B * C, L)).view(B, C, -1)
        h, _ = self.hist_mha(hist, hist, hist, key_padding_mask=_safe_kpm(b.hist_mask))
        hh = h[:, None, :, :].expand(B, C, H, -1); cc = cand[:, :, None, :].expand(B, C, H, -1)
        a = self.inter(torch.cat([hh, cc], -1)).squeeze(-1).masked_fill(~b.hist_mask[:, None, :], -1e9)
        user = torch.einsum("bch,bchd->bcd", torch.softmax(a, -1), hh)
        return (user * cand).sum(-1)


class DiffAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__(); self.h, self.dh = heads, dim // heads
        self.q, self.k, self.v, self.o = nn.Linear(dim, 2 * dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.lq1, self.lk1 = nn.Parameter(torch.randn(self.dh) * .1), nn.Parameter(torch.randn(self.dh) * .1)
        self.lq2, self.lk2 = nn.Parameter(torch.randn(self.dh) * .1), nn.Parameter(torch.randn(self.dh) * .1)
        self.norm, self.drop = nn.LayerNorm(dim), nn.Dropout(dropout)

    def forward(self, x, mask):
        B, N, D = x.shape
        q = self.q(x).view(B, N, 2, self.h, self.dh); k = self.k(x).view(B, N, 2, self.h, self.dh); v = self.v(x).view(B, N, self.h, self.dh)
        def att(qi, ki):
            s = torch.einsum("bnhd,bmhd->bhnm", qi, ki) / (self.dh ** .5)
            return torch.softmax(s.masked_fill(~mask[:, None, None, :], -1e9), -1)
        a1, a2 = att(q[:, :, 0], k[:, :, 0]), att(q[:, :, 1], k[:, :, 1])
        lam = torch.exp((self.lq1 * self.lk1).sum()) - torch.exp((self.lq2 * self.lk2).sum()) + .8
        out = self.o(torch.einsum("bhnm,bmhd->bnhd", a1 - lam * a2, v).reshape(B, N, D))
        return self.norm(x + self.drop(out))


class GraphProp(nn.Module):
    def __init__(self, d, cfg):
        super().__init__(); self.n_users, self.layers = d.n_users, cfg.gcn_layers
        self.user_emb, self.news_emb = nn.Embedding(d.n_users, cfg.dim), nn.Embedding(d.n_news, cfg.dim)
        self.register_buffer("adj", d.build_adjacency())

    def forward(self):
        x = torch.cat([self.user_emb.weight, self.news_emb.weight]); agg = x
        for _ in range(self.layers): x = torch.sparse.mm(self.adj, x); agg = agg + x
        agg /= self.layers + 1
        return agg[:self.n_users], agg[self.n_users:]


class SuperRec(nn.Module):
    variant = "supermodel"
    def __init__(self, d, cfg):
        super().__init__(); self.variant = type(self).variant
        use_bge = self.variant != "no_bge"
        if use_bge: self.news = LLMNewsEncoder(d, cfg)
        else: self.news = SelfAttnNewsEncoder(d, cfg)
        if self.variant == "no_diff":
            self.hist_mha = nn.MultiheadAttention(cfg.dim, cfg.heads, dropout=cfg.dropout, batch_first=True)
            self.hist_norm, self.hist_drop = nn.LayerNorm(cfg.dim), nn.Dropout(cfg.dropout)
        else: self.hist_diff = DiffAttention(cfg.dim, cfg.heads, cfg.dropout)
        self.cand_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        if self.variant != "no_graph": self.graph, self.gate = GraphProp(d, cfg), nn.Parameter(torch.tensor(-4.0))
        if self.variant == "no_ca": self.no_ca_pool = AdditiveAttention(cfg.dim)
        if self.variant != "no_cl": self.user_pool, self.drop = AdditiveAttention(cfg.dim), nn.Dropout(cfg.dropout)
        self.tau, self.cl_weight, self.scale = cfg.cl_tau, cfg.cl_weight, cfg.dim ** .5

    def _encode(self, b):
        B, H, L = b.hist_title.shape; C = b.cand_title.shape[1]
        if self.variant == "no_bge":
            hist = self.news(b.hist_title.reshape(B * H, L)).view(B, H, -1)
            cand = self.news(b.cand_title.reshape(B * C, L)).view(B, C, -1)
        else:
            hist, cand = self.news(b.hist_idx), self.news(b.cand_idx)
        return hist, cand

    def score(self, b):
        hist, cand = self._encode(b)
        if self.variant == "no_diff":
            h, _ = self.hist_mha(hist, hist, hist, key_padding_mask=_safe_kpm(b.hist_mask))
            h = self.hist_norm(hist + self.hist_drop(h))
        else: h = self.hist_diff(hist, b.hist_mask)
        if self.variant == "no_ca":
            user = self.no_ca_pool(h, b.hist_mask)
            content = (F.normalize(user, dim=-1)[:, None, :] * F.normalize(cand, dim=-1)).sum(-1)
        else:
            att = torch.einsum("bcd,bhd->bch", self.cand_proj(cand), h) / self.scale
            att = torch.softmax(att.masked_fill(~b.hist_mask[:, None, :], -1e9), -1)
            user = torch.einsum("bch,bhd->bcd", att, h)
            content = (F.normalize(user, dim=-1) * F.normalize(cand, dim=-1)).sum(-1)
        if self.variant == "no_graph": return content
        gu, gn = self.graph(); graph = torch.einsum("bd,bcd->bc", F.normalize(gu[b.user_idx], dim=-1), F.normalize(gn[b.cand_idx], dim=-1))
        return content + torch.sigmoid(self.gate) * graph

    def extra_loss(self, b):
        if self.variant == "no_cl": return torch.zeros((), device=b.hist_idx.device)
        hist, _ = self._encode(b)
        v1 = F.normalize(self.user_pool(self.drop(hist), b.hist_mask), dim=-1); v2 = F.normalize(self.user_pool(self.drop(hist), b.hist_mask), dim=-1)
        logits = (v1 @ v2.t()) / self.tau; same = b.user_idx[:, None].eq(b.user_idx[None, :])
        return self.cl_weight * (torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~same, float("-inf")), 1)).mean()


class SuperNoDiff(SuperRec): variant = "no_diff"
class SuperNoCA(SuperRec): variant = "no_ca"
class SuperNoGraph(SuperRec): variant = "no_graph"
class SuperNoCL(SuperRec): variant = "no_cl"
class SuperNoBGE(SuperRec): variant = "no_bge"


REGISTRY = {
    "nrms": NRMS, "naml": NAML, "fastformer": Fastformer, "caum": CAUM,
    "lightgcn": LightGCN, "llmenc_ca": LLMEncCA, "supermodel": SuperRec,
    "super_no_diff": SuperNoDiff, "super_no_ca": SuperNoCA,
    "super_no_graph": SuperNoGraph, "super_no_cl": SuperNoCL,
    "super_no_bge": SuperNoBGE,
    "popularity": Popularity, "bge_zeroshot": BGEZeroShot, "bge_zs_pop": BGEZeroShotPop,
}
BASELINE_MODELS = ["nrms", "naml", "fastformer", "caum", "lightgcn"]
LLM_MODELS = ["llmenc_ca"]
SUPERMODEL_MODELS = ["supermodel", "super_no_diff", "super_no_ca", "super_no_graph", "super_no_cl", "super_no_bge"]
HEURISTIC_MODELS = ["popularity", "bge_zeroshot", "bge_zs_pop"]
FINAL_MODELS = BASELINE_MODELS + LLM_MODELS + SUPERMODEL_MODELS + HEURISTIC_MODELS


def build_model(name, data, cfg):
    if name not in REGISTRY: raise KeyError(name)
    return REGISTRY[name](data, cfg)
