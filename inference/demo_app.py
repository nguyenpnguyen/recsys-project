"""Local HTML research demo for the V10 MIND-small benchmark (stdlib server + static index.html)."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse
import argparse
import json
import random

import numpy as np

from demo_core import (
    DEFAULT_REPLAY_MODELS,
    DemoDataError,
    PROJECT_ROOT,
    PredictionStore,
    impression_metrics,
    load_demo_data,
    load_llmenc_ca,
    score_custom,
)

HERE = Path(__file__).resolve().parent
CHARTS = {"val_auc_vs_epoch.png", "val_ndcg10_vs_epoch.png"}


class Demo:
    def __init__(self):
        self.data = load_demo_data(PROJECT_ROOT)
        self.store = PredictionStore(self.data)
        self.impression_ids = [int(v) for v in self.store.impression_ids]
        self.categories = sorted({a.category for a in self.data.news_by_id.values()})
        self._model = None
        self._model_lock = Lock()

    def article(self, news_id):
        a = self.data.news_by_id[news_id]
        return {"id": news_id, "title": a.title, "category": a.category, "subcategory": a.subcategory}

    def meta(self):
        return {
            "overview": list(self.data.result_rows),
            "models": list(self.data.model_names),
            "default_models": list(DEFAULT_REPLAY_MODELS),
            "categories": self.categories,
            "impression_range": [min(self.impression_ids), max(self.impression_ids)],
            "n_articles": len(self.data.news_by_id),
        }

    def replay(self, impression_id, models, reveal):
        row = self.data.impression_by_id.get(str(impression_id))
        if row is None:
            raise DemoDataError(f"Không tìm thấy impression {impression_id} trong MINDsmall_dev.")
        start, end = self.store.span(row.impression_id)
        scores = {m: self.store.scores(m)[start:end] for m in models}
        ranks = {}
        for m, s in scores.items():
            order = np.argsort(s)[::-1]
            r = np.empty(len(order), dtype=np.int64)
            r[order] = np.arange(1, len(order) + 1)
            ranks[m] = r
        candidates = []
        for i, news_id in enumerate(row.candidate_ids):
            item = self.article(news_id) | {"ranks": {m: int(ranks[m][i]) for m in models}}
            if reveal:
                item["clicked"] = bool(row.labels[i])
            candidates.append(item)
        if models:
            candidates.sort(key=lambda c: c["ranks"][models[0]])
        out = {
            "impression_id": row.impression_id, "timestamp": row.timestamp,
            "history": [self.article(n) for n in row.history_ids[-50:]],
            "history_total": len(row.history_ids), "candidates": candidates,
        }
        if reveal:
            out["metrics"] = [{"Model": m, **impression_metrics(row.labels, s)} for m, s in scores.items()]
        return out

    def search(self, query, category, offset=0, limit=20):
        q = query.strip().casefold()
        if len(q) < 2:
            return {"items": [], "total": 0}
        ids = [a.news_id for a in self.data.news_by_id.values()
               if (not category or a.category == category) and q in a.title.casefold()]
        return {"items": [self.article(i) for i in ids[offset:offset + limit]], "total": len(ids)}

    def articles(self, ids):
        return [self.article(i) for i in ids if i in self.data.news_by_id]

    def rank(self, history, candidates):
        with self._model_lock:
            if self._model is None:
                self._model = load_llmenc_ca(self.data)
            scores = score_custom(self._model, self.data, history, candidates)
        return [float(s) for s in scores]


class Handler(BaseHTTPRequestHandler):
    demo: Demo

    def _send(self, body: bytes, ctype: str, status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def _handle(self, fn):
        try:
            self._json(fn())
        except (DemoDataError, ValueError, RuntimeError, KeyError) as exc:
            self._json({"error": str(exc)}, 400)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        d = self.demo
        if url.path == "/":
            self._send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path.startswith("/charts/") and url.path[8:] in CHARTS:
            path = d.data.root / "inference" / "artifacts" / "v10_final" / url.path[8:]
            if path.is_file() and path.stat().st_size:
                self._send(path.read_bytes(), "image/png")
            else:
                self._json({"error": f"Thiếu biểu đồ: {path}"}, 404)
        elif url.path == "/api/meta":
            self._json(d.meta())
        elif url.path == "/api/random":
            self._json({"id": random.choice(d.impression_ids)})
        elif url.path == "/api/replay":
            models = [m for m in q.get("models", "").split(",") if m]
            self._handle(lambda: d.replay(q.get("id", ""), models, q.get("reveal") == "1"))
        elif url.path == "/api/articles":
            self._handle(lambda: d.articles(q.get("ids", "").split(",")))
        elif url.path == "/api/search":
            self._handle(lambda: d.search(
                q.get("q", ""), q.get("category", ""), int(q.get("offset", 0)), min(int(q.get("limit", 20)), 100)))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/rank":
            return self._json({"error": "not found"}, 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            history, candidates = list(body["history"]), list(body["candidates"])
        except (ValueError, KeyError, TypeError):
            return self._json({"error": "Body phải là JSON có history và candidates."}, 400)
        self._handle(lambda: {"scores": self.demo.rank(history, candidates)})

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        Handler.demo = Demo()
    except (DemoDataError, OSError) as exc:
        raise SystemExit(f"{exc}\nTải artifact: hf download nguyenpn/recsys-artifacts --local-dir inference/artifacts")
    print(f"Demo: http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
