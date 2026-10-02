"""Small integration checks for the local V10 demo."""
from pathlib import Path
from threading import Thread
from http.server import ThreadingHTTPServer
import json
import unittest
import urllib.request

import numpy as np

import demo_app

from demo_core import (
    DEFAULT_REPLAY_MODELS,
    PredictionStore,
    impression_metrics,
    load_demo_data,
    load_llmenc_ca,
    score_custom,
)


HERE = Path(__file__).resolve().parent


class DemoCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_demo_data()
        cls.predictions = PredictionStore(cls.data)

    def test_catalog_and_prediction_alignment(self):
        self.assertEqual(len(self.data.news_ids), 65238)
        self.assertEqual(len(self.data.impressions), 73152)
        self.assertEqual(len(set(self.predictions.impression_ids.tolist())), 73152)
        for model_name in DEFAULT_REPLAY_MODELS:
            scores = self.predictions.scores(model_name)
            self.assertEqual(len(scores), len(self.predictions.candidate_indices))

    def test_metrics_and_input_validation(self):
        values = impression_metrics([1, 0, 1], [0.8, 0.2, 0.5])
        self.assertGreater(values["MRR"], 0)
        self.assertGreater(values["nDCG@10"], 0)
        first, second, third = self.data.news_ids[:3]
        with self.assertRaises(ValueError):
            score_custom(None, self.data, [], [first, second])
        with self.assertRaises(ValueError):
            score_custom(None, self.data, [first], [first, second])
        with self.assertRaises(ValueError):
            score_custom(None, self.data, [first, first], [second, third])
        with self.assertRaises(ValueError):
            score_custom(None, self.data, ["not-a-mind-news-id"], [second, third])

    def test_checkpoint_reproduces_saved_scores(self):
        model = load_llmenc_ca(self.data)
        stored = self.predictions.scores("llmenc_ca")
        checked = 0
        for row in self.data.impressions:
            if not row.history_ids or len(row.candidate_ids) < 2:
                continue
            actual = score_custom(model, self.data, row.history_ids[-50:], row.candidate_ids)
            start, end = self.predictions.span(row.impression_id)
            expected = stored[start:end]
            np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=1e-4)
            self.assertEqual(np.argsort(actual).tolist(), np.argsort(expected).tolist())
            checked += 1
            if checked == 5:
                break
        self.assertEqual(checked, 5)
        custom_scores = score_custom(model, self.data, self.data.news_ids[:1], self.data.news_ids[1:4])
        self.assertEqual(custom_scores.shape, (3,))
        self.assertTrue(np.isfinite(custom_scores).all())


class DemoAppTests(unittest.TestCase):
    def test_http_api(self):
        demo_app.Handler.demo = demo_app.Demo()
        server = ThreadingHTTPServer(("127.0.0.1", 0), demo_app.Handler)
        Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        get = lambda path: urllib.request.urlopen(base + path).read()
        try:
            self.assertIn(b"<title>", get("/"))
            meta = json.loads(get("/api/meta"))
            rid = json.loads(get("/api/random"))["id"]
            models = ",".join(meta["default_models"])
            data = json.loads(get(f"/api/replay?id={rid}&models={models}&reveal=1"))
            self.assertTrue(data["candidates"] and data["metrics"])
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
