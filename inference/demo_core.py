"""Data loading, artifact checks, replay, and local LLMEncCA inference."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
import hashlib
import json

import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import metrics  # noqa: E402  (training/)
from models import build_model  # noqa: E402
DEFAULT_REPLAY_MODELS = ("naml", "llmenc_ca", "supermodel", "super_no_cl")
MAX_HISTORY = 50
MAX_CANDIDATES = 50


class DemoDataError(RuntimeError):
    """Raised when local MIND data or V10 artifacts are missing or inconsistent."""


@dataclass(frozen=True)
class NewsArticle:
    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str


@dataclass(frozen=True)
class Impression:
    impression_id: str
    timestamp: str
    history_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    candidate_indices: tuple[int, ...]
    labels: tuple[int, ...]


@dataclass
class DemoData:
    root: Path
    news_by_id: dict[str, NewsArticle]
    news_ids: tuple[str, ...]
    news_index: dict[str, int]
    impressions: tuple[Impression, ...]
    impression_by_id: dict[str, Impression]
    result_rows: tuple[dict, ...]
    model_names: tuple[str, ...]
    config: dict


def _required(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise DemoDataError(f"Thiếu hoặc file rỗng: {path}")
    return path


def _read_news(paths: tuple[Path, ...]) -> OrderedDict[str, NewsArticle]:
    news: OrderedDict[str, NewsArticle] = OrderedDict()
    for path in paths:
        _required(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    raise DemoDataError(f"News row không đủ cột: {path}:{line_number}")
                news[fields[0]] = NewsArticle(
                    news_id=fields[0],
                    category=fields[1],
                    subcategory=fields[2],
                    title=fields[3],
                    abstract=fields[4],
                )
    return news


def _read_embedding_ids(path: Path) -> tuple[str, ...]:
    _required(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            ids = tuple(str(value) for value in archive["ids"].astype(str))
            model_name = str(archive["model_name"].item())
    except (OSError, KeyError, ValueError) as exc:
        raise DemoDataError(f"Không đọc được embedding metadata: {path}: {exc}") from exc
    if model_name != "BAAI/bge-small-en-v1.5":
        raise DemoDataError(f"Embedding model không khớp BGE v1.5: {model_name}")
    return ids


def _read_impressions(path: Path, news_index: dict[str, int]) -> tuple[Impression, ...]:
    _required(path)
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                raise DemoDataError(f"Behavior row không đủ cột: {path}:{line_number}")
            impression_id, _user_id, timestamp, history_text, candidate_text = fields[:5]
            if impression_id in seen:
                raise DemoDataError(f"Impression ID bị lặp: {impression_id}")
            seen.add(impression_id)
            history_ids = tuple(history_text.split())
            candidates, labels = [], []
            for token in candidate_text.split():
                try:
                    news_id, label_text = token.rsplit("-", 1)
                    label = int(label_text)
                except (ValueError, TypeError) as exc:
                    raise DemoDataError(f"Candidate sai định dạng tại {path}:{line_number}") from exc
                if label not in (0, 1):
                    raise DemoDataError(f"Label phải là 0 hoặc 1 tại {path}:{line_number}")
                candidates.append(news_id)
                labels.append(label)
            missing = [news_id for news_id in (*history_ids, *candidates) if news_id not in news_index]
            if missing:
                raise DemoDataError(f"News ID không có trong catalog tại {path}:{line_number}: {missing[0]}")
            rows.append(Impression(
                impression_id=impression_id,
                timestamp=timestamp,
                history_ids=history_ids,
                candidate_ids=tuple(candidates),
                candidate_indices=tuple(news_index[news_id] for news_id in candidates),
                labels=tuple(labels),
            ))
    return tuple(rows)


def load_demo_data(root: Path = PROJECT_ROOT) -> DemoData:
    root = Path(root).resolve()
    news = _read_news((root / "dataset/MINDsmall_train/news.tsv", root / "dataset/MINDsmall_dev/news.tsv"))
    news_ids = tuple(news)
    embedding_ids = _read_embedding_ids(root / "inference/artifacts/news_emb.npz")
    if news_ids != embedding_ids:
        raise DemoDataError("Thứ tự news ID trong news.tsv và news_emb.npz không khớp")
    news_index = {news_id: index + 1 for index, news_id in enumerate(news_ids)}
    impressions = _read_impressions(root / "dataset/MINDsmall_dev/behaviors.tsv", news_index)

    result_dir = root / "inference" / "artifacts" / "v10_final"
    _required(result_dir / "results.json")
    _required(result_dir / "config.json")
    try:
        results = json.loads((result_dir / "results.json").read_text(encoding="utf-8"))
        config = json.loads((result_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoDataError(f"Không đọc được metadata kết quả V10: {exc}") from exc
    model_names = tuple(results.get("models", ()))
    result_rows = tuple(results.get("rows", ()))
    if len(model_names) != 12 or {row.get("model") for row in result_rows} != set(model_names):
        raise DemoDataError("results.json không chứa đủ 12 model V10")
    return DemoData(root, news, news_ids, news_index, impressions,
                    {row.impression_id: row for row in impressions},
                    result_rows, model_names, config)


class PredictionStore:
    """Loads a shared candidate slate and selected model scores on demand."""

    def __init__(self, data: DemoData, reference_model: str = "naml"):
        self.data = data
        self.model_names = data.model_names
        self._lock = RLock()
        self._scores: dict[str, np.ndarray] = {}
        self._shared_model = reference_model
        (self.impression_ids, self.offsets, self.candidate_indices,
         self.labels, reference_scores) = self._read_archive(reference_model)
        self._scores[reference_model] = reference_scores
        self.impression_position = {value: i for i, value in enumerate(self.impression_ids)}
        self._validate_against_behaviors()

    def _path(self, model_name: str) -> Path:
        if model_name not in self.model_names:
            raise DemoDataError(f"Model không nằm trong kết quả V10: {model_name}")
        return _required(self.data.root / "inference" / "artifacts" / "v10_final" / "models" /
                         model_name / "test_predictions.npz")

    def _read_archive(self, model_name: str):
        path = self._path(model_name)
        try:
            with np.load(path, allow_pickle=False) as archive:
                required = {"impression_ids", "offsets", "candidate_news_indices", "labels", "scores"}
                if not required.issubset(archive.files):
                    raise DemoDataError(f"Thiếu prediction arrays trong {path}")
                values = (
                    archive["impression_ids"].astype(str),
                    archive["offsets"].astype(np.int64),
                    archive["candidate_news_indices"].astype(np.int64),
                    archive["labels"].astype(np.float32),
                    archive["scores"].astype(np.float32),
                )
        except (OSError, ValueError, KeyError) as exc:
            raise DemoDataError(f"Không đọc được predictions {path}: {exc}") from exc
        ids, offsets, candidate_indices, labels, scores = values
        if (len(offsets) != len(ids) + 1 or not len(offsets) or offsets[0] != 0
                or np.any(np.diff(offsets) < 0) or offsets[-1] != len(candidate_indices)
                or len(labels) != len(candidate_indices) or len(scores) != len(candidate_indices)):
            raise DemoDataError(f"Prediction array shape/offset không hợp lệ: {path}")
        if not np.isfinite(scores).all() or not np.isin(labels, (0, 1)).all():
            raise DemoDataError(f"Predictions có score hoặc label không hợp lệ: {path}")
        if len(set(ids.tolist())) != len(ids):
            raise DemoDataError(f"Impression ID bị lặp trong predictions: {path}")
        return ids, offsets, candidate_indices, labels, scores

    def _validate_against_behaviors(self) -> None:
        if self.impression_ids.tolist() != [row.impression_id for row in self.data.impressions]:
            raise DemoDataError("Impression ID predictions không khớp thứ tự MINDsmall_dev")
        for i, row in enumerate(self.data.impressions):
            start, end = self.offsets[i:i + 2]
            if not np.array_equal(self.candidate_indices[start:end], row.candidate_indices):
                raise DemoDataError(f"Candidate slate không khớp impression {row.impression_id}")
            if not np.array_equal(self.labels[start:end], row.labels):
                raise DemoDataError(f"Labels không khớp impression {row.impression_id}")

    def scores(self, model_name: str) -> np.ndarray:
        with self._lock:
            if model_name in self._scores:
                return self._scores[model_name]
            ids, offsets, candidate_indices, labels, scores = self._read_archive(model_name)
            if (not np.array_equal(ids, self.impression_ids)
                    or not np.array_equal(offsets, self.offsets)
                    or not np.array_equal(candidate_indices, self.candidate_indices)
                    or not np.array_equal(labels, self.labels)):
                raise DemoDataError(f"Model {model_name} không cùng impression/candidate slate với {self._shared_model}")
            self._scores[model_name] = scores
            return scores

    def span(self, impression_id: str) -> tuple[int, int]:
        try:
            index = self.impression_position[str(impression_id)]
        except KeyError as exc:
            raise DemoDataError(f"Không tìm thấy impression ID: {impression_id}") from exc
        return int(self.offsets[index]), int(self.offsets[index + 1])


def impression_metrics(labels, scores) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    return {
        "MRR": metrics.mrr_score(labels, scores),
        "nDCG@5": metrics.ndcg_score(labels, scores, 5),
        "nDCG@10": metrics.ndcg_score(labels, scores, 10),
    }


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_llmenc_ca(data: DemoData):
    model_dir = data.root / "inference" / "artifacts" / "v10_final" / "models" / "llmenc_ca"
    checkpoint = _required(model_dir / "best.pt")
    metadata_path = _required(model_dir / "best_meta.json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoDataError(f"Không đọc được metadata LLMEncCA: {exc}") from exc
    actual_sha = checkpoint_sha256(checkpoint)
    if actual_sha != metadata.get("checkpoint_sha256"):
        raise DemoDataError("SHA-256 checkpoint llmenc_ca không khớp best_meta.json")
    if metadata.get("model") != "llmenc_ca":
        raise DemoDataError("best_meta.json không phải checkpoint llmenc_ca")

    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise DemoDataError("Phiên bản PyTorch phải hỗ trợ torch.load(weights_only=True)") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise DemoDataError(f"Không nạp được checkpoint llmenc_ca: {exc}") from exc
    embedding = state.get("news.emb")
    expected_rows = len(data.news_ids) + 1
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2 or embedding.shape[0] != expected_rows:
        raise DemoDataError("Embedding trong checkpoint không khớp số news ID đã xác minh")
    cfg = data.config.get("early_stopping", {})
    model_cfg = SimpleNamespace(dim=int(cfg.get("dim", 64)), dropout=float(cfg.get("dropout", 0.2)))
    model_data = SimpleNamespace(llm_emb=embedding.numpy())
    model = build_model("llmenc_ca", model_data, model_cfg)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise DemoDataError(f"Checkpoint không khớp kiến trúc LLMEncCA: {exc}") from exc
    del state, model_data
    model.eval()
    return model


def score_custom(model, data: DemoData, history_ids, candidate_ids) -> np.ndarray:
    history_ids = list(history_ids)
    candidate_ids = list(candidate_ids)
    if not 1 <= len(history_ids) <= MAX_HISTORY:
        raise ValueError(f"Lịch sử phải có từ 1 đến {MAX_HISTORY} bài")
    if not 2 <= len(candidate_ids) <= MAX_CANDIDATES:
        raise ValueError(f"Danh sách ứng viên phải có từ 2 đến {MAX_CANDIDATES} bài")
    if len(set(history_ids)) != len(history_ids):
        raise ValueError("Lịch sử có bài bị lặp")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Danh sách ứng viên có bài bị lặp")
    if set(history_ids) & set(candidate_ids):
        raise ValueError("Một bài không thể nằm đồng thời trong lịch sử và danh sách ứng viên")
    unknown = (set(history_ids) | set(candidate_ids)) - data.news_index.keys()
    if unknown:
        raise ValueError(f"News ID không tồn tại trong MIND-small: {sorted(unknown)[0]}")

    history_indices = [data.news_index[news_id] for news_id in history_ids]
    padded = history_indices + [0] * (MAX_HISTORY - len(history_indices))
    mask = [True] * len(history_indices) + [False] * (MAX_HISTORY - len(history_indices))
    batch = SimpleNamespace(
        hist_idx=torch.tensor([padded], dtype=torch.long),
        cand_idx=torch.tensor([[data.news_index[news_id] for news_id in candidate_ids]], dtype=torch.long),
        hist_mask=torch.tensor([mask], dtype=torch.bool),
    )
    with torch.inference_mode():
        scores = model.score(batch).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    if scores.shape != (len(candidate_ids),) or not np.isfinite(scores).all():
        raise RuntimeError("LLMEncCA trả về score sai kích thước hoặc không hữu hạn")
    return scores
