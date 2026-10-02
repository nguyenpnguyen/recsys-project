"""V10 benchmark runner: chronological split, one best checkpoint per model."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, tempfile, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import metrics
from data import MindData, iter_batches, _eval_parts
from models import BASELINE_MODELS, FINAL_MODELS, HEURISTIC_MODELS, REGISTRY, build_model


@dataclass
class Cfg:
    dim: int = 64; dropout: float = .2; heads: int = 2; gcn_layers: int = 2
    seed: int = 0; cl_tau: float = .1; cl_weight: float = .1; lr: float = 1e-3
    weight_decay: float = 1e-5; max_epochs: int = 12; min_epochs: int = 3
    patience: int = 2; min_delta: float = .001; monitor: str = "ndcg@10"
    mode: str = "max"; batch_size: int = 64; eval_batch_size: int = 32


def evaluate(model, data, cfg, device, rows, collect=False):
    if not rows: raise ValueError("cannot evaluate an empty split")
    model.eval(); impressions = []; ids = []; offsets = [0]; cand_ids = []; labels_flat = []; scores_flat = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch, raw_rows in zip(iter_batches(rows, cfg.eval_batch_size, data.collate_eval),
                                   [rows[i:i + cfg.eval_batch_size] for i in range(0, len(rows), cfg.eval_batch_size)]):
            batch = batch.to(device); logits = model.score(batch)
            if not torch.isfinite(logits).all(): raise FloatingPointError("non-finite logits")
            logits = logits.masked_fill(~batch.cand_mask, float("-inf"))
            sc, lab, mask, ci = logits.cpu().numpy(), batch.labels.cpu().numpy(), batch.cand_mask.cpu().numpy(), batch.cand_idx.cpu().numpy()
            for j, row in enumerate(raw_rows):
                m = mask[j]; impressions.append((lab[j][m], sc[j][m]))
                if collect:
                    ids.append(_eval_parts(row)[0]); cand_ids.extend(ci[j][m].tolist()); labels_flat.extend(lab[j][m].tolist()); scores_flat.extend(sc[j][m].tolist()); offsets.append(len(cand_ids))
    result = metrics.aggregate(impressions); result["infer_impr_per_s"] = len(impressions) / max(time.perf_counter() - t0, 1e-9)
    if collect:
        result["predictions"] = {"impression_ids": np.asarray(ids, dtype=str), "offsets": np.asarray(offsets, np.int64),
                                  "candidate_news_indices": np.asarray(cand_ids, np.int64), "labels": np.asarray(labels_flat, np.float32),
                                  "scores": np.asarray(scores_flat, np.float32)}
    return result


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def save_state_atomic(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f: tmp = Path(f.name)
    try: torch.save(state, tmp); os.replace(tmp, path)
    finally: tmp.unlink(missing_ok=True)


def train_model(model, data, cfg, device, model_dir):
    model.to(device); trainable = params(model) > 0  # heuristic baselines: one validation pass, no optimisation
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) if trainable else None
    steps = math.ceil(len(data.train_core) / cfg.batch_size); history = []; best = None; stale = 0; step = 0; t0 = time.perf_counter()
    best_path = model_dir / "best.pt"
    for epoch in range(1, cfg.max_epochs + 1):
        model.train(); total, batches = 0., 0; epoch_t = time.perf_counter()
        for batch in (iter_batches(data.train_core, cfg.batch_size, data.collate_train, shuffle=True, seed=cfg.seed + epoch) if trainable else ()):
            batch = batch.to(device); logits = model.score(batch); target = torch.zeros(logits.size(0), dtype=torch.long, device=device)
            rank_loss = F.cross_entropy(logits, target)
            aux_loss = model.extra_loss(batch) if hasattr(model, "extra_loss") else 0.
            loss = rank_loss + aux_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch} step={step + 1}: ranking={float(rank_loss):.4g}, auxiliary={float(aux_loss):.4g}")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1; total += loss.item(); batches += 1
        val = evaluate(model, data, cfg, device, data.validation); rec = {"epoch": epoch, "global_step": step, "train_loss": total / max(batches, 1), "val_auc": val["auc"], "val_ndcg@10": val["ndcg@10"], "elapsed_s": time.perf_counter() - epoch_t}; history.append(rec)
        improved = best is None or rec["val_ndcg@10"] > best["val_ndcg@10"] + cfg.min_delta
        if improved: best, stale = dict(rec), 0; save_state_atomic(model.state_dict(), best_path)
        elif epoch >= cfg.min_epochs: stale += 1
        print(f"  epoch {epoch:02d} step={step:,} loss={rec['train_loss']:.4f} val_auc={rec['val_auc']:.4f} val_ndcg@10={rec['val_ndcg@10']:.4f}")
        if not trainable or (epoch >= cfg.min_epochs and stale >= cfg.patience): break
    if best is None: raise RuntimeError("no best checkpoint")
    return history, time.perf_counter() - t0, steps, best


def load_state(model, path, device):
    try: state = torch.load(path, map_location=device, weights_only=True)
    except TypeError: state = torch.load(path, map_location=device)
    model.load_state_dict(state)


def params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
def group(name):
    if name in BASELINE_MODELS: return "baseline"
    if name == "llmenc_ca": return "llm-experiment"
    if name in HEURISTIC_MODELS: return "no-training"
    if name == "supermodel": return "proposed"
    return "super-ablation"


def make_table(rows):
    cols = ["model", "group", "control", "best_epoch", "epochs_trained", "train_steps", "test_auc", "test_mrr", "test_ndcg@5", "test_ndcg@10", "delta_auc", "delta_mrr", "delta_ndcg@5", "delta_ndcg@10", "params", "train_s", "infer_impr/s"]
    head = "| " + " | ".join(cols) + " |\n|" + "|".join(" --- " if c in cols[:3] else " ---: " for c in cols) + "|"
    lines = [head]
    for r in rows:
        lines.append("| " + " | ".join("—" if r[c] is None else (f"{r[c]:.4f}" if c.startswith(("test_", "delta_")) else f"{r[c]:.0f}" if c in ("params", "train_steps", "infer_impr/s") else str(r[c])) for c in cols) + " |")
    return "\n".join(lines)


def plot(histories, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for key, name, filename in [("val_auc", "Validation AUC", "val_auc_vs_epoch.png"), ("val_ndcg@10", "Validation nDCG@10", "val_ndcg10_vs_epoch.png")]:
        fig, ax = plt.subplots(figsize=(13, 7))
        for model, hist in histories.items():
            x = [r["epoch"] for r in hist]; y = [r[key] for r in hist]; ax.plot(x, y, marker="o", label=model)
            best = max(hist, key=lambda r: r["val_ndcg@10"]); ax.scatter([best["epoch"]], [best[key]], s=45)
        ax.set_xlabel("epoch"); ax.set_ylabel(name); ax.set_title(f"{name} vs epoch — V10 final"); ax.grid(alpha=.25); ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8); fig.tight_layout(); fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--source", choices=["mind"], default="mind"); ap.add_argument("--mind-train"); ap.add_argument("--mind-dev"); ap.add_argument("--news-emb", required=True); ap.add_argument("--dev-ratio", type=float, default=.1); ap.add_argument("--models", nargs="+", default=FINAL_MODELS); ap.add_argument("--max-epochs", type=int, default=12); ap.add_argument("--min-epochs", type=int, default=3); ap.add_argument("--patience", type=int, default=2); ap.add_argument("--min-delta", type=float, default=.001); ap.add_argument("--dim", type=int, default=64); ap.add_argument("--heads", type=int, default=2); ap.add_argument("--gcn-layers", type=int, default=2); ap.add_argument("--cl-tau", type=float, default=.1); ap.add_argument("--cl-weight", type=float, default=.1); ap.add_argument("--dropout", type=float, default=.2); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--weight-decay", type=float, default=1e-5); ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--eval-batch", type=int, default=32); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/v10_final"); args = ap.parse_args(argv)
    if not args.mind_train or not args.mind_dev or os.path.abspath(args.mind_train) == os.path.abspath(args.mind_dev): raise ValueError("provide separate MINDsmall_train and MINDsmall_dev")
    if not set(args.models) <= set(FINAL_MODELS) or len(set(args.models)) != len(args.models): raise ValueError("models must be distinct FINAL_MODELS")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(args.news_emb, allow_pickle=True); emb = {(i.decode() if isinstance(i, bytes) else str(i)): v for i, v in zip(z["ids"], z["vecs"])}
    data = MindData.from_mind(args.mind_train, args.mind_dev, llm_embeddings=emb, dev_ratio=args.dev_ratio, seed=args.seed)
    cfg = Cfg(dim=args.dim, dropout=args.dropout, heads=args.heads, gcn_layers=args.gcn_layers, seed=args.seed, cl_tau=args.cl_tau, cl_weight=args.cl_weight, lr=args.lr, weight_decay=args.weight_decay, max_epochs=args.max_epochs, min_epochs=args.min_epochs, patience=args.patience, min_delta=args.min_delta, batch_size=args.batch_size, eval_batch_size=args.eval_batch)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True); models_dir = out_dir / "models"; models_dir.mkdir(exist_ok=True)
    config = {"models": list(args.models), "seed": args.seed, "bge_model": str(z["model_name"]), "dataset": "MIND-small", "input_text": "title + abstract", "monitor": "ndcg@10", "early_stopping": vars(cfg)}; (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "split.json").write_text(json.dumps(data.split_info, indent=2))
    rows, histories = [], {}
    for name in args.models:
        print(f"\n===== {name} ({group(name)}) ====="); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        model_dir = models_dir / name; model_dir.mkdir(parents=True, exist_ok=True); model = build_model(name, data, cfg)
        history, train_s, steps, best = train_model(model, data, cfg, device, model_dir); histories[name] = history
        load_state(model, model_dir / "best.pt", device); checkpoint_sha = sha256(model_dir / "best.pt"); (model_dir / "best_meta.json").write_text(json.dumps({"model": name, "best_epoch": best["epoch"], "best_val_auc": best["val_auc"], "best_val_ndcg@10": best["val_ndcg@10"], "checkpoint_sha256": checkpoint_sha, "config": config}, indent=2))
        test = evaluate(model, data, cfg, device, data.test, collect=True); pred = test.pop("predictions"); np.savez(model_dir / "test_predictions.npz", **pred); test_meta = dict(test, checkpoint_path=str(model_dir / "best.pt"), checkpoint_sha256=checkpoint_sha); (model_dir / "test_metrics.json").write_text(json.dumps(test_meta, indent=2))
        rows.append({"model": name, "group": group(name), "control": None, "best_epoch": best["epoch"], "epochs_trained": len(history), "train_steps": history[-1]["global_step"], "test_auc": test["auc"], "test_mrr": test["mrr"], "test_ndcg@5": test["ndcg@5"], "test_ndcg@10": test["ndcg@10"], "delta_auc": None, "delta_mrr": None, "delta_ndcg@5": None, "delta_ndcg@10": None, "params": params(model), "train_s": train_s, "infer_impr/s": test["infer_impr_per_s"]})
        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    by_name = {r["model"]: r for r in rows}; ref = by_name.get("supermodel")
    for row in rows:
        if row["group"] == "super-ablation" and ref: row["control"] = "supermodel"; row["delta_auc"] = row["test_auc"] - ref["test_auc"]; row["delta_mrr"] = row["test_mrr"] - ref["test_mrr"]; row["delta_ndcg@5"] = row["test_ndcg@5"] - ref["test_ndcg@5"]; row["delta_ndcg@10"] = row["test_ndcg@10"] - ref["test_ndcg@10"]
        elif row["model"] == "supermodel": row["delta_auc"] = row["delta_mrr"] = row["delta_ndcg@5"] = row["delta_ndcg@10"] = 0.; row["control"] = "—"
        else: row["control"] = "—"
    rows.sort(key=lambda r: r["test_ndcg@10"], reverse=True); plot(histories, out_dir)
    (out_dir / "history.json").write_text(json.dumps({"models": histories}, indent=2)); (out_dir / "results.json").write_text(json.dumps({"models": args.models, "rows": rows, "split": data.split_info}, indent=2)); (out_dir / "results.md").write_text("# V10 final benchmark\n\n" + make_table(rows) + "\n")
    print(make_table(rows))



def verify(out_dir):
    """Check every V10 artifact in ``out_dir`` and that saved scores reproduce saved metrics."""
    out_dir = Path(out_dir)
    for filename in ("config.json", "split.json", "results.json", "results.md", "history.json", "val_auc_vs_epoch.png", "val_ndcg10_vs_epoch.png"):
        assert (out_dir / filename).exists() and (out_dir / filename).stat().st_size > 0, f"missing artifact: {out_dir / filename}"
    results = json.loads((out_dir / "results.json").read_text()); history = json.loads((out_dir / "history.json").read_text())
    names = results["models"]
    assert len(results["rows"]) == len(names) and {r["model"] for r in results["rows"]} == set(names)
    assert all(r["test_ndcg@10"] >= results["rows"][i + 1]["test_ndcg@10"] for i, r in enumerate(results["rows"][:-1]))
    assert set(history["models"]) == set(names)
    split = json.loads((out_dir / "split.json").read_text())
    assert split["train_day_max"] < split["validation_day_min"] and not (set(split["train_days"]) & set(split["validation_days"]))
    for name in names:
        d = out_dir / "models" / name
        for filename in ("best.pt", "best_meta.json", "test_metrics.json", "test_predictions.npz"):
            assert (d / filename).exists() and (d / filename).stat().st_size > 0, f"missing {name}/{filename}"
        assert json.loads((d / "best_meta.json").read_text())["checkpoint_sha256"] == sha256(d / "best.pt")
        test_meta = json.loads((d / "test_metrics.json").read_text())
        with np.load(d / "test_predictions.npz", allow_pickle=False) as pred:
            offsets, labels, scores = pred["offsets"], pred["labels"], pred["scores"]
            assert len(offsets) == len(pred["impression_ids"]) + 1
            assert len(labels) == len(scores) == len(pred["candidate_news_indices"]) == offsets[-1]
        recomputed = metrics.aggregate((labels[a:b], scores[a:b]) for a, b in zip(offsets[:-1], offsets[1:]))
        for m in ("auc", "mrr", "ndcg@5", "ndcg@10"):
            assert np.isclose(recomputed[m], test_meta[m], atol=1e-6), f"{name}: {m} mismatch"
        print(f"✅ verified {name}")
    print(f"✅ V10: {len(names)} rows, chronological split, epoch curves, and checkpoint/test artifacts verified")
    print((out_dir / "results.md").read_text())


if __name__ == "__main__": main()
