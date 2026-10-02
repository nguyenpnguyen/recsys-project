# Gợi ý tin tức trên MIND-small (V10)

Benchmark 15 mô hình gợi ý tin tức trên MIND-small: 5 baseline (NRMS, NAML, Fastformer, CAUM, LightGCN), LLMEncCA (news encoder BGE-small-en-v1.5), Supermodel, 5 biến thể ablation và 3 model không cần train (`popularity`: CTR online đã làm trơn, chỉ đếm từ impression trước đó; `bge_zeroshot`: độ giống BGE giữa candidate và trung bình history; `bge_zs_pop`: tổng hai điểm trên sau chuẩn hoá). Tập validation được tách theo ngày từ `MINDsmall_train`; `MINDsmall_dev` chỉ dùng một lần làm tập test, sau khi nạp checkpoint tốt nhất của từng model (chọn theo nDCG@10 trên validation).

```
training/    script train + đánh giá, notebook chạy trên Kaggle
dataset/     MIND-small (không commit)
inference/   demo HTML (stdlib server): tổng quan benchmark, phát lại impression, xếp hạng tương tác
```

## Dữ liệu

Tải MIND-small và giải nén vào `dataset/MINDsmall_train/` và `dataset/MINDsmall_dev/` (xem `dataset/README.md`).

## Training

### Trên Kaggle

1. Import `training/kaggle_run.ipynb` vào một notebook Kaggle mới.
2. Settings: bật GPU + Internet. Add Input: dataset chứa `MINDsmall_train/` và `MINDsmall_dev/`.
3. Run All. Notebook tự clone repo này, import các script trong `training/` và ghi `news_emb.npz` cùng `v10_final/` vào `/kaggle/working/`.

### Chạy local

```bash
uv venv --python 3.12 training/.venv
uv pip install --python training/.venv/bin/python -r training/requirements.txt
cd training
.venv/bin/python llm_embed.py --news ../dataset/MINDsmall_train/news.tsv ../dataset/MINDsmall_dev/news.tsv \
    --out ../inference/artifacts/news_emb.npz
.venv/bin/python bench.py --mind-train ../dataset/MINDsmall_train --mind-dev ../dataset/MINDsmall_dev \
    --news-emb ../inference/artifacts/news_emb.npz --out ../inference/artifacts/v10_final
```

Notebook cũng benchmark lại các model dùng embedding (`llmenc_ca`, `supermodel`, `bge_zeroshot`, `bge_zs_pop`) với `Qwen/Qwen3-Embedding-0.6B`, ghi ra `news_emb_qwen3.npz` và `v10_qwen3/`. Chạy local:

```bash
.venv/bin/python llm_embed.py --news ../dataset/MINDsmall_train/news.tsv ../dataset/MINDsmall_dev/news.tsv \
    --model Qwen/Qwen3-Embedding-0.6B --batch-size 32 --out ../inference/artifacts/news_emb_qwen3.npz
.venv/bin/python bench.py --mind-train ../dataset/MINDsmall_train --mind-dev ../dataset/MINDsmall_dev \
    --news-emb ../inference/artifacts/news_emb_qwen3.npz --out ../inference/artifacts/v10_qwen3 \
    --models llmenc_ca supermodel bge_zeroshot bge_zs_pop
```

Cell cuối của notebook upload `news_emb.npz` và `v10_final/` lên Hugging Face Hub (`nguyenpn/recsys-artifacts`). Trên Kaggle thêm Secret `HF_TOKEN` (quyền write); local chạy `hf auth login` trước. Upload thủ công:

```bash
hf upload nguyenpn/recsys-artifacts inference/artifacts/news_emb.npz news_emb.npz
hf upload nguyenpn/recsys-artifacts inference/artifacts/v10_final v10_final
```

`kaggle_run.ipynb` cũng chạy được local từ thư mục `training/` với cùng các đường dẫn trên.

| File | Vai trò |
| --- | --- |
| `data.py` | Đọc MIND, tách validation theo ngày, tạo batch |
| `models.py` | Các model và registry `build_model` |
| `bench.py` | Train với early stopping, test checkpoint tốt nhất, ghi kết quả; `verify()` kiểm tra artifact |
| `metrics.py` | AUC, MRR, nDCG@5/10 theo từng impression |
| `llm_embed.py` | Tính trước embedding BGE từ title + abstract |

## Demo inference

Cần output của training trong `inference/artifacts/` (tải từ Hugging Face Hub sau khi cài môi trường, hoặc tự train):

```
inference/artifacts/news_emb.npz
inference/artifacts/v10_final/{results.json, config.json, *.png, models/<name>/...}
```

```bash
uv venv --python 3.12 inference/.venv
uv pip install --python inference/.venv/bin/python -r inference/requirements.txt
inference/.venv/bin/hf download nguyenpn/recsys-artifacts --local-dir inference/artifacts
inference/.venv/bin/python inference/demo_app.py   # http://127.0.0.1:8000
cd inference && .venv/bin/python -m unittest test_demo.py   # test
```

Sau khi tải artifact, demo chạy trên CPU, không cần mạng. Đặc tả chi tiết xem `inference/PLAN_DEMO_V10.md`.
