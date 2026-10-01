# Kế hoạch V10 — benchmark chính thức và true ablation trên MIND-small

## 1. Mục tiêu

Tạo `newsrec-bench-kaggle-v10-final.ipynb` từ V9 để thực hiện một lần chạy chính thức trên Kaggle. V10 phải:

1. Train lại các baseline, `LLMEncCA`, `Supermodel-full` và từng ablation của Supermodel trên cùng dữ liệu và giao thức.
2. Loại hoàn toàn các model screening `hybridopt`, `graphrec`, `nrms_diff`, `nrms_cl` và mọi NRMS ablation khác.
3. Mỗi ablation của Supermodel được khởi tạo và train lại từ đầu; không tắt module chỉ lúc inference.
4. Chọn checkpoint bằng validation nDCG@10, sau đó đánh giá đúng một lần trên `MINDsmall_dev`.
5. Tổng hợp tất cả model và delta ablation trong đúng một bảng kết quả.
6. Vẽ validation AUC và validation nDCG@10 theo epoch.
7. Chạy độc lập trên Kaggle, không giả định file notebook nguồn tồn tại trong filesystem.

V10 là benchmark cuối trên MIND-small. Không trộn số từ V8, V9 hoặc paper vào bảng kết quả V10.

## 2. Phạm vi model

### 2.1. Baseline

```python
BASELINE_MODELS = [
    "nrms",
    "naml",
    "fastformer",
    "caum",
    "lightgcn",
]
```

Đây là baseline của bài toán. `LLMEncCA` không thuộc baseline.

### 2.2. LLM encoding experiment

```python
LLM_MODELS = [
    "llmenc_ca",
]
```

`LLMEncCA` dùng frozen BGE embedding và candidate-aware aggregation. Model này là đối chứng cho việc dùng LLM/news embedding nhẹ, không phải baseline truyền thống và không phải ablation của Supermodel.

### 2.3. Supermodel và true ablation

```python
SUPERMODEL_MODELS = [
    "supermodel",
    "super_no_diff",
    "super_no_ca",
    "super_no_graph",
    "super_no_cl",
    "super_no_bge",
]
```

```python
FINAL_MODELS = BASELINE_MODELS + LLM_MODELS + SUPERMODEL_MODELS
```

Tổng cộng 12 cấu hình.

### 2.4. Model không chạy

Không đăng ký hoặc gọi các model sau trong V10:

```text
hybridopt
graphrec
nrms_diff
nrms_cl
nrms_mla
nrms_ssm
nrms_rope
nrms_multi
llmenc
```

Nếu code cũ vẫn còn để tham khảo, chúng không được xuất hiện trong `FINAL_MODELS`, bảng kết quả, history hoặc biểu đồ.

## 3. Định nghĩa Supermodel-full

`supermodel` gồm năm thành phần cần khảo sát:

1. Frozen `BAAI/bge-small-en-v1.5` news embedding.
2. Residual Differential Attention trên click history.
3. Candidate-aware user aggregation.
4. LightGCN graph branch và learned fusion gate.
5. Multi-positive contrastive auxiliary loss.

Điểm cuối kết hợp content score và graph score bằng learned gate. Tất cả ablation phải giữ nguyên các thành phần còn lại và chỉ thay đổi đúng thành phần được nêu.

## 4. Định nghĩa true ablation

| Model | Thay đổi duy nhất so với `supermodel` | Cách triển khai |
|---|---|---|
| `super_no_diff` | Bỏ Differential Attention | Thay bằng vanilla multi-head self-attention cùng `dim` và số head; giữ residual, mask và pooling |
| `super_no_ca` | Bỏ candidate-aware aggregation | Dùng candidate-independent additive attention/pooling đã có trong NRMS |
| `super_no_graph` | Bỏ graph branch | Không tính LightGCN embedding, graph score hoặc fusion gate; chỉ dùng content score |
| `super_no_cl` | Bỏ contrastive objective | Giữ nguyên kiến trúc, đặt `cl_weight = 0` từ đầu quá trình train |
| `super_no_bge` | Bỏ pretrained BGE embedding | Thay bằng learned title/category news encoder đang có; giữ DiffAttn, candidate-aware, graph và CL |

Quy tắc bắt buộc:

- Mỗi variant khởi tạo model và optimizer mới.
- Không load weight từ `supermodel`.
- Không tắt module sau khi đã train `supermodel`.
- Dùng cùng split, seed, negative sampling, training budget và early-stopping rule.
- `supermodel` là control duy nhất của năm ablation trên.

## 5. Dataset và split

### 5.1. Dataset

Chỉ dùng:

```text
MINDsmall_train/
MINDsmall_dev/
```

V10 không dùng MIND-large.

### 5.2. Split chính thức của lần chạy

```text
MINDsmall_train/behaviors.tsv
├── train_core: các ngày sớm hơn
└── validation: ngày cuối hoặc các ngày cuối nguyên vẹn, mục tiêu gần 10%

MINDsmall_dev/behaviors.tsv
└── test: 100% impressions
```

Quy tắc:

- Parse cột `Time`, chuẩn hóa về ngày và sắp xếp các ngày tăng dần.
- Lấy trọn ngày mới nhất vào validation; nếu chưa đạt 10% tổng impressions thì lấy thêm ngày liền trước, cho đến khi đạt hoặc vượt mục tiêu 10%.
- Không cắt một ngày thành cả train và validation. Vì vậy tỷ lệ thực tế có thể không đúng chính xác 90/10 và phải được ghi lại.
- `train_core` chỉ gồm các ngày sớm hơn toàn bộ ngày validation: `max(train_day) < min(validation_day)`.
- Split trước khi sinh negative samples và trước khi xây graph edges.
- Không split theo training sample đã được nhân từ positive clicks.
- Graph adjacency chỉ dùng history và positive clicks thuộc `train_core`.
- Validation và test giữ toàn bộ candidate slate và labels của từng impression.
- `MINDsmall_dev` không dùng để chọn epoch, tune threshold hoặc sửa hyperparameter sau khi xem kết quả.
- Lưu ngày nhỏ nhất/lớn nhất, danh sách ngày, số impression, tỷ lệ thực tế và checksum/index summary của ba split vào artifact để kiểm tra tất cả model dùng cùng split.

Split này deterministic theo dữ liệu và không cần random seed. `SEED = 0` vẫn được dùng cho model initialization, negative sampling và batch shuffle.

## 6. BGE encoding

```python
BGE_MODEL = "BAAI/bge-small-en-v1.5"
```

Quy trình:

1. Đọc union của `MINDsmall_train/news.tsv` và `MINDsmall_dev/news.tsv` theo `news_id`.
2. Ghép `title + abstract` làm input text.
3. Encode đúng một lần bằng `BAAI/bge-small-en-v1.5`.
4. L2-normalize embedding và lưu `float32` trong `news_emb.npz`.
5. Dùng chung file này cho `llmenc_ca`, `supermodel`, `super_no_diff`, `super_no_ca`, `super_no_graph` và `super_no_cl`.
6. `super_no_bge` và baseline không đọc BGE embedding trong forward pass.

Validation trước khi train:

- Coverage tối thiểu 99.9% news ID, không tính PAD.
- Không có NaN hoặc Inf.
- Embedding dimension thống nhất.
- Cache file nếu đã tồn tại và metadata `model_name`, input format và số news khớp; không encode lại vô ích.

Không thử BGE-M3, BGE-base, Jina hoặc model embedding khác trong V10.

## 7. Training protocol

### 7.1. Cấu hình chung

```python
SEED = 0
DIM = 64
BATCH_SIZE = 64
EVAL_BATCH = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
N_NEG = 4
MAX_HIST = 50
TARGET_VAL_RATIO = 0.10

MAX_EPOCHS = 12
MIN_EPOCHS = 3
PATIENCE = 2
MIN_DELTA = 0.001
MONITOR = "ndcg@10"
MODE = "max"
```

V10 chỉ tăng `MAX_EPOCHS` từ 8 lên 12. Các điều kiện early stopping khác giữ nguyên để thay đổi training budget mà không đổi đồng thời quá nhiều yếu tố.

### 7.2. Quy tắc train

Mọi model dùng chung:

- Data object và split.
- Batch construction và negative sampling.
- Optimizer Adam, learning rate và weight decay.
- Batch size.
- Maximum epoch và early stopping.
- Seed.
- Metric implementation.

Sau mỗi epoch:

1. Evaluate toàn bộ validation split.
2. Ghi train loss, validation AUC và validation nDCG@10.
3. Lưu checkpoint nếu validation nDCG@10 tăng hơn best score ít nhất `MIN_DELTA`.
4. Sau `MIN_EPOCHS`, dừng nếu không cải thiện trong `PATIENCE` epoch liên tiếp.
5. Khi kết thúc, load lại checkpoint tốt nhất từ file trên đĩa, không dùng trực tiếp state cuối trong memory.
6. Evaluate `MINDsmall_dev` đúng một lần bằng checkpoint vừa load.
7. Giữ checkpoint và toàn bộ test artifacts sau khi notebook kết thúc.

Không dùng test metric để chọn checkpoint hoặc quyết định tiếp tục train.

### 7.3. Best checkpoint và test artifacts

Mỗi model có đúng một checkpoint được giữ lại:

```text
results/v10_final/models/<model>/best.pt
```

Quy trình bắt buộc:

1. Khi validation nDCG@10 cải thiện, ghi đè atomically `best.pt` của model đó.
2. Lưu `best_meta.json` gồm model name, best epoch, best validation metrics, config và SHA-256 của checkpoint.
3. Sau training, tạo model instance tương ứng và load `best.pt` từ đĩa.
4. Chỉ instance đã load checkpoint này được dùng để chạy final test.
5. Không lưu checkpoint cho mọi epoch và không xóa `best.pt` sau test.

Sau test, lưu:

```text
results/v10_final/models/<model>/test_metrics.json
results/v10_final/models/<model>/test_predictions.npz
```

`test_predictions.npz` phải đủ để tính lại metric mà không chạy inference lại, gồm tối thiểu:

- `impression_ids`.
- `offsets` cho candidate slate có độ dài thay đổi.
- `candidate_news_ids` hoặc global news indices kèm mapping.
- Flattened `labels`.
- Flattened raw `scores` trước khi rank.

`test_metrics.json` lưu AUC, MRR, nDCG@5, nDCG@10, inference time, throughput, checkpoint path và checkpoint SHA-256.

## 8. Logging theo epoch

Mỗi model lưu một record sau mỗi epoch:

```json
{
  "epoch": 1,
  "global_step": 3328,
  "train_loss": 0.0,
  "val_auc": 0.0,
  "val_ndcg@10": 0.0,
  "elapsed_s": 0.0
}
```

`global_step` vẫn được lưu để audit training budget, nhưng không dùng làm trục X của learning curve.

Lưu toàn bộ history tại:

```text
results/v10_final/history.json
```

## 9. Learning curves theo epoch

Dùng `matplotlib` đã có; không thêm plotting dependency.

### 9.1. Validation AUC

```text
results/v10_final/val_auc_vs_epoch.png
```

- Trục X: epoch nguyên `1..epochs_trained`.
- Trục Y: validation AUC.
- Một đường cho mỗi model.
- Một marker tại mỗi epoch.
- Đánh dấu best epoch của mỗi model.
- Không smoothing và không nội suy.

### 9.2. Validation nDCG@10

```text
results/v10_final/val_ndcg10_vs_epoch.png
```

- Trục X: epoch.
- Trục Y: validation nDCG@10.
- Quy tắc hiển thị giống biểu đồ AUC.

Legend phải dùng màu/line style phân biệt được cả 12 model và đặt ngoài plot nếu che dữ liệu. Không vẽ test metrics theo epoch.

## 10. Một bảng kết quả duy nhất

Tạo đúng một bảng, chứa baseline, LLM experiment, Supermodel và mọi ablation. Không tạo bảng ablation riêng.

Sắp xếp giảm dần theo test nDCG@10:

| model | group | control | best epoch | epochs trained | train steps | test AUC | test MRR | test nDCG@5 | test nDCG@10 | ΔAUC | ΔMRR | ΔnDCG@5 | ΔnDCG@10 | params | train s | infer impr/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Quy ước:

- Baseline: `group = baseline`, `control = —`, các cột delta để `—`.
- `llmenc_ca`: `group = llm-experiment`, `control = —`, các cột delta để `—`.
- `supermodel`: `group = proposed`, `control = —`, các cột delta bằng `0.0000`.
- Năm ablation: `group = super-ablation`, `control = supermodel`.
- Với ablation: `delta = test metric(ablation) - test metric(supermodel)`.
- Delta âm nghĩa là bỏ thành phần làm giảm chất lượng, tức thành phần đó có đóng góp tích cực trong lần chạy.

Lưu bảng và dữ liệu nguồn tại:

```text
results/v10_final/results.md
results/v10_final/results.json
```

Không đưa số V8/V9 hoặc kết quả paper vào bảng này.

## 11. Artifacts bắt buộc

```text
results/v10_final/
├── config.json
├── split.json
├── results.json
├── results.md
├── history.json
├── val_auc_vs_epoch.png
├── val_ndcg10_vs_epoch.png
└── models/
    └── <model>/
        ├── best.pt
        ├── best_meta.json
        ├── test_metrics.json
        └── test_predictions.npz
```

Mỗi model phải giữ lại đúng một `best.pt`: checkpoint đã được chọn bằng validation nDCG@10 và đã được load để chạy test. Không giữ checkpoint của epoch khác.

`config.json` phải lưu ít nhất:

- Danh sách model theo đúng thứ tự chạy.
- Seed và hyperparameters.
- Tên BGE model.
- Input text format `title + abstract`.
- Tên dataset `MIND-small`.
- Monitor và early-stopping settings.

## 12. Kiểm tra cuối notebook trên Kaggle

Chỉ kiểm tra artifacts được tạo trong lần chạy hiện tại:

```python
from pathlib import Path

result_dir = Path("results/v10_final")
required = [
    "config.json",
    "split.json",
    "results.json",
    "results.md",
    "history.json",
    "val_auc_vs_epoch.png",
    "val_ndcg10_vs_epoch.png",
]

for filename in required:
    path = result_dir / filename
    assert path.exists() and path.stat().st_size > 0, f"missing artifact: {path}"

for model_name in FINAL_MODELS:
    model_dir = result_dir / "models" / model_name
    for filename in ("best.pt", "best_meta.json", "test_metrics.json", "test_predictions.npz"):
        path = model_dir / filename
        assert path.exists() and path.stat().st_size > 0, f"missing model artifact: {path}"
```

Tuyệt đối không dùng các assertion dạng:

```python
assert Path("newsrec-bench-kaggle-v8-full.ipynb").exists()
assert Path("newsrec-bench-kaggle-v9-final.ipynb").exists()
assert Path("newsrec-bench-kaggle-v10-final.ipynb").exists()
```

Kaggle thực thi notebook nhưng không đảm bảo file `.ipynb` có mặt trong working directory. Notebook phải tự chứa code cần chạy và chỉ xác nhận output artifacts của chính nó.

Các validation khác:

- `len(results) == 12`.
- Tên model trong results khớp chính xác `FINAL_MODELS`, không thừa hoặc thiếu.
- Không có `hybridopt`, `graphrec` hoặc NRMS ablation.
- Mỗi model có ít nhất `MIN_EPOCHS` history records.
- `best_epoch <= epochs_trained <= MAX_EPOCHS`.
- Mọi metric, time và throughput là số hữu hạn.
- Năm ablation đều có `control = supermodel` và delta được tính lại từ raw test metrics.
- Train, validation và test có số impression lớn hơn 0.
- `max(train_day) < min(validation_day)` và không có ngày nào xuất hiện ở cả train lẫn validation.
- Tỷ lệ validation thực tế và danh sách ngày được ghi trong `split.json`.
- Graph edges chỉ đến từ `train_core`.
- Hai PNG có kích thước lớn hơn 0 và mỗi biểu đồ có đủ 12 model trong legend.
- Mỗi model có đúng một `best.pt`, một `best_meta.json`, một `test_metrics.json` và một `test_predictions.npz`.
- SHA-256 trong `best_meta.json` và `test_metrics.json` khớp file `best.pt` đã dùng để test.
- Metrics tính lại từ `test_predictions.npz` khớp `test_metrics.json` trong sai số số học cho phép.

## 13. Kế hoạch triển khai

### Task 1 — Tạo notebook V10

- [ ] Copy nội dung V9 sang `newsrec-bench-kaggle-v10-final.ipynb`.
- [ ] Đổi title, run name và output path sang V10.
- [ ] Giữ notebook self-contained; không đọc hoặc assert notebook V8/V9.

### Task 2 — Thu gọn registry

- [ ] Giữ năm baseline.
- [ ] Giữ `llmenc_ca`.
- [ ] Giữ `supermodel`.
- [ ] Thêm năm `super_no_*`.
- [ ] Loại screening variants khỏi `FINAL_MODELS` và output.

### Task 3 — Implement true ablation

- [ ] Mỗi `super_no_*` thay đúng một thành phần.
- [ ] Reuse module hiện có thay vì tạo framework ablation mới.
- [ ] Mỗi variant train từ initialization mới.
- [ ] Kiểm tra forward output shape giống `supermodel`.

### Task 4 — Data protocol

- [ ] Đọc MIND-small train và dev riêng.
- [ ] Parse và sắp xếp cột `Time` theo ngày.
- [ ] Lấy các ngày cuối nguyên vẹn làm validation, mục tiêu gần 10% impressions.
- [ ] Xác nhận mọi ngày train xảy ra trước mọi ngày validation.
- [ ] Sinh train samples và graph chỉ sau khi split.
- [ ] Giữ toàn bộ MINDsmall_dev làm test.
- [ ] Lưu ngày, tỷ lệ thực tế, số impression và checksum vào `split.json`.

### Task 5 — BGE cache

- [ ] Encode union train/dev news bằng `BAAI/bge-small-en-v1.5`.
- [ ] Dùng `title + abstract`.
- [ ] Kiểm tra coverage, dimension và finite values.
- [ ] Tái sử dụng một file embedding cho mọi model cần BGE.

### Task 6 — Training và checkpoint

- [ ] Tăng max epoch lên 12.
- [ ] Evaluate validation sau mỗi epoch.
- [ ] Early stop và giữ đúng một `best.pt` theo validation nDCG@10 cho mỗi model.
- [ ] Load `best.pt` từ đĩa trước test và lưu checkpoint SHA-256.
- [ ] Evaluate test đúng một lần mỗi model.
- [ ] Lưu `best_meta.json`, `test_metrics.json` và `test_predictions.npz` cho mỗi model.

### Task 7 — Báo cáo

- [ ] Ghi history theo epoch.
- [ ] Tạo một bảng duy nhất gồm đủ 12 model và delta ablation.
- [ ] Vẽ hai learning curves theo epoch.
- [ ] Lưu đủ artifacts chung và bốn artifacts riêng cho mỗi model.

### Task 8 — Kaggle acceptance check

- [ ] Không phụ thuộc tên file notebook.
- [ ] Không còn assertion kiểm tra V8/V9 source.
- [ ] Đủ 12 model rows và không có model ngoài phạm vi.
- [ ] Mọi metric hữu hạn.
- [ ] Hai biểu đồ tồn tại và có đủ model.
- [ ] Đủ 12 best checkpoints và test artifacts tương ứng.
- [ ] Test metrics có thể tính lại từ prediction artifacts.

## 14. Tiêu chí hoàn thành

V10 hoàn thành khi một lần `Run All` trên Kaggle:

1. Dùng MIND-small thật và shared BGE-small-en-v1.5 embeddings.
2. Train thành công đủ 12 cấu hình theo cùng protocol.
3. Mỗi ablation được train lại từ đầu và chỉ bỏ một thành phần của Supermodel.
4. Split train/validation theo ngày, với toàn bộ ngày train xảy ra trước validation.
5. Lưu và load đúng một best checkpoint theo validation nDCG@10 cho mỗi model trước khi test.
6. Lưu test predictions và test metrics tương ứng với checkpoint đã chọn.
7. Xuất đúng một bảng tổng hợp có metric và delta.
8. Xuất hai learning curves theo epoch.
9. Pass toàn bộ artifact checks mà không cần file notebook nguồn tồn tại.

## 15. Ngoài phạm vi V10

- Không chạy MIND-large.
- Không dùng BGE-M3, BGE-base, Jina hoặc embedding model khác.
- Không chạy `hybridopt`, `graphrec` hoặc NRMS ablation.
- Không tune hyperparameter riêng cho từng model.
- Không dùng test set để chọn checkpoint.
- Không inference-time masking thay cho retraining ablation.
- Không tạo bảng ablation riêng.
- Không thêm hard-negative mining, FAISS, LoRA, generative reranking hoặc architecture mới.
- Không yêu cầu file `.ipynb` nguồn tồn tại trong Kaggle working directory.

V10 chỉ trả lời hai câu hỏi: dưới cùng MIND-small protocol, mô hình nào tốt nhất; và khi train lại từ đầu, mỗi thành phần của Supermodel đóng góp bao nhiêu?
