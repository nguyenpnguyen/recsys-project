# Kế hoạch demo V10

## Mục tiêu

Tạo một demo nghiên cứu chạy cục bộ trên MIND-small bằng kết quả và checkpoint V10 hiện có. Demo có phần tổng quan benchmark, phát lại impression đã đánh giá và một luồng tương tác nhỏ cho người xem tự chọn lịch sử đọc cùng tập tin ứng viên. Không train lại model.

## Phạm vi

- Dùng dữ liệu `dataset/MINDsmall_train/`, `dataset/MINDsmall_dev/`, embedding `inference/artifacts/news_emb.npz` và artifact trong `inference/artifacts/v10_final/`.
- Trang tổng quan hiển thị bảng kết quả đủ 12 model và hai biểu đồ validation AUC, nDCG@10 đã lưu.
- Phát lại dùng score đã lưu; model mặc định là NAML, LLMEncCA, Supermodel và `super_no_cl`. Người xem có thể chọn thêm model V10 khác.
- Tương tác tùy chỉnh dùng checkpoint `llmenc_ca` để xếp hạng 1–50 bài lịch sử và 2–50 bài ứng viên lấy từ MIND-small.
- Chạy cục bộ trên CPU, không gọi dịch vụ ngoài, không chạy BGE lúc mở ứng dụng và không công khai dữ liệu MIND.
- Đây là reranker trên kho MIND năm 2019, không phải hệ thống lấy tin thời gian thực. Tình huống tương tác do người xem tạo không có ground truth và không được gắn metric đánh giá.

## Cấu phần và file

| File | Trách nhiệm |
| --- | --- |
| `PLAN_DEMO_V10.md` | Đặc tả, giới hạn và tiêu chí nghiệm thu của demo. |
| `demo_core.py` | Đọc catalog/impression, xác minh mapping và artifact, cung cấp API nội bộ để phát lại ranking và chạy LLMEncCA. Không chứa UI. |
| `demo_app.py` | Ứng dụng Streamlit với màn hình Tổng quan, Phát lại và Tương tác. |
| `test_demo.py` | Kiểm tra dữ liệu, đối chiếu score checkpoint với score V10 và kiểm tra input sai bằng `unittest`. |
| `requirements.txt` | Các dependency trực tiếp và phiên bản đã xác minh trên Python 3.12. |

Tái sử dụng `training/models.py` và `training/metrics.py` mà không sửa. Các thư mục dữ liệu, embedding, checkpoint và kết quả được mở chỉ đọc.

## Dữ liệu và API nội bộ

`demo_core.py` cung cấp các cấu trúc/API tối thiểu:

- `load_demo_data(root)`: đọc metadata bài từ hai `news.tsv`, đọc impression từ `MINDsmall_dev/behaviors.tsv`, ánh xạ news ID và xác minh thứ tự 65.238 ID khớp `news_emb.npz`.
- `PredictionStore(root, data)`: nạp kết quả tổng hợp; giữ offset, ID impression, candidate index và nhãn từ một file tham chiếu; nạp score từng model theo yêu cầu và xác minh slate/nhãn khớp file tham chiếu. Cache các mảng score đã dùng.
- `load_llmenc_ca(root)`: xác minh SHA-256 theo `best_meta.json`, nạp checkpoint CPU với `weights_only=True`, dựng đúng cấu hình từ `config.json`, đặt model ở chế độ inference.
- `score_custom(model, data, history_ids, candidate_ids)`: kiểm tra ID/giới hạn/trùng lặp, pad history tới 50 phần tử như lúc train, trả score theo thứ tự candidate đầu vào.
- `impression_metrics(labels, scores)`: gọi metric hiện có để tính MRR và nDCG@5/@10 sau khi người xem yêu cầu hiện nhãn.

Prediction NPZ phải có `impression_ids`, `offsets`, `candidate_news_indices`, `labels`, `scores`; offsets phải bắt đầu bằng 0, không giảm, và kết thúc tại số candidate. ID impression phải duy nhất. Khi phát lại, dữ liệu behaviors phải khớp ID, candidate index và labels. Nếu thiếu file, checkpoint sai hash, news mapping sai hoặc artifact không đồng bộ, dừng và nêu rõ lỗi.

## Luồng giao diện

### Tổng quan

- Hiển thị bảng 12 model từ `results.json`: nhóm, best epoch, test AUC/MRR/nDCG@5/nDCG@10, tham số, thời gian train và tốc độ suy luận.
- Hiển thị trực tiếp `val_auc_vs_epoch.png` và `val_ndcg10_vs_epoch.png`.
- Giải thích: baseline là NRMS, NAML, Fastformer, CAUM, LightGCN; LLMEncCA là thí nghiệm encoding; Supermodel và các biến thể là nhóm proposed/ablation.

### Phát lại impression

- Chọn ID 1–73.152 hoặc lấy impression ngẫu nhiên; ẩn user ID.
- Hiện timestamp, lịch sử đọc và metadata bài ứng viên.
- Cho chọn model; mặc định NAML, LLMEncCA, Supermodel, `super_no_cl`. Hiển thị thứ hạng từng model trong một bảng, không so trực tiếp score giữa các model.
- Nhãn click bị ẩn ban đầu. Khi bấm “Hiện click thực tế”, hiển thị nhãn và MRR/nDCG của impression đó.

### Tương tác

- Tìm trong title/category của kho bài MIND; chỉ hiển thị tối đa 50 kết quả cho mỗi truy vấn.
- Cho thêm/xóa bài trong lịch sử và candidate pool; chặn bài trùng, giao giữa hai pool, lịch sử rỗng, trên 50 bài lịch sử, dưới 2 hoặc trên 50 bài ứng viên.
- Nút “Xếp hạng” gọi LLMEncCA CPU; hiển thị rank, title, category và score theo thứ tự giảm dần.
- Không hiện labels hay metric trong luồng này. Thay đổi lịch sử/candidate làm mất kết quả ranking cũ cho tới lần bấm xếp hạng tiếp theo.

## Thứ tự triển khai

1. Tạo môi trường riêng Python 3.12 và `requirements-demo.txt`; ghi các phiên bản dependency đã cài/chạy.
2. Viết loader, ánh xạ ID, validation artifact và `PredictionStore` trong `demo_core.py`.
3. Viết suy luận CPU cho LLMEncCA, khớp mapping và score với dữ liệu V10.
4. Viết ba màn hình trong `demo_app.py`; cache catalog, prediction store và model đã nạp.
5. Viết `test_demo.py`, chạy unit checks, mở app cục bộ và kiểm tra các luồng chính.

## Cài đặt và chạy cục bộ

```bash
uv venv --python 3.12 .venv-demo
uv pip install --python .venv-demo/bin/python -r inference/requirements.txt
.venv-demo/bin/python -m streamlit run inference/demo_app.py --server.address 127.0.0.1
```

Streamlit chỉ bind loopback; không công khai app trên mạng LAN hoặc Internet.

## Kiểm tra và tiêu chí hoàn thành

- `cd inference && python -m unittest test_demo.py` chạy thành công.
- Có đúng 73.152 impression duy nhất; catalog news có đúng 65.238 ID theo cùng thứ tự embedding; predictions khớp behaviors và cùng candidate slate.
- Trên một số impression dev có history, suy luận CPU từ checkpoint LLMEncCA khớp score lưu V10 trong sai số số học `atol=5e-4`, `rtol=1e-4` và cho cùng thứ hạng.
- Kiểm tra ID không tồn tại, history rỗng, bài trùng, history/candidate giao nhau, giới hạn số bài và artifact thiếu/sai hash.
- `python -m streamlit run demo_app.py` mở được cả ba màn hình; không cần Kaggle, GPU, train lại hoặc kết nối mạng ở thời điểm chạy.

## Giả định

- Người xem là hội đồng/người quan tâm nghiên cứu; ngôn ngữ UI là tiếng Việt, tiêu đề tin giữ nguyên tiếng Anh.
- Bản đầu dùng artifact V10; hướng novelty sẽ được thử nghiệm và đánh giá riêng.
- Supermodel và `super_no_cl` chỉ phát lại kết quả cho impression MIND có sẵn; luồng tùy chỉnh dùng LLMEncCA vì nhánh graph phụ thuộc user ID lúc train.
- Dữ liệu MIND chỉ dùng cục bộ, không kèm vào gói phát hành công khai.
