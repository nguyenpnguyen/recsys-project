"""Local Streamlit research demo for the V10 MIND-small benchmark."""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import streamlit as st

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


st.set_page_config(page_title="News Recommendation · V10", layout="wide")


@st.cache_resource
def get_data():
    return load_demo_data(PROJECT_ROOT)


@st.cache_resource
def get_prediction_store(_data):
    return PredictionStore(_data)


@st.cache_resource
def get_interactive_model(_data):
    return load_llmenc_ca(_data)


def article_title(data, news_id: str) -> str:
    article = data.news_by_id[news_id]
    return f"{article.title} · {article.category}/{article.subcategory} [{news_id}]"


def _set_random_impression(impression_ids):
    st.session_state["replay_input_id"] = random.choice(impression_ids)


def _overview(data):
    st.subheader("Benchmark V10 trên MIND-small")
    st.caption("Bảng là kết quả test lưu từ V10; biểu đồ là learning curve trên validation.")
    columns = [
        ("model", "Model"), ("group", "Nhóm"), ("best_epoch", "Epoch tốt nhất"),
        ("test_auc", "Test AUC"), ("test_mrr", "Test MRR"),
        ("test_ndcg@5", "Test nDCG@5"), ("test_ndcg@10", "Test nDCG@10"),
        ("params", "Tham số"), ("train_s", "Train (s)"),
        ("infer_impr/s", "Impression/s"),
    ]
    table = [{label: row[key] for key, label in columns} for row in data.result_rows]
    st.dataframe(table, width="stretch", hide_index=True)
    chart_cols = st.columns(2)
    for col, filename, title in (
        (chart_cols[0], "val_auc_vs_epoch.png", "Validation AUC theo epoch"),
        (chart_cols[1], "val_ndcg10_vs_epoch.png", "Validation nDCG@10 theo epoch"),
    ):
        path = data.root / "inference" / "artifacts" / "v10_final" / filename
        if path.is_file() and path.stat().st_size:
            col.image(str(path), caption=title, width="stretch")
        else:
            col.error(f"Thiếu biểu đồ: {path}")
    st.markdown(
        "**Nhóm model:** baseline truyền thống gồm NRMS, NAML, Fastformer, CAUM và LightGCN; "
        "LLMEncCA là thí nghiệm encoding; Supermodel và `super_no_*` là model đề xuất/ablation."
    )
    st.info("MIND-small là dữ liệu lưu trữ năm 2019. Demo xếp hạng trong tập ứng viên đã cho, không lấy tin trực tuyến.")


def _replay(data, store):
    st.subheader("Phát lại một impression của MINDsmall_dev")
    col_id, col_random = st.columns([3, 1])
    impression_ids = [int(value) for value in store.impression_ids]
    max_id = max(impression_ids)
    st.session_state.setdefault("replay_input_id", min(impression_ids))
    with col_id:
        impression_number = st.number_input(
            "Impression ID", min_value=min(impression_ids), max_value=max_id,
            step=1, key="replay_input_id",
        )
    with col_random:
        st.write("")
        st.write("")
        st.button("Chọn ngẫu nhiên", on_click=_set_random_impression, args=(impression_ids,))
    chosen_id = str(int(impression_number))
    row = data.impression_by_id.get(chosen_id)
    if row is None:
        st.error(f"Không tìm thấy impression {chosen_id} trong MINDsmall_dev.")
        return
    st.caption(f"Impression {row.impression_id} · thời điểm {row.timestamp} · {len(row.candidate_ids)} ứng viên")

    with st.expander(f"Lịch sử đọc ({len(row.history_ids)} bài)", expanded=False):
        if row.history_ids:
            st.dataframe(
                [{"#": i + 1, "Bài đã đọc": article_title(data, news_id)}
                 for i, news_id in enumerate(row.history_ids[-50:])],
                width="stretch", hide_index=True,
            )
        else:
            st.write("Impression này không có lịch sử đọc.")

    selected_models = st.multiselect(
        "Model so sánh", options=list(data.model_names), default=list(DEFAULT_REPLAY_MODELS),
        format_func=lambda name: name,
    )
    if not selected_models:
        st.warning("Chọn ít nhất một model để hiển thị thứ hạng.")
        return
    try:
        model_scores = {name: store.scores(name) for name in selected_models}
        start, end = store.span(chosen_id)
    except DemoDataError as exc:
        st.error(str(exc))
        return

    rank_by_model = {}
    for model_name, all_scores in model_scores.items():
        order = np.argsort(all_scores[start:end])[::-1]
        ranks = np.empty(len(order), dtype=np.int64)
        ranks[order] = np.arange(1, len(order) + 1)
        rank_by_model[model_name] = ranks

    reveal_key = "revealed_impression_id"
    if st.button("Hiện click thực tế", key=f"reveal-{chosen_id}"):
        st.session_state[reveal_key] = chosen_id
    show_labels = st.session_state.get(reveal_key) == chosen_id
    candidate_rows = []
    for position, news_id in enumerate(row.candidate_ids):
        article = data.news_by_id[news_id]
        item = {
            "Bài báo": article.title,
            "Chủ đề": article.category,
            "News ID": news_id,
        }
        for model_name in selected_models:
            item[f"{model_name} rank"] = int(rank_by_model[model_name][position])
        if show_labels:
            item["Click thực tế"] = "Có" if row.labels[position] else "Không"
        candidate_rows.append(item)
    candidate_rows.sort(key=lambda item: item[f"{selected_models[0]} rank"])
    st.dataframe(candidate_rows, width="stretch", hide_index=True)
    if show_labels:
        st.caption("Metric tính trên impression này; nhãn không tham gia tạo thứ hạng.")
        metric_rows = []
        for model_name, all_scores in model_scores.items():
            values = impression_metrics(row.labels, all_scores[start:end])
            metric_rows.append({"Model": model_name, **values})
        st.dataframe(metric_rows, width="stretch", hide_index=True)


def _remove_selector(label, state_key, articles):
    values = st.session_state[state_key]
    if not values:
        st.caption(f"{label}: chưa chọn bài.")
        return
    selected = st.selectbox(
        label, options=values,
        format_func=lambda news_id: article_title(articles, news_id),
        key=f"remove-select-{state_key}",
    )
    if st.button(f"Xóa bài khỏi {label.lower()}", key=f"remove-{state_key}"):
        st.session_state[state_key] = [news_id for news_id in values if news_id != selected]
        st.session_state.pop("custom_result", None)
        st.rerun()


def _interactive(data):
    st.subheader("Thử lịch sử và tập ứng viên")
    st.caption("Tất cả bài lấy từ kho MIND-small. Kết quả là xếp hạng minh họa, không có nhãn click để đánh giá.")
    st.session_state.setdefault("custom_history", [])
    st.session_state.setdefault("custom_candidates", [])

    query_col, category_col = st.columns([3, 1])
    with query_col:
        query = st.text_input("Tìm theo tiêu đề", placeholder="Nhập ít nhất 2 ký tự")
    categories = ["Tất cả"] + sorted({article.category for article in data.news_by_id.values()})
    with category_col:
        category = st.selectbox("Chủ đề", categories)
    selected_id = None
    if len(query.strip()) >= 2:
        normalized = query.strip().casefold()
        matches = [
            article for article in data.news_by_id.values()
            if (category == "Tất cả" or article.category == category)
            and normalized in article.title.casefold()
        ][:50]
        if matches:
            selected_id = st.selectbox(
                "Kết quả tìm kiếm", options=[article.news_id for article in matches],
                format_func=lambda news_id: article_title(data, news_id),
            )
            add_history, add_candidates = st.columns(2)
            history = st.session_state["custom_history"]
            candidates = st.session_state["custom_candidates"]
            with add_history:
                disabled = selected_id in history or selected_id in candidates or len(history) >= 50
                if st.button("Thêm vào lịch sử", disabled=disabled):
                    history.append(selected_id)
                    st.session_state.pop("custom_result", None)
                    st.rerun()
            with add_candidates:
                disabled = selected_id in candidates or selected_id in history or len(candidates) >= 50
                if st.button("Thêm vào ứng viên", disabled=disabled):
                    candidates.append(selected_id)
                    st.session_state.pop("custom_result", None)
                    st.rerun()
        else:
            st.info("Không có bài phù hợp với tiêu đề và chủ đề đã chọn.")
    else:
        st.caption("Tìm kiếm trong title của 65.238 bài; abstract được giữ trong catalog để tham khảo.")

    history_col, candidate_col = st.columns(2)
    with history_col:
        st.markdown(f"**Lịch sử ({len(st.session_state['custom_history'])}/50)**")
        _remove_selector("Lịch sử", "custom_history", data)
    with candidate_col:
        st.markdown(f"**Ứng viên ({len(st.session_state['custom_candidates'])}/50)**")
        _remove_selector("Ứng viên", "custom_candidates", data)

    history = st.session_state["custom_history"]
    candidates = st.session_state["custom_candidates"]
    can_rank = 1 <= len(history) <= 50 and 2 <= len(candidates) <= 50
    if st.button("Xếp hạng bằng LLMEncCA", disabled=not can_rank):
        try:
            model = get_interactive_model(data)
            scores = score_custom(model, data, history, candidates)
            st.session_state["custom_result"] = {
                "signature": (tuple(history), tuple(candidates)),
                "scores": scores.tolist(),
            }
        except (DemoDataError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
    else:
        if not can_rank:
            st.caption("Chọn 1–50 bài trong lịch sử và 2–50 bài ứng viên để bật xếp hạng.")

    result = st.session_state.get("custom_result")
    if result and result["signature"] == (tuple(history), tuple(candidates)):
        order = np.argsort(np.asarray(result["scores"]))[::-1]
        st.dataframe([
            {
                "Rank": rank + 1,
                "Bài báo": data.news_by_id[candidates[position]].title,
                "Chủ đề": data.news_by_id[candidates[position]].category,
                "Score": result["scores"][position],
            }
            for rank, position in enumerate(order)
        ], width="stretch", hide_index=True)


def main():
    st.title("News Recommendation · V10")
    st.caption("Demo nghiên cứu cục bộ trên MIND-small · dữ liệu lưu trữ năm 2019")
    try:
        data = get_data()
        store = get_prediction_store(data)
    except (DemoDataError, OSError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    overview, replay, interactive = st.tabs(["Tổng quan", "Phát lại impression", "Thử lịch sử"])
    with overview:
        _overview(data)
    with replay:
        _replay(data, store)
    with interactive:
        _interactive(data)


if __name__ == "__main__":
    main()
