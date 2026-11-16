from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "mo_hinh_nam"
DATA_PATH = ROOT / "mushrooms.csv"

st.set_page_config(
    page_title="Mushroom Classifier | Nhận diện nấm",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --forest:#173d32; --moss:#426b4f; --cream:#f7f5ee; --ink:#20332b; --muted:#748078; --line:#e4e8df; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f4f5ee 0%,#fbfaf6 38%,#f7f5ee 100%); color:var(--ink); }
.block-container { max-width:1180px; padding-top:2rem; padding-bottom:3rem; }
#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent; }
.hero { position:relative; overflow:hidden; padding:34px 38px; border-radius:26px; color:#f8f7ef;
  background:radial-gradient(circle at 84% 20%,rgba(203,174,107,.32),transparent 25%),linear-gradient(125deg,#15382f 0%,#24533e 58%,#476c4e 100%);
  box-shadow:0 18px 50px rgba(26,61,48,.16); margin-bottom:22px; }
.hero:after { content:'🍄'; position:absolute; right:7%; top:6%; font-size:112px; opacity:.16; transform:rotate(12deg); }
.eyebrow { letter-spacing:.16em; text-transform:uppercase; font-size:11px; font-weight:700; color:#d6c99f; }
.hero h1 { font-family:'Manrope',sans-serif; font-size:clamp(30px,4vw,48px); line-height:1.08; margin:12px 0 10px; color:#fffdf5; }
.hero p { max-width:690px; color:#e1e9df; margin:0; font-size:15px; line-height:1.65; }
.pill { display:inline-block; border:1px solid rgba(255,255,255,.25); border-radius:999px; padding:7px 12px; margin-top:20px; color:#f4f0df; font-size:12px; }
.section-title { font-family:'Manrope',sans-serif; color:var(--forest); font-size:22px; font-weight:800; margin:10px 0 4px; }
.section-sub { color:var(--muted); margin:0 0 16px; font-size:14px; }
.panel { padding:20px 22px; border:1px solid var(--line); background:rgba(255,255,255,.78); border-radius:18px; box-shadow:0 6px 22px rgba(36,54,40,.045); }
.result-card { border:1px solid var(--line); background:#fff; border-radius:18px; padding:18px; min-height:160px; box-shadow:0 8px 24px rgba(35,55,42,.055); }
.result-label { color:#66736b; font-size:12px; font-weight:700; letter-spacing:.04em; text-transform:uppercase; }
.result-class { font-family:'Manrope',sans-serif; font-size:22px; font-weight:800; margin:10px 0 5px; }
.result-meta { color:#6d7971; font-size:13px; }
.safe-note { padding:13px 16px; border-left:4px solid #c49743; border-radius:10px; background:#fbf5e7; color:#594b2d; font-size:13px; line-height:1.55; }
.footer-note { text-align:center; padding:22px 8px 4px; color:#8a948b; font-size:12px; }
div[data-testid="stForm"] { border:0; padding:0; }
div[data-testid="stFormSubmitButton"] button, .stButton button { border-radius:12px; font-weight:700; min-height:44px; }
.stButton button[kind="primary"], div[data-testid="stFormSubmitButton"] button { background:#254d3b; border-color:#254d3b; }
div[data-testid="stSelectbox"] label { color:#344a3c; font-weight:600; }
@media (max-width:700px) { .hero { padding:26px 22px; } .hero:after { right:2%; top:2%; font-size:75px; } .block-container { padding-left:1rem; padding-right:1rem; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


FEATURE_LABELS = {
    "cap-shape": "Hình dạng mũ",
    "cap-surface": "Bề mặt mũ",
    "cap-color": "Màu mũ",
    "bruises": "Vết bầm trên mũ",
    "odor": "Mùi",
    "gill-attachment": "Kiểu gắn phiến",
    "gill-spacing": "Khoảng cách phiến",
    "gill-size": "Kích thước phiến",
    "gill-color": "Màu phiến",
    "stalk-shape": "Hình dạng cuống",
    "stalk-root": "Gốc cuống",
    "stalk-surface-above-ring": "Bề mặt cuống phía trên vòng",
    "stalk-surface-below-ring": "Bề mặt cuống phía dưới vòng",
    "stalk-color-above-ring": "Màu cuống phía trên vòng",
    "stalk-color-below-ring": "Màu cuống phía dưới vòng",
    "veil-color": "Màu màng nấm",
    "ring-number": "Số vòng",
    "ring-type": "Loại vòng",
    "spore-print-color": "Màu bào tử",
    "population": "Kiểu quần thể",
    "habitat": "Môi trường sống",
}

COLOR_NAMES = {
    "n": "Nâu", "b": "Vàng da bò", "c": "Quế", "g": "Xám",
    "r": "Xanh lá", "p": "Hồng", "u": "Tím", "e": "Đỏ",
    "w": "Trắng", "y": "Vàng", "k": "Đen", "o": "Cam",
    "h": "Nâu sô-cô-la",
}

VALUE_LABELS = {
    "cap-shape": {"b":"Hình chuông", "c":"Hình nón", "x":"Lồi", "f":"Phẳng", "k":"Có núm", "s":"Lõm"},
    "cap-surface": {"f":"Dạng sợi", "g":"Có rãnh", "y":"Có vảy", "s":"Nhẵn"},
    "bruises": {"t":"Có vết bầm", "f":"Không có vết bầm"},
    "odor": {"a":"Mùi hạnh nhân", "l":"Mùi hồi", "c":"Mùi creosote", "y":"Mùi tanh", "f":"Mùi hôi", "m":"Mùi mốc", "n":"Không mùi", "p":"Mùi hăng", "s":"Mùi cay"},
    "gill-attachment": {"a":"Gắn liền", "d":"Phiến chạy xuống cuống", "f":"Tách khỏi cuống", "n":"Có khía"},
    "gill-spacing": {"c":"Phiến sít", "w":"Phiến dày", "d":"Phiến thưa"},
    "gill-size": {"b":"Rộng", "n":"Hẹp"},
    "stalk-shape": {"e":"Phình ở dưới", "t":"Thuôn ở dưới"},
    "stalk-root": {"b":"Phình củ", "c":"Dạng chùy", "u":"Dạng cốc", "e":"Đều", "z":"Có rễ dạng sợi", "r":"Có rễ", "?":"Không rõ — dùng giá trị thay thế"},
    "stalk-surface-above-ring": {"f":"Dạng sợi", "y":"Có vảy", "k":"Dạng lụa", "s":"Nhẵn"},
    "stalk-surface-below-ring": {"f":"Dạng sợi", "y":"Có vảy", "k":"Dạng lụa", "s":"Nhẵn"},
    "ring-number": {"n":"Không có vòng", "o":"Một vòng", "t":"Hai vòng"},
    "ring-type": {"c":"Dạng mạng", "e":"Mau biến mất", "f":"Xòe rộng", "l":"Lớn", "n":"Không có vòng", "p":"Dạng rủ", "s":"Bao quanh cuống", "z":"Có vân"},
    "population": {"a":"Mọc nhiều", "c":"Mọc thành cụm", "n":"Mọc dày", "s":"Rải rác", "v":"Mọc thành nhóm", "y":"Mọc đơn lẻ"},
    "habitat": {"g":"Đồng cỏ", "l":"Lá cây", "m":"Bãi cỏ tự nhiên", "p":"Lối đi", "u":"Khu đô thị", "w":"Nơi phế thải", "d":"Rừng"},
}

for color_column in (
    "cap-color", "gill-color", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-color", "spore-print-color",
):
    VALUE_LABELS[color_column] = COLOR_NAMES


def display_value(column: str, value: str) -> str:
    if value == "?":
        return "Không rõ — dùng giá trị thay thế (?)"
    meaning = VALUE_LABELS.get(column, {}).get(value, value)
    return f"{meaning}  ·  mã {value}"


@st.cache_resource(show_spinner="Đang nạp các mô hình đã huấn luyện…")
def load_assets():
    config_path = MODEL_DIR / "thong_tin.json"
    preprocessing_path = MODEL_DIR / "tien_xu_ly.pkl"
    encoder_path = MODEL_DIR / "encoder.keras"
    required_paths = [
        config_path,
        preprocessing_path,
        encoder_path,
        MODEL_DIR / "A_KNN.pkl",
        MODEL_DIR / "A_GaussianNB.pkl",
        MODEL_DIR / "B_KNN.pkl",
        MODEL_DIR / "B_GaussianNB.pkl",
    ]
    missing = [path.name for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Thiếu tệp trong thư mục mo_hinh_nam: " + ", ".join(missing)
        )

    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    preprocessing = joblib.load(preprocessing_path)
    encoder = tf.keras.models.load_model(encoder_path, compile=False)
    models = {
        "A - KNN": joblib.load(MODEL_DIR / "A_KNN.pkl"),
        "A - GaussianNB": joblib.load(MODEL_DIR / "A_GaussianNB.pkl"),
        "B - KNN": joblib.load(MODEL_DIR / "B_KNN.pkl"),
        "B - GaussianNB": joblib.load(MODEL_DIR / "B_GaussianNB.pkl"),
    }
    return config, preprocessing, encoder, models


@st.cache_data(show_spinner=False)
def load_demo_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH, dtype=str)


st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Dự án học máy · Đề tài 09</div>
      <h1>Nhận diện nấm<br>qua đặc điểm quan sát</h1>
      <p>Chọn các đặc trưng của mẫu nấm để xem bốn cấu hình mô hình dự đoán lớp Ăn được hoặc Độc. Có thể đối chiếu đặc trưng One-Hot với biểu diễn được học bằng Autoencoder.</p>
      <span class="pill">🍃 Luồng A · One-Hot Encoding</span>
      <span class="pill">🧠 Luồng B · Autoencoder</span>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    config, preprocessing, encoder, models = load_assets()
except Exception as error:
    st.error("Chưa nạp được mô hình. " + str(error))
    st.info(
        "Hãy tải thư mục **mo_hinh_nam** lên GitHub cùng cấp với app.py. "
        "Thư mục cần có `thong_tin.json`, `tien_xu_ly.pkl`, `encoder.keras` "
        "và bốn tệp mô hình `.pkl`."
    )
    st.stop()

feature_columns = preprocessing["cot_giu"]
valid_values = config["gia_tri_nhap"]

st.markdown('<div class="section-title">Thông tin đầu vào</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Chọn đặc điểm quan sát được. Mỗi lựa chọn hiển thị tên tiếng Việt và mã gốc để bạn dễ đối chiếu với bộ dữ liệu.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    top_left, top_right = st.columns([1.25, 1])
    with top_left:
        st.markdown("**Mẫu minh họa**")
        st.caption("Có thể nạp ngẫu nhiên một dòng từ mushrooms.csv để thử nhanh.")
    with top_right:
        if st.button("🎲 Nạp mẫu minh họa", use_container_width=True):
            demo_data = load_demo_data()
            if demo_data is None:
                st.warning("Không tìm thấy mushrooms.csv ở thư mục gốc repository.")
            else:
                sample = demo_data.sample(n=1).iloc[0]
                sample_values = {}
                for column in feature_columns:
                    raw_value = str(sample[column])
                    if raw_value not in [str(v) for v in valid_values[column]]:
                        raw_value = str(valid_values[column][0])
                    widget_key = f"mushroom_{column}"
                    st.session_state[widget_key] = raw_value
                    sample_values[column] = raw_value
                st.session_state["demo_sample"] = {
                    "values": sample_values,
                    "label": str(sample["class"]),
                }
                st.session_state.pop("latest_predictions", None)
                st.rerun()

    st.markdown("---")
    groups = [
        ("🍄 Mũ nấm", ["cap-shape", "cap-surface", "cap-color", "bruises", "odor"]),
        ("🌿 Phiến nấm", ["gill-attachment", "gill-spacing", "gill-size", "gill-color"]),
        ("🪵 Cuống nấm", ["stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring"]),
        ("⭕ Màng và vòng", ["veil-color", "ring-number", "ring-type"]),
        ("🌲 Bào tử và môi trường", ["spore-print-color", "population", "habitat"]),
    ]

    submitted = False
    with st.form("mushroom-input-form"):
        input_values = {}
        for group_title, group_columns in groups:
            present_columns = [c for c in group_columns if c in feature_columns]
            if not present_columns:
                continue
            st.markdown(f"#### {group_title}")
            columns = st.columns(3)
            for index, column in enumerate(present_columns):
                options = [str(value) for value in valid_values[column]]
                if column == "stalk-root" and "?" not in options:
                    options.append("?")
                key = f"mushroom_{column}"
                if key not in st.session_state or st.session_state[key] not in options:
                    st.session_state[key] = options[0]
                with columns[index % 3]:
                    st.selectbox(
                        FEATURE_LABELS.get(column, column),
                        options=options,
                        format_func=lambda value, c=column: display_value(c, value),
                        key=key,
                    )
                input_values[column] = None
        st.markdown(" ")
        submitted = st.form_submit_button(
            "✨ Dự đoán với các mô hình",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        row_values = {
            column: (np.nan if st.session_state[f"mushroom_{column}"] == "?"
                     else str(st.session_state[f"mushroom_{column}"]))
            for column in feature_columns
        }
        row = pd.DataFrame([row_values], columns=feature_columns)
        try:
            filled = preprocessing["imputer"].transform(row)
            features_a = preprocessing["onehot"].transform(filled)
            features_b = encoder.predict(features_a, verbose=0)
            feature_data = {"A": features_a, "B": features_b}
            output_rows = []

            for model_name, model in models.items():
                stream = model_name.split(" - ", 1)[0]
                model_input = feature_data[stream]
                prediction = int(model.predict(model_input)[0])
                class_position = list(model.classes_).index(1)
                poison_probability = float(model.predict_proba(model_input)[0, class_position])
                output_rows.append({
                    "Mô hình": model_name,
                    "Dự đoán": "Độc" if prediction == 1 else "Ăn được",
                    "Xác suất lớp Độc": poison_probability,
                })

            st.session_state["latest_predictions"] = output_rows
            sample_info = st.session_state.get("demo_sample")
            current_values = {c: st.session_state[f"mushroom_{c}"] for c in feature_columns}
            same_as_sample = bool(
                sample_info
                and all(current_values[c] == sample_info["values"].get(c) for c in feature_columns)
            )
            st.session_state["sample_is_current"] = same_as_sample
        except Exception as error:
            st.error(f"Không thể dự đoán: {error}")


predictions = st.session_state.get("latest_predictions")
if predictions:
    st.markdown('<div class="section-title">Kết quả dự đoán</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Kết quả được tính lại khi bạn bấm nút dự đoán. Nếu thay đổi lựa chọn, hãy bấm nút một lần nữa để cập nhật.</div>',
        unsafe_allow_html=True,
    )

    current_sample = st.session_state.get("demo_sample")
    if current_sample and st.session_state.get("sample_is_current"):
        actual = "Độc" if current_sample["label"] == "p" else "Ăn được"
        st.info(f"Mẫu minh họa có nhãn trong dữ liệu là **{actual}**. Nhãn này chỉ dùng đối chiếu khi đầu vào vẫn giữ nguyên như mẫu vừa nạp.")
    elif current_sample:
        st.caption("Bạn đã chỉnh đầu vào sau khi nạp mẫu; nhãn gốc của mẫu không còn được dùng để đối chiếu.")

    best_display = predictions[0]
    if all(item["Dự đoán"] == predictions[0]["Dự đoán"] for item in predictions):
        agreement_text = "Cả bốn mô hình đang dự đoán cùng một lớp."
    else:
        agreement_text = "Các mô hình chưa thống nhất; hãy xem bảng so sánh bên dưới."
    accent = "#a63835" if best_display["Dự đoán"] == "Độc" else "#276342"
    st.markdown(
        f"""<div class="panel" style="margin:8px 0 16px;border-left:5px solid {accent};">
        <div class="result-label">Tóm tắt</div>
        <div class="result-class" style="color:{accent};">{best_display['Dự đoán']}</div>
        <div class="result-meta">{agreement_text}</div></div>""",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    for column, result in zip(cols, predictions):
        color = "#a63835" if result["Dự đoán"] == "Độc" else "#276342"
        with column:
            st.markdown(
                f"""<div class="result-card">
                <div class="result-label">{result['Mô hình']}</div>
                <div class="result-class" style="color:{color};">{result['Dự đoán']}</div>
                <div class="result-meta">Xác suất lớp Độc<br><b>{result['Xác suất lớp Độc']:.1%}</b></div>
                </div>""",
                unsafe_allow_html=True,
            )

    comparison = pd.DataFrame(predictions)
    comparison["Xác suất lớp Độc"] = comparison["Xác suất lớp Độc"].map(lambda value: f"{value:.2%}")
    st.markdown("#### Bảng đối chiếu bốn cấu hình")
    st.dataframe(comparison, hide_index=True, use_container_width=True)

st.markdown(
    '<div class="safe-note"><b>Lưu ý:</b> Đây là sản phẩm học tập để minh họa mô hình phân loại trên bộ dữ liệu. Không dùng dự đoán này để xác định nấm ngoài thực tế có an toàn để ăn hay không.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="footer-note">Đề tài 09 · Phân loại nấm · Luồng A: One-Hot Encoding · Luồng B: Autoencoder</div>',
    unsafe_allow_html=True,
)
