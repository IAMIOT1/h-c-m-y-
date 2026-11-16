from __future__ import annotations

import json
import io
import hashlib
from PIL import Image, ImageOps, UnidentifiedImageError
import logging
import warnings
from importlib.metadata import version
from html import escape
from sklearn.exceptions import InconsistentVersionWarning
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
div[data-testid="stSelectbox"] [data-baseweb="select"] > div { background:#fffdf8; color:#20332b; }
.stApp label, .stApp h3, .stApp h4 { color:#20332b; }
@media (max-width:700px) {
  div[data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
  div[data-testid="stColumn"] { width:100% !important; flex:1 1 100% !important; min-width:0; }
  .result-card { min-height:0; }
}
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
    "gill-spacing": {"c":"Phiến sít", "w":"Phiến thưa", "d":"Phiến rất sít"},
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
    expected = config.get("scikit_learn")
    if not expected or expected != version("scikit-learn"):
        raise ValueError(
            f"Phiên bản scikit-learn chưa khớp mô hình. "
            f"Cần scikit-learn=={expected or 'phiên bản lúc huấn luyện'}, "
            f"hiện tại là {version('scikit-learn')}. Cập nhật requirements.txt rồi triển khai lại."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        preprocessing = joblib.load(preprocessing_path)
    for key in ("cot_giu", "imputer", "onehot"):
        if key not in preprocessing:
            raise ValueError(f"tien_xu_ly.pkl thiếu thành phần {key}.")
    columns = preprocessing["cot_giu"]
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Danh sách cột trong bộ tiền xử lý không hợp lệ.")
    for column in columns:
        if not config.get("gia_tri_nhap", {}).get(column):
            raise ValueError(f"thong_tin.json thiếu lựa chọn cho cột {column}.")

    if config.get("tensorflow") != tf.__version__:
        raise ValueError(f"Cần tensorflow-cpu=={config.get('tensorflow')}; hiện tại là {tf.__version__}.")
    encoder = tf.keras.models.load_model(encoder_path, compile=False)
    models = {
        "A - KNN": joblib.load(MODEL_DIR / "A_KNN.pkl"),
        "A - GaussianNB": joblib.load(MODEL_DIR / "A_GaussianNB.pkl"),
        "B - KNN": joblib.load(MODEL_DIR / "B_KNN.pkl"),
        "B - GaussianNB": joblib.load(MODEL_DIR / "B_GaussianNB.pkl"),
    }
    if encoder.input_shape[-1] != config["so_chieu_onehot"] or encoder.output_shape[-1] != config["so_chieu_an"]:
        raise ValueError("Số chiều Encoder không khớp thong_tin.json.")
    for name, model in models.items():
        expected_dim = config["so_chieu_onehot"] if name.startswith("A") else config["so_chieu_an"]
        if model.n_features_in_ != expected_dim:
            raise ValueError(f"{name}: số chiều không khớp cấu hình.")
        if set(model.classes_) != {0, 1}:
            raise ValueError(f"{name}: nhãn mô hình phải là 0 và 1.")
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
    </section>
    """,
    unsafe_allow_html=True,
)



def decode_photo(raw):
    """Decode only supported still images; keep original resolution for inference."""
    if not raw or len(raw) > 10 * 1024 * 1024:
        raise ValueError("Ảnh phải có dung lượng tối đa 10 MB.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as photo:
            if photo.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Vui lòng chọn ảnh JPG, PNG hoặc WebP.")
            if photo.width * photo.height > 20_000_000:
                raise ValueError("Ảnh quá lớn. Vui lòng giảm xuống dưới 20 megapixel.")
            photo.load()
            return ImageOps.exif_transpose(photo).convert("RGB")


@st.cache_resource(show_spinner="Đang nạp mô hình ảnh…")
def load_image_model(model_stamp, config_stamp):
    # Contract for a separately TRAINED image model; never use encoder.keras here.
    # image_config.json:
    # {"class_names":["e","p"], "input_size":[224,224],
    #  "preprocessing_in_model":true, "resize":"bilinear_stretch",
    #  "input_range":"0_255", "output":"softmax"}
    # Exported model must include its training-time normalization layers.
    folder = ROOT / "mo_hinh_anh"
    cfg = json.loads((folder / "image_config.json").read_text(encoding="utf-8"))
    names = cfg.get("class_names", [])
    size = cfg.get("input_size", [])
    if len(names) != 2 or set(names) != {"e", "p"}:
        raise ValueError("Cấu hình mô hình ảnh phải xác định đúng thứ tự hai lớp e và p.")
    if len(size) != 2 or any(type(n) is not int or not 16 <= n <= 1024 for n in size):
        raise ValueError("Kích thước đầu vào mô hình ảnh không hợp lệ.")
    if (cfg.get("preprocessing_in_model") is not True or cfg.get("input_range") != "0_255"
        or cfg.get("resize") != "bilinear_stretch" or cfg.get("output") != "softmax"):
        raise ValueError("Quy trình xử lý ảnh chưa tương thích với ứng dụng. Cần đối chiếu cách huấn luyện.")
    model = tf.keras.models.load_model(folder / "image_model.keras", compile=False)
    if model.input_shape != (None, size[0], size[1], 3) or model.output_shape != (None, 2):
        raise ValueError("Kiến trúc mô hình ảnh không khớp cấu hình.")
    return model, cfg


def predict_photo(photo, model, cfg):
    pixels = tf.image.resize(np.asarray(photo, dtype=np.float32), cfg["input_size"],
                             method="bilinear", antialias=False)
    probabilities = np.asarray(model(pixels[None, ...], training=False))[0]
    if (probabilities.shape != (2,) or not np.isfinite(probabilities).all()
        or (probabilities < 0).any() or (probabilities > 1).any()
        or not np.isclose(probabilities.sum(), 1, atol=1e-4)):
        raise ValueError("Đầu ra mô hình ảnh không phải xác suất hợp lệ.")
    return cfg["class_names"][int(np.argmax(probabilities))], float(probabilities[cfg["class_names"].index("p")])


mode = st.radio("Cách sử dụng", ["Thuộc tính + ảnh minh họa", "Phân loại trực tiếp từ ảnh"],
                horizontal=True, key="prediction_mode")
with st.container(border=True):
    st.markdown("### 📷 Ảnh mẫu nấm")
    source = st.radio("Nguồn ảnh", ["Tải ảnh lên", "Chụp bằng camera"],
                      horizontal=True, key="photo_source")
    if source == "Tải ảnh lên":
        uploaded_photo = st.file_uploader("Chọn ảnh JPG, PNG hoặc WebP · tối đa 10 MB",
                                         type=["jpg", "jpeg", "png", "webp"], key="uploaded_photo")
    else:
        st.caption("Cho phép trình duyệt truy cập camera. Nếu không có camera, hãy chọn Tải ảnh lên.")
        uploaded_photo = st.camera_input("Chụp mẫu nấm", key="captured_photo")
    photo = None
    photo_id = None
    if uploaded_photo is not None:
        try:
            raw = uploaded_photo.getvalue()
            photo = decode_photo(raw)
            photo_id = hashlib.sha256(raw).hexdigest()
            preview = photo.copy()
            preview.thumbnail((1000, 700))
            st.image(preview, caption="Ảnh đang chọn", width="content")
        except (ValueError, OSError, UnidentifiedImageError,
                Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            st.error(f"Không đọc được ảnh: {error}")
    st.caption("Ảnh được gửi đến máy chủ để hiển thị/xử lý. Ứng dụng không ghi ảnh vào tệp hoặc GitHub.")

if st.session_state.get("photo_identity") != (photo_id, source, mode):
    st.session_state.pop("photo_prediction", None)
    st.session_state.pop("latest_predictions", None)
    st.session_state.pop("prediction_input", None)
    st.session_state["photo_identity"] = (photo_id, source, mode)

if mode == "Phân loại trực tiếp từ ảnh":
    folder = ROOT / "mo_hinh_anh"
    image_path = folder / "image_model.keras"
    config_path = folder / "image_config.json"
    ready = image_path.is_file() and config_path.is_file()
    if not ready:
        st.info("Chưa có mô hình phân loại ảnh. Cần huấn luyện mô hình bằng ảnh có nhãn trước khi sử dụng chế độ này.")
        st.caption("Các mô hình KNN, GaussianNB và Encoder hiện tại chỉ xử lý thuộc tính dạng bảng.")
    if st.button("🔎 Phân tích ảnh", type="primary", width="stretch", disabled=not ready or photo is None):
        st.session_state.pop("photo_prediction", None)
        try:
            image_model, image_cfg = load_image_model(image_path.stat().st_mtime_ns, config_path.stat().st_mtime_ns)
            with st.spinner("Đang phân tích ảnh…"):
                st.session_state["photo_prediction"] = predict_photo(photo, image_model, image_cfg)
        except Exception as error:
            logging.exception("Image inference failed")
            st.error("Không thể phân tích ảnh. Cần kiểm tra mô hình ảnh và cấu hình tiền xử lý.")
            with st.expander("Chi tiết lỗi"):
                st.code(str(error), language=None)
    if "photo_prediction" in st.session_state:
        label, probability = st.session_state["photo_prediction"]
        st.markdown("### Kết quả từ mô hình ảnh")
        st.write("Lớp dự đoán: **" + {"e": "Ăn được", "p": "Độc"}[label] + "**")
        st.write(f"Xác suất lớp Độc theo mô hình: **{probability:.2%}**")
    st.warning("Demo học tập. Không dùng ảnh hoặc kết quả này để quyết định ăn nấm ngoài thực tế.")
    st.stop()

st.caption("Ảnh chỉ hỗ trợ quan sát. Kết quả bên dưới được tính từ các thuộc tính bạn nhập; ảnh không tự điền thuộc tính.")

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


def options_for(column):
    values = list(dict.fromkeys(str(v) for v in config["gia_tri_nhap"][column]))
    if column == "stalk-root" and "?" not in values:
        values.append("?")
    return values


def clear_prediction():
    st.session_state.pop("latest_predictions", None)
    st.session_state.pop("prediction_input", None)


def predict_values(values, preprocessing, encoder, models):
    """Chỉ transform/predict bằng tài nguyên đã lưu; không fit lại."""
    columns = preprocessing["cot_giu"]
    row = pd.DataFrame([{c: np.nan if values[c] == "?" else values[c]
                         for c in columns}], columns=columns)
    features_a = preprocessing["onehot"].transform(
        preprocessing["imputer"].transform(row))
    if hasattr(features_a, "toarray"):
        features_a = features_a.toarray()
    features_b = encoder.predict(features_a, verbose=0)
    if not np.isfinite(features_a).all() or not np.isfinite(features_b).all():
        raise ValueError("Đặc trưng sau xử lý chứa giá trị không hợp lệ.")
    features = {"A": features_a, "B": features_b}
    rows = []
    for name, model in models.items():
        data = features[name.split(" - ")[0]]
        pred = int(model.predict(data)[0])
        pos = list(model.classes_).index(1)
        probability = float(model.predict_proba(data)[0, pos])
        if pred not in (0, 1) or not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"Kết quả của {name} không hợp lệ.")
        rows.append({"Mô hình": name, "Dự đoán": "Độc" if pred == 1 else "Ăn được",
                     "Xác suất lớp Độc": probability})
    return rows


feature_columns = preprocessing["cot_giu"]
st.success("Đã sẵn sàng · Nạp đủ bộ tiền xử lý, Encoder và 4 mô hình")
flow_filter = st.radio(
    "Chọn luồng kết quả", ["all", "A", "B"], horizontal=True, key="flow_filter",
    format_func=lambda x: {"all": "So sánh cả hai luồng", "A": "Luồng A · One-Hot",
                           "B": "Luồng B · Autoencoder"}[x])
st.caption("Luồng A dùng đặc trưng One-Hot. Luồng B dùng đặc trưng nén qua Encoder.")

st.markdown('<div class="section-title">Đặc điểm mẫu nấm</div>', unsafe_allow_html=True)
st.caption("1. Nhập đặc điểm hoặc nạp mẫu → 2. Bấm Dự đoán → 3. Đối chiếu kết quả")

with st.container(border=True):
    if st.button("🎲 Nạp mẫu minh họa", width="stretch"):
        try:
            demo_data = load_demo_data()
            if demo_data is None or demo_data.empty:
                raise ValueError("Cần tệp mushrooms.csv cùng cấp app.py để nạp mẫu.")
            missing = set(feature_columns + ["class"]) - set(demo_data.columns)
            if missing:
                raise ValueError("Dữ liệu minh họa thiếu cột: " + ", ".join(sorted(missing)))
            sample = demo_data.sample(n=1).iloc[0]
            values = {c: "?" if pd.isna(sample[c]) else str(sample[c]) for c in feature_columns}
            for c, value in values.items():
                if value not in options_for(c):
                    raise ValueError(f"Mẫu chứa giá trị chưa hỗ trợ: {c}={value}. Không tự thay thế mẫu.")
            for c, value in values.items():
                st.session_state[f"mushroom_{c}"] = value
            st.session_state["demo_sample"] = {"values": values, "label": str(sample["class"])}
            clear_prediction()
        except Exception as error:
            st.warning(str(error))
    st.caption("Mẫu lấy từ mushrooms.csv, có thể thuộc dữ liệu huấn luyện; không liên quan đến ảnh bạn tải lên.")
    groups = [
        ("🍄 Mũ và mùi", ["cap-shape", "cap-surface", "cap-color", "bruises", "odor"]),
        ("🌿 Phiến nấm", ["gill-attachment", "gill-spacing", "gill-size", "gill-color"]),
        ("🪵 Cuống nấm", ["stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring"]),
        ("⭕ Màng và vòng", ["veil-color", "ring-number", "ring-type"]),
        ("🌲 Bào tử và môi trường", ["spore-print-color", "population", "habitat"]),
    ]
    grouped = {c for _, cols in groups for c in cols}
    extra = [c for c in feature_columns if c not in grouped]
    if extra:
        groups.append(("Đặc điểm bổ sung", extra))
    current_values = {}
    for title, columns in groups:
        present = [c for c in columns if c in feature_columns]
        if not present:
            continue
        st.markdown(f"#### {title}")
        grid = st.columns(3)
        for i, c in enumerate(present):
            options = options_for(c)
            key = f"mushroom_{c}"
            if st.session_state.get(key) not in options:
                st.session_state[key] = options[0]
            with grid[i % 3]:
                current_values[c] = st.selectbox(
                    FEATURE_LABELS.get(c, c), options,
                    format_func=lambda v, c=c: display_value(c, v),
                    key=key, on_change=clear_prediction)

    sample_info = st.session_state.get("demo_sample")
    if sample_info:
        if sample_info["values"] == current_values:
            label = {"p": "Độc", "e": "Ăn được"}.get(sample_info["label"], "Không xác định")
            st.info(f"Nhãn gốc của mẫu: **{label}** — chỉ để đối chiếu, không đưa vào mô hình.")
        else:
            st.caption("Đã sửa đầu vào. Nhãn của mẫu minh họa không còn áp dụng.")
    if st.button("✨ Dự đoán với các mô hình", type="primary", width="stretch"):
        clear_prediction()
        try:
            with st.spinner("Đang phân tích đặc điểm hiện tại…"):
                rows = predict_values(current_values, preprocessing, encoder, models)
            st.session_state["latest_predictions"] = rows
            st.session_state["prediction_input"] = current_values.copy()
        except Exception as error:
            logging.exception("Prediction failed")
            st.error("Không thể dự đoán. Kiểm tra tính tương thích của thư viện và các tệp mô hình.")
            with st.expander("Chi tiết lỗi để kiểm tra"):
                st.code(str(error), language=None)

if st.session_state.get("prediction_input") != current_values:
    clear_prediction()
all_predictions = st.session_state.get("latest_predictions", [])
predictions = [p for p in all_predictions
               if flow_filter == "all" or p["Mô hình"].startswith(flow_filter + " - ")]
if predictions:
    st.markdown('<div class="section-title">Kết quả từ các thuộc tính đã nhập</div>', unsafe_allow_html=True)
    labels = {p["Dự đoán"] for p in predictions}
    if len(labels) == 1:
        summary = f"{len(predictions)} mô hình đang hiển thị cùng dự đoán: {predictions[0]['Dự đoán']}"
    else:
        summary = "Các mô hình chưa thống nhất — xem kết quả từng mô hình bên dưới."
    st.info(summary)
    cols = st.columns(2)
    for i, result in enumerate(predictions):
        color = "#a63835" if result["Dự đoán"] == "Độc" else "#276342"
        with cols[i % 2]:
            st.markdown(f"""<div class="result-card" style="margin-bottom:14px; border-top:4px solid {color}">
                <div class="result-label">{escape(result['Mô hình'])}</div>
                <div class="result-class" style="color:{color}">{result['Dự đoán']}</div>
                <div class="result-meta">Xác suất lớp Độc theo mô hình<br>
                <b>{result['Xác suất lớp Độc']:.2%}</b></div></div>""", unsafe_allow_html=True)
    comparison = pd.DataFrame(predictions)
    st.dataframe(comparison.style.format({"Xác suất lớp Độc": "{:.2%}"}),
                 hide_index=True, width="stretch")
    st.caption("Xác suất là đầu ra của mô hình, không phải mức bảo đảm an toàn.")
    with st.expander("Xem đặc điểm đã dùng để dự đoán"):
        st.dataframe(pd.DataFrame([
            {"Thuộc tính": FEATURE_LABELS.get(c, c), "Giá trị": display_value(c, current_values[c])}
            for c in feature_columns]), hide_index=True, width="stretch")
else:
    st.info("Chọn đặc điểm rồi bấm Dự đoán. Khi sửa đầu vào, kết quả cũ sẽ được xóa.")

st.markdown('<div class="safe-note"><b>Demo học tập:</b> Không dùng kết quả để quyết định ăn nấm ngoài thực tế.</div>', unsafe_allow_html=True)
st.markdown('<div class="footer-note">Đề tài 09 · Phân loại nấm · KNN & Gaussian Naive Bayes</div>', unsafe_allow_html=True)
