from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "plant_disease_final_model.keras"

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)
# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: #f4f8f5;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

[data-testid="stSidebar"] {
    background: #e7f2e9;
}

.hero {
    background: linear-gradient(120deg, #123d2a, #26734d);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 8px;
}

.hero p {
    color: #e1f2e6;
    font-size: 16px;
}

.section-title {
    color: #174b32;
    font-size: 23px;
    font-weight: 700;
    margin: 15px 0;
}

.result-card {
    background: white;
    border: 1px solid #dce9df;
    border-radius: 16px;
    padding: 24px;
    margin-top: 12px;
}

.result-name {
    color: #17643b;
    font-size: 25px;
    font-weight: 700;
}

.small-text {
    color: #607467;
    font-size: 14px;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border: 1px solid #dce9df;
    border-radius: 12px;
}

.stButton button {
    background: #17643b;
    color: white;
    border-radius: 10px;
    border: none;
}

.footer {
    text-align: center;
    color: #718477;
    padding: 25px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/plant_disease_final_model.keras",
        compile=False
    )

@st.cache_data
def load_classes():
    with open("models/class_names.json", "r") as f:
        return json.load(f)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## 🌿 PlantCare AI")
    st.caption("Intelligent Plant Disease Detection")

    st.divider()

    st.markdown("### Navigation")
    st.markdown("🏠  Home")
    st.markdown("🔬  Disease Detection")
    st.markdown("📊  Prediction Analysis")

    st.divider()

    st.markdown("### About Project")
    st.write(
        "An AI-powered plant disease classification "
        "system using EfficientNetB0 and deep learning."
    )

    st.divider()
    st.caption("Model: EfficientNetB0")
    st.caption("Dataset: PlantVillage")
    st.caption("Classes: 38")

    # ---------------- PREDICTION HISTORY ----------------

    if "history" not in st.session_state:
        st.session_state.history = []

# ---------------- MAIN HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🌱 Plant Disease Detection</h1>
    <p>
        Identify plant diseases using Artificial Intelligence.
        Upload a leaf image and let our deep learning model
        analyze it.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD RESOURCES ----------------

try:
    with st.spinner("Loading AI model..."):
        model = load_model()
        class_names = load_classes()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# ---------------- UPLOAD SECTION ----------------

st.markdown(
    '<div class="section-title">🔬 Analyze Your Plant</div>',
    unsafe_allow_html=True
)

st.write("Upload a clear photograph of a plant leaf.")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG or PNG image."
)

if uploaded_file is None:

    st.info("👆 Upload a plant leaf image to start disease detection.")

    st.markdown("""
    <div class="result-card">
        <h3 style="color:#17643b;">How it works</h3>
        <p>1. Upload a plant leaf photograph.</p>
        <p>2. AI analyzes the image.</p>
        <p>3. View the predicted class and confidence.</p>
    </div>
    """, unsafe_allow_html=True)

else:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown("### 📷 Uploaded Image")
            st.image(image, use_container_width=True)
            st.caption("Your uploaded plant leaf photograph")

        # Prepare image
        resized_image = image.resize((224, 224))

        image_array = np.array(
            resized_image,
            dtype=np.float32
        )

        image_array = np.expand_dims(image_array, axis=0)

        # Prediction
        with st.spinner("Analyzing plant leaf..."):
            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

        if len(predictions) != len(class_names):
            st.error("The model output and class names do not match.")
            st.stop()

        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])

        st.session_state.history.append({
            "Disease": class_names[predicted_index],
            "Confidence": f"{confidence * 100:.2f}%",
            "Image": uploaded_file.name
            })
        top_indices = np.argsort(predictions)[::-1][:3]

        with right:
            st.markdown("### 🧠 AI Prediction")

            st.markdown(f"""
            <div class="result-card">
                <p class="small-text">PREDICTED CLASS</p>
                <div class="result-name">
                    {class_names[predicted_index]}
                </div>
                <p class="small-text">
                    Highest scoring model prediction
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.metric(
                "Prediction Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.progress(confidence)

            st.caption(
                "Confidence is the model's output score, "
                "not a guarantee of diagnostic accuracy."
            )

        # ---------------- TOP 3 RESULTS ----------------

        st.divider()

        st.markdown(
            '<div class="section-title">📊 Prediction Analysis</div>',
            unsafe_allow_html=True
        )

        for rank, index in enumerate(top_indices, start=1):

            score = float(predictions[index])

            st.write(
                f"**{rank}. {class_names[index]}** — "
                f"{score * 100:.2f}%"
            )

            st.progress(score)

        st.divider()

        st.markdown("""
        <div class="result-card">
            <h4 style="color:#17643b;">🌿 Important Note</h4>
            <p>
                This application provides experimental AI-based
                predictions. It is not a substitute for expert
                agricultural diagnosis.
            </p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Image processing error: {e}")

        # ---------------- HISTORY PAGE ----------------

st.divider()
st.markdown("## 📋 Prediction History")

if st.session_state.history:
    st.dataframe(
        st.session_state.history,
        use_container_width=True
    )

    if st.button("Clear Prediction History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No prediction history available yet.")

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    PlantCare AI | MCA Project | Deep Learning with EfficientNetB0
</div>
""", unsafe_allow_html=True)
