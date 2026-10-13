import pandas as pd
import streamlit as st

from app.model_loader import load_artifacts, get_model, get_pipeline, get_label_encoder

st.set_page_config(page_title="Stellar Classifier", page_icon="🌌", layout="centered")


@st.cache_resource
def load_all():
    load_artifacts()
    return get_model(), get_pipeline(), get_label_encoder()


model, pipeline, label_encoder = load_all()

CLASS_INFO = {
    "GALAXY": "🌀 Galaxy",
    "STAR": "⭐ Star",
    "QSO": "💫 Quasar (QSO)",
}

st.title("🌌 Stellar Classification (SDSS17)")
st.write(
    "Predict whether an astronomical object is a **Star**, a **Galaxy** or a **Quasar** "
    "from its magnitudes in five filters, its sky position and its redshift."
)

st.subheader("Object measurements")

col1, col2 = st.columns(2)
with col1:
    u = st.number_input("u (ultraviolet)", value=23.87, format="%.4f")
    g = st.number_input("g (green)", value=22.27, format="%.4f")
    r = st.number_input("r (red)", value=20.39, format="%.4f")
    i = st.number_input("i (near infrared)", value=19.01, format="%.4f")
    z = st.number_input("z (infrared)", value=18.52, format="%.4f")
with col2:
    alpha = st.number_input("alpha (right ascension)", value=135.68, min_value=0.0, max_value=360.0, format="%.4f")
    delta = st.number_input("delta (declination)", value=32.49, min_value=-90.0, max_value=90.0, format="%.4f")
    redshift = st.number_input("redshift", value=0.634, format="%.6f")

if st.button("Classify", type="primary"):
    row = pd.DataFrame([{
        "alpha": alpha, "delta": delta,
        "u": u, "g": g, "r": r, "i": i, "z": z,
        "redshift": redshift,
    }])

    X = pipeline.transform(row)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    predicted_class = label_encoder.inverse_transform([pred])[0]
    probs = pd.Series(proba, index=label_encoder.classes_, name="Probability")

    st.success(f"Prediction: **{CLASS_INFO.get(predicted_class, predicted_class)}**")
    st.caption(f"Confidence: {probs.max():.1%}")

    st.subheader("Class probabilities")
    st.bar_chart(probs)

    st.subheader("Color indices")
    st.dataframe(
        pd.DataFrame({
            "u-g": [u - g], "g-r": [g - r], "r-i": [r - i], "i-z": [i - z],
        }).round(3),
        hide_index=True,
    )

st.divider()
st.caption(
    "The model relies on redshift, a spectroscopic measurement, so it is not meant for "
    "objects that only have photometry. Default values are an arbitrary example object."
)
