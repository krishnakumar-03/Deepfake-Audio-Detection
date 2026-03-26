import streamlit as st
from keras.models import load_model
from utils.preprocess import preprocess_audio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------
# Load Models
# ------------------------
cnn_model = load_model("models/cnn1.keras")
cnn2_model = load_model("models/cnn2.keras")
lstm_model = load_model("models/lstm.keras")
mobilenet_model = load_model("models/mobilenet.keras")

st.set_page_config(page_title="Deepfake Audio Detector", layout="centered")

st.title("🎵 Deepfake Audio Detection")
st.write("Upload an audio file to detect if it's **Real or Fake**")

# ------------------------
# Upload Audio
# ------------------------
uploaded_file = st.file_uploader("Upload audio", type=['wav','mp3','flac'])

if uploaded_file:
    temp_path = "temp_audio.wav"

    # Save uploaded file
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ------------------------
    # Audio Playback
    # ------------------------
    st.subheader("🔊 Audio Playback")
    st.audio(temp_path)

    # ------------------------
    # Preprocess Audio
    # ------------------------
    X, mel_plot = preprocess_audio(temp_path)

    # ------------------------
    # Display Mel Spectrogram
    # ------------------------
    st.subheader("📊 Mel-Spectrogram")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.imshow(mel_plot, aspect='auto', origin='lower')
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel Frequency")
    st.pyplot(fig)

    # ------------------------
    # Predictions
    # ------------------------
    pred_cnn = cnn_model.predict(X)[0][0]
    pred_cnn2 = cnn2_model.predict(X)[0][0]
    pred_lstm = lstm_model.predict(X)[0][0]
    pred_mobilenet = mobilenet_model.predict(X)[0][0]

    # ------------------------
    # Format Function
    # ------------------------
    def format_pred(pred):
        real = 1 - pred
        fake = pred
        label = "Fake" if pred > 0.5 else "Real"
        return label, real*100, fake*100

    # ------------------------
    # Display Individual Predictions
    # ------------------------
    st.subheader("🤖 Model Predictions")

    label_cnn, real_cnn, fake_cnn = format_pred(pred_cnn)
    label_cnn2, real_cnn2, fake_cnn2 = format_pred(pred_cnn2)
    label_lstm, real_lstm, fake_lstm = format_pred(pred_lstm)
    label_mob, real_mob, fake_mob = format_pred(pred_mobilenet)

    st.write(f"CNN1:        {label_cnn} (Real: {real_cnn:.1f}%, Fake: {fake_cnn:.1f}%)")
    st.write(f"CNN2:        {label_cnn2} (Real: {real_cnn2:.1f}%, Fake: {fake_cnn2:.1f}%)")
    st.write(f"LSTM:        {label_lstm} (Real: {real_lstm:.1f}%, Fake: {fake_lstm:.1f}%)")
    st.write(f"MobileNet:   {label_mob} (Real: {real_mob:.1f}%, Fake: {fake_mob:.1f}%)")

    # ------------------------
    # Ensemble Prediction (Improved Weights)
    # ------------------------
    ensemble_pred = (
        0.5 * pred_cnn2 +   # strongest
        0.3 * pred_mobilenet +
        0.2 * pred_lstm
    )

    ensemble_label = "Fake" if ensemble_pred > 0.5 else "Real"
    real_conf = (1 - ensemble_pred) * 100
    fake_conf = ensemble_pred * 100

    st.subheader("✅ Ensemble Prediction")

    st.write(
        f"Prediction: **{ensemble_label}** "
        f"(Real: {real_conf:.1f}%, Fake: {fake_conf:.1f}%)"
    )

    # Confidence bar
    st.progress(int(fake_conf) if ensemble_label == "Fake" else int(real_conf))

    # ------------------------
    # Download Report
    # ------------------------
    st.subheader("📄 Download Report")

    report_data = {
        "Model": ["CNN1", "CNN2", "LSTM", "MobileNet", "Ensemble"],
        "Prediction": [
            label_cnn,
            label_cnn2,
            label_lstm,
            label_mob,
            ensemble_label
        ],
        "Real Confidence (%)": [
            f"{real_cnn:.1f}",
            f"{real_cnn2:.1f}",
            f"{real_lstm:.1f}",
            f"{real_mob:.1f}",
            f"{real_conf:.1f}"
        ],
        "Fake Confidence (%)": [
            f"{fake_cnn:.1f}",
            f"{fake_cnn2:.1f}",
            f"{fake_lstm:.1f}",
            f"{fake_mob:.1f}",
            f"{fake_conf:.1f}"
        ]
    }

    df = pd.DataFrame(report_data)
    csv = df.to_csv(index=False)

    st.download_button(
        label="⬇️ Download Prediction Report",
        data=csv,
        file_name="prediction_report.csv",
        mime="text/csv"
    )