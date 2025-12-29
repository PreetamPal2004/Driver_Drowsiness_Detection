import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import tensorflow as tf

# ⚠️ Fix: Prevent GPU/TF errors in cloud
tf.get_logger().setLevel('ERROR')

# 🚫 Disable TF GPU search (Streamlit cloud has no GPU)
tf.config.set_visible_devices([], 'GPU')

# Load Models (CPU Compatible)
@st.cache_resource
def load_models():
    eye_model = tf.keras.models.load_model("eye_cnn.h5", compile=False)
    mouth_model = tf.keras.models.load_model("mouth_cnn.h5", compile=False)
    return eye_model, mouth_model

eye_model, mouth_model = load_models()

IMG_SIZE = 32

# WebRTC STUN only (TURN optional, remove if errors)
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
}

st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")

st.title("🚗 Driver Drowsiness & Yawn Detection - Web Version")
st.write("📸 **Camera runs in your browser, not Streamlit server.**")
st.info("If camera doesn't start, allow permissions and reload page.")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Preprocess frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).reshape(1,IMG_SIZE,IMG_SIZE,1)/255.0

        # Predictions
        eye_prob = float(eye_model.predict(resized, verbose=0)[0][1])
        mouth_prob = float(mouth_model.predict(resized, verbose=0)[0][1])

        # States
        eye_state = "😴 Drowsy" if eye_prob > 0.7 else "🙂 Awake"
        yawn_state = "🟡 Yawning" if mouth_prob > 0.65 else "🟢 Normal"

        # Display text
        cv2.putText(img, f"Eye: {eye_state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)
        cv2.putText(img, f"Mouth: {yawn_state}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTC_CONFIGURATION,
)
