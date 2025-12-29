import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import tensorflow as tf

# Load Models
eye_model = tf.keras.models.load_model("eye_cnn.h5")
mouth_model = tf.keras.models.load_model("mouth_cnn.h5")

IMG_SIZE = 32

# WebRTC config for browser camera
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.title("🚗 Driver Drowsiness & Yawn Detection (Web Version)")
st.write("Camera will run in browser, not on server. Works on Streamlit Cloud.")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (IMG_SIZE, IMG_SIZE))
        resized = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

        eye_prob = eye_model.predict(resized, verbose=0)[0][1]
        mouth_prob = mouth_model.predict(resized, verbose=0)[0][1]

        eye_state = "😴 Drowsy" if eye_prob > 0.7 else "🙂 Awake"
        yawn_state = "🟡 Yawning" if mouth_prob > 0.65 else "🟢 Normal"

        cv2.putText(img, f"Eye: {eye_state}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(img, f"Mouth: {yawn_state}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="drowsy",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor
)
