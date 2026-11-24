import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ─── PAGE SETUP ─────────────────────────────────────────────
st.set_page_config(page_title="SignBridge", page_icon="🤟", layout="wide")
st.title("🤟 SignBridge")
st.caption("Real-time offline sign language translation — no cloud, just a webcam.")

# ─── LOAD AI MODEL ──────────────────────────────────────────
# We load the model and the letter classes we saved yesterday
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\aaa\SignBridgeProject\models\static_model.h5")
    classes = np.load(r"C:\Users\aaa\SignBridgeProject\models\classes.npy", allow_pickle=True)
    return model, classes

model, classes = load_model()

# ─── MEDIAPIPE SETUP ────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ─── VIDEO PROCESSOR ────────────────────────────────────────
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_letter = ""
        self.confidence = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Flip the image so it acts like a mirror
        img = cv2.flip(img, 1)
        
        # 2. Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # 3. If a hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the skeleton on the screen
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 4. Extract the 63 numbers (21 landmarks x 3 coords)
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                
                # 5. Feed to AI
                if len(row) == 63:
                    # Make prediction
                    prediction = model.predict(np.array([row]), verbose=0)
                    class_id = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    # If the AI is confident enough (e.g., > 70%)
                    if confidence > 0.70:
                        self.current_letter = classes[class_id]
                        self.confidence = confidence
                    else:
                        self.current_letter = "?"
                        self.confidence = confidence

        # Display text on the video
        if self.current_letter:
            cv2.putText(img, f"{self.current_letter} ({self.confidence*100:.0f}%)", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─── UI LAYOUT ──────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Feed")
    webrtc_streamer(
        key="signbridge",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("📝 Translation")
    st.info("Make a sign in front of the camera. The AI will guess the letter here.")
    # Note: In the next step, we will connect the video processor to this text box.