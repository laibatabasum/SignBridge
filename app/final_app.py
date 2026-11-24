import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import tensorflow as tf
from collections import Counter

st.set_page_config(page_title="SignBridge", page_icon="🤟")
st.title("🤟 SignBridge")

# Load the model using TensorFlow
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\aaa\SignBridgeProject\models\static_model.h5")
    classes = np.load(r"C:\Users\aaa\SignBridgeProject\models\classes.npy", allow_pickle=True)
    return model, classes

model, classes = load_model()

# MediaPipe Setup
base_options = python.BaseOptions(model_asset_path=r"C:\Users\aaa\SignBridgeProject\models\hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_letter = ""
        self.history = []  # Keeps track of the last 10 predictions

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                h, w, _ = img.shape
                row = []
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
                    row.extend([lm.x, lm.y, lm.z])
                
                if len(row) == 63:
                    prediction = model.predict(np.array([row]), verbose=0)
                    class_id = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    # SMOOTHING LOGIC
                    if confidence > 0.50: # Only add if the AI is reasonably sure
                        predicted_letter = classes[class_id]
                        self.history.append(predicted_letter)
                        
                        # Keep only the last 10 guesses
                        if len(self.history) > 10:
                            self.history.pop(0)
                        
                        # Find the most common letter in the last 10 guesses
                        most_common = Counter(self.history).most_common(1)[0]
                        letter = most_common[0]
                        count = most_common[1]
                        
                        # Only display if it won the majority vote (6 out of 10 times)
                        if count >= 6:
                            self.current_letter = f"{letter} ({confidence*100:.0f}%)"
                        else:
                            self.current_letter = "..." 
        else:
            # If no hand is detected, clear the history so it resets instantly
            self.history.clear()
            self.current_letter = ""

        if self.current_letter:
            cv2.putText(img, self.current_letter, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

col1, col2 = st.columns([2, 1])
with col1:
    webrtc_streamer(key="signbridge", video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
with col2:
    st.success("✅ TensorFlow AI Loaded!")
    st.info("The AI now uses a smoothing filter. It needs to see a sign for a split second before locking it in.")