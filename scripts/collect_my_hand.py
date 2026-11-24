import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0, min_detection_confidence=0.5)

DATA_FILE = "data/custom_landmarks.csv"
os.makedirs("data", exist_ok=True)

# Create the file with a header if it doesn't exist
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
        writer.writerow(header)

cap = cv2.VideoCapture(0)
print("\n=== CUSTOM DATA COLLECTION ===")
print("1. Press a letter key (e.g., 'L', 'A', 'B') to set the label.")
print("2. Hold the sign in the camera.")
print("3. Press SPACE to capture 30 frames.")
print("4. Press 'Q' to quit.\n")

current_label = None

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    label_text = f"Label: {current_label}" if current_label else "Label: (press a key)"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("SignBridge - Collect Data", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord(' '):
        if current_label and results.multi_hand_landmarks:
            print(f"Capturing 30 frames for '{current_label}'...")
            for _ in range(30):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    row = []
                    for lm in res.multi_hand_landmarks[0].landmark:
                        row.extend([lm.x, lm.y, lm.z])
                    with open(DATA_FILE, "a", newline="") as f:
                        csv.writer(f).writerow([current_label] + row)
            print(f"✅ Saved!")
        else:
            print("❌ Make sure you set a label and show your hand!")
    elif 32 < key < 127:
        current_label = chr(key).upper()

cap.release()
cv2.destroyAllWindows()