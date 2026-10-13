import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# --- Data storage ---
DATA_FILE = "data/landmarks.csv"
os.makedirs("data", exist_ok=True)

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)
    print(f"Created {DATA_FILE}")

# --- Helper: normalize landmarks ---
def normalize_landmarks(hand_landmarks):
    """Convert 21 landmarks into 63 normalized numbers (relative to wrist)."""
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    landmarks = landmarks - landmarks[0]  # subtract wrist
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks = landmarks / max_dist
    return landmarks.flatten()

# --- Main loop ---
print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("\n=== DATA COLLECTION MODE ===")
print("Press a letter/number key to set label (A, B, C, ...)")
print("Press SPACE to save sample")
print("Press Q to quit\n")

current_label = None
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

    label_text = f"Label: {current_label}" if current_label else "Label: (press a key)"
    count_text = f"Saved: {saved_count}"

    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, count_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, "SPACE=save  Q=quit", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("SignBridge - Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        if current_label is None:
            print("Set a label first")
        elif not results.multi_hand_landmarks:
            print("No hand detected — nothing to save")
        else:
            landmarks = normalize_landmarks(results.multi_hand_landmarks[0])
            with open(DATA_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([current_label] + landmarks.tolist())
            saved_count += 1
            print(f"Saved sample {saved_count} for label '{current_label}'")
    elif 32 < key < 127:
        current_label = chr(key).upper()
        print(f"Label set to: {current_label}")

cap.release()
cv2.destroyAllWindows()
print(f"\nTotal saved: {saved_count}")
print(f"Data file: {DATA_FILE}")