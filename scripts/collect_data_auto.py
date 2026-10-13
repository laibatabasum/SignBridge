import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

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

# --- Normalization ---
def normalize_landmarks(hand_landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    landmarks = landmarks - landmarks[0]
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks = landmarks / max_dist
    return landmarks.flatten()

# --- Auto-capture settings ---
CAPTURE_DURATION = 3.0   # seconds
CAPTURE_FPS = 10         # samples per second
TOTAL_SAMPLES = int(CAPTURE_DURATION * CAPTURE_FPS)  # 30 samples

# --- Main loop ---
print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("\n=== AUTO DATA COLLECTION MODE ===")
print("1. Press a letter key (A, B, C...) to set the label")
print("2. Show the sign")
print("3. Press SPACE — it captures 30 samples over 3 seconds")
print("4. Hold the sign steady during the countdown")
print("5. Press Q to quit\n")

current_label = None
total_saved = 0
capturing = False
capture_start_time = 0
samples_this_capture = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_detected = bool(results.multi_hand_landmarks)

    if hand_detected:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

    # --- Capture mode ---
    if capturing:
        elapsed = time.time() - capture_start_time
        remaining = max(0, CAPTURE_DURATION - elapsed)

        # Capture a sample if enough time passed
        target_samples_so_far = int(elapsed * CAPTURE_FPS)
        if (samples_this_capture < target_samples_so_far
                and samples_this_capture < TOTAL_SAMPLES):
            if hand_detected:
                landmarks = normalize_landmarks(results.multi_hand_landmarks[0])
                with open(DATA_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([current_label] + landmarks.tolist())
                samples_this_capture += 1
                total_saved += 1

        # Draw countdown
        cv2.putText(frame, f"CAPTURING: {remaining:.1f}s",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(frame, f"Samples: {samples_this_capture}/{TOTAL_SAMPLES}",
                    (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Done?
        if samples_this_capture >= TOTAL_SAMPLES:
            capturing = False
            print(f"✅ Captured {samples_this_capture} samples for '{current_label}'")
            print(f"   Total saved this session: {total_saved}")
            samples_this_capture = 0

    # --- HUD ---
    label_text = f"Label: {current_label}" if current_label else "Label: (press a key)"
    count_text = f"Total saved: {total_saved}"

    cv2.putText(frame, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, count_text, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, "SPACE=capture  Q=quit", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("SignBridge - Auto Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        if current_label is None:
            print("Set a label first")
        elif not hand_detected:
            print("Show your hand first")
        elif not capturing:
            capturing = True
            capture_start_time = time.time()
            samples_this_capture = 0
            print(f"Capturing 30 samples for '{current_label}'...")
    elif 32 < key < 127:
        current_label = chr(key).upper()
        print(f"Label set to: {current_label}")

cap.release()
cv2.destroyAllWindows()
print(f"\nTotal samples saved this session: {total_saved}")