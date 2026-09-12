import cv2
import mediapipe as mp

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam opened! Show your hand. Press 'q' or click X to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Flip the frame horizontally so it acts like a mirror
    frame = cv2.flip(frame, 1)

    # MediaPipe needs RGB, OpenCV gives BGR
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run hand detection
    results = hands.process(rgb)

    # If a hand was found, draw the landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

    cv2.imshow("SignBridge - Hand Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if cv2.getWindowProperty("SignBridge - Hand Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("Closed.")