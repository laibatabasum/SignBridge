import cv2

print("Opening webcam...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam opened! Press 'q' OR click the X button to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    cv2.imshow("SignBridge - Webcam Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if cv2.getWindowProperty("SignBridge - Webcam Test", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")