import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────
RAW_DIR = r"C:\Users\aaa\SignBridgeProject\data\raw"
OUTPUT_CSV = r"C:\Users\aaa\SignBridgeProject\data\landmarks.csv"
# ───────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
# We use static_image_mode=True because we are processing individual images
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

data = []
total_processed = 0

print("Starting landmark extraction. This will take a few minutes...")

for letter in sorted(os.listdir(RAW_DIR)):
    letter_folder = os.path.join(RAW_DIR, letter)
    if not os.path.isdir(letter_folder):
        continue

    images = [f for f in os.listdir(letter_folder) if f.lower().endswith(('.jpg', '.png'))]
    
    for img_name in images:
        img_path = os.path.join(letter_folder, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # If a hand is detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract x, y, z coordinates for all 21 landmarks
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
                
            row.append(letter) # Add the label (e.g., 'A') at the end
            data.append(row)
            total_processed += 1

    print(f"✅ {letter}: Processed {len(images)} images. Total valid hands found so far: {total_processed}")

# Create column names (x0, y0, z0, x1, y1, z1... x20, y20, z20, label)
columns = []
for i in range(21):
    columns.extend([f"x{i}", f"y{i}", f"z{i}"])
columns.append("label")

# Save to CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n🎉 Done! Saved {len(df)} valid hand landmarks to:")
print(f"📁 {OUTPUT_CSV}")