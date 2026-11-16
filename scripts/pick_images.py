import os
import zipfile
import random

# ─── CONFIG ────────────────────────────────────────────────
# The path to the ZIP file in Google Drive
ZIP_PATH = r"G:\My Drive\SignBridge-Data\asl_dataset\asl-alphabet.zip"
DEST_DIR = r"C:\Users\aaa\SignBridgeProject\data\raw"
IMAGES_PER_LETTER = 300

# Letters we want (skip J and Z)
LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1) if chr(c) not in ('J', 'Z')]

random.seed(42)
# ───────────────────────────────────────────────────────────

os.makedirs(DEST_DIR, exist_ok=True)
total_copied = 0

print("Opening ZIP file... (this might take a few seconds)")

# Open the zip file directly!
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    # Get a list of all file paths inside the zip
    all_files = z.namelist()

    for letter in LETTERS:
        # Find all images for this letter inside the zip
        # The path inside the zip looks like: asl_alphabet_train/asl_alphabet_train/A/A1.jpg
        letter_files = [f for f in all_files if f.startswith(f"asl_alphabet_train/asl_alphabet_train/{letter}/") and f.lower().endswith(('.jpg', '.png'))]

        if len(letter_files) == 0:
            print(f"⚠️  No files found for {letter} inside the zip.")
            continue

        # Pick 300 random images
        if len(letter_files) < IMAGES_PER_LETTER:
            chosen = letter_files
        else:
            chosen = random.sample(letter_files, IMAGES_PER_LETTER)

        dst_folder = os.path.join(DEST_DIR, letter)
        os.makedirs(dst_folder, exist_ok=True)

        # Extract only the chosen images directly to your C: drive
        for file_path in chosen:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(dst_folder, filename)
            
            # Read the image from the zip and write it to the destination
            with z.open(file_path) as source, open(dest_path, "wb") as target:
                target.write(source.read())
            
            total_copied += 1

        print(f"✅ {letter}: extracted {len(chosen)} images")

print(f"\n🎉 Done! Total images extracted: {total_copied}")
print(f"📁 Location: {DEST_DIR}")