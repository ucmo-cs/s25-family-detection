import cv2
import numpy as np
import os

def match_fiw_style(input_path, output_path):
    """Light processing to match FIW dataset style"""
    img = cv2.imread(input_path)

    if img is None:
        print(f"Failed to load: {input_path}")
        return

    # Just resize to 160x160
    img = cv2.resize(img, (160, 160))

    # Very light blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Slight saturation reduction
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.85
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    cv2.imwrite(output_path, img)
    print(f"Processed: {output_path}")

# Process your photos
input_folder = "../TestPhotosCleaned"
output_folder = "../TestPhotosCleaned/fiw_style"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        input_path = os.path.join(input_folder, filename)
        # Skip if it's not a file
        if not os.path.isfile(input_path):
            continue
        output_path = os.path.join(output_folder, filename)
        match_fiw_style(input_path, output_path)

print("\nDone! Test with photos in TestPhotosCleaned/fiw_style/")
