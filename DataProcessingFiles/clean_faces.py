import os
import cv2
from PIL import Image
import face_recognition

def preprocess_and_convert_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        input_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}.jpg")

        try:
            # Load and detect face
            img = face_recognition.load_image_file(input_path)
            face_locations = face_recognition.face_locations(img)

            if not face_locations:
                print(f"❌ No face found in: {filename}")
                continue

            # Use first face detected
            top, right, bottom, left = face_locations[0]
            face_crop = img[top:bottom, left:right]

            # Convert to BGR (OpenCV) and normalize lighting
            face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            img_yuv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            normalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

            # Resize to 160x160
            resized = cv2.resize(normalized, (160, 160))

            # Convert back to RGB and save as JPEG
            final_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            final_img.save(output_path, "JPEG", quality=95)

            print(f"✅ Processed: {filename} → {base_name}.jpg")

        except Exception as e:
            print(f"⚠️ Failed on {filename}: {e}")

if __name__ == "__main__":
    preprocess_and_convert_images("TestPhotos", "./TestPhotosCleaned")