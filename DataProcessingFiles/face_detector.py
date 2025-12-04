import cv2
import os


def detect_faces_from_folder(folder_path, output_folder="../TestPhotosCleaned"):

    #load the algo
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(filepath)

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Draw rectangles around the detected faces and save them
            for i, (x, y, w, h) in enumerate(faces):
                # Add 30% padding around face to match FIW style
                padding = int(0.3 * w)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(img.shape[1], x + w + padding)
                y2 = min(img.shape[0], y + h + padding)

                cropped_face = img[y1:y2, x1:x2]
                # Resize to square for consistency
                cropped_face = cv2.resize(cropped_face, (160, 160))
                cv2.imwrite(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_face_{i}.jpg"), cropped_face)

            # cv2.imwrite(os.path.join(output_folder, f"detected_{filename}"), img)
    print("Face detection complete. Detected faces are saved in the 'detected_faces' folder.")


folder_path = "../TestPhotos"
detect_faces_from_folder(folder_path)