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
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cropped_face = img[y:y + h, x:x + w]
                cv2.imwrite(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_face_{i}.jpg"), cropped_face)

            # Optionally, save the image with bounding boxes
            # cv2.imwrite(os.path.join(output_folder, f"detected_{filename}"), img)
    print("Face detection complete. Detected faces are saved in the 'detected_faces' folder.")


# Example usage:
folder_path = "../TestPhotos"  # Replace with the path to your folder
detect_faces_from_folder(folder_path)