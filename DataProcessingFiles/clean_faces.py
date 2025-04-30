import cv2
import os

for file in os.listdir('../TestPhotos'):
    img = cv2.imread('TestPhotos/' + file)
    print(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
        faces = img[y:y+h, x:x+w]
        cv2.imshow('face', faces)
        cv2.imwrite('TestPhotosCleaned/' + file, faces)
cv2.imshow('face', img)
cv2.waitKey(0)