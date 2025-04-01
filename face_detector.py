#importing cv2 library
import cv2
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import os


def detect_face():
    print("Detecting faces...")
    #loading haar case algorithm file into alg variable
    alg_file = "haarcascade_frontalface_default.xml"
    #passing algo to OpenCV
    haarcascade = cv2.CascadeClassifier(alg_file)
    #loading image path into file_name variable -- replace <INSERT YOUR IMAGE NAME HERE> with the path to image
    file_name="family_pic.jpg"
    #reading img
    img = cv2.imread(file_name, 0)
    # creating black and white version of image
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #detecting faces
    faces = haarcascade.detectMultiScale(grey_img, scaleFactor=1.05, minNeighbors=6, minSize=(80, 80))

    i = 0
    #for each face detected cycle through and select only face from image and upload as lone image file
    for x, y, w, h in faces:
        #crop image to pull only face (face card is insane btw)
        cropped_img = img[y: y+h, x: x+w]
        #laoding target image path into target_file_nam
        target_file_name = 'stored-faces/' +str(i) + '.jpg'
        cv2.imwrite(target_file_name, cropped_img)
        i += 1
    print("See directory for faces detected!")
detect_face()