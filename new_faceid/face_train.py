import cv2
import os
import numpy as np
from PIL import Image

path = os.path.dirname(os.path.abspath(__file__))
recognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
dataPath = path+r'/dataSet'

def get_images_and_labels(datapath):
     image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]
     images = []
     labels = []
     for image_path in image_paths:
         image_pil = Image.open(image_path).convert('L')
         image = np.array(image_pil, 'uint8')
         nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
         faces = faceCascade.detectMultiScale(image)
         for (x, y, w, h) in faces:
             images.append(image[y: y + h, x: x + w])
             labels.append(nbr)
             cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
             cv2.waitKey(100)
     return images, labels

images, labels = get_images_and_labels(dataPath)
recognizer.train(images, np.array(labels))
recognizer.save(path+r'/trainer/trainer.yml')
cv2.destroyAllWindows()