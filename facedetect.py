# face detection image based

import cv2 as cv2
import numpy as np

faceClassif = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

image = cv2.imread('images/workshop.jpg')
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceClassif.detectMultiScale(image,
  scaleFactor = 1.1,
  minNeighbors = 8,
  minSize = (30,30),
  maxSize = (100,100)
  )

for (x,y,w,h) in faces:
  cv2.rectangle(image, (x,y),(x+w, y+h), (0,0,255), 2)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
