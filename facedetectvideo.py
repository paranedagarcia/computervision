import cv2
import numpy as np
import matplotlib.pyplot as plt

# 0, ; 2, OBS
# 3, manycam
cap = cv2.VideoCapture(3)

faceClassif = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

while True:
	ret,frame = cap.read()
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceClassif.detectMultiScale(frame,
    scaleFactor = 1.1,
    minNeighbors = 8
    )

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

	cv2.imshow('frame',frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()