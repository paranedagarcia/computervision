# Fase 1 - Reconocimiento de caras
# Detecta y captura imagenes para entrenamiento
# ...
import cv2
import os
import imutils

persona = 'patricio'
dataPath = '/Users/patricio/git/computervision/data'
personaPath = dataPath + '/'+ persona

if not os.path.exists(personaPath):
    os.makedirs(personaPath)

#cap = cv2.VideoCapture(3) # stream capture
cap = cv2.VideoCapture("data/patricio.mov") # video capture

faceClassif = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
count = 0

while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    newFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
        face = newFrame[y:y+h, x:x+w]
        face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personaPath + '/cara_{}.jpg'.format(count), face)
        count = count + 1
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27 or count >=300:
        break

cap.release()
cv2.destroyAllWindows()
