# Fase 3 - Reconocimiento de caras
# Probando modelo de reconocimiento
#
import cv2
import os
import imutils

dataPath = '/Users/patricio/git/computervision/data'
# captura lista de directorios dentro de la ruta dataPath, un directorio por persona
personas = [filename for filename in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, filename))] 
print('Rostros: ', personas)

face_recon = cv2.face.EigenFaceRecognizer_create()

face_recon.read('data/modelEigenface.xml')

# prueba del modelo
cap = cv2.VideoCapture(3) # desde webcam
#cap = cv2.VideoCapture('data/patricio.mov') # desde un video

faceClassif = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    newFrame = gray.copy()

    rostros = faceClassif.detectMultiScale(gray, 1.3,5)

    for (x,y,w,h) in rostros:
        rostro = newFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)
        response = face_recon.predict(rostro)

        cv2.putText(frame, '{}'.format(response), (x, y-5), 1, 1.2, (255,255,0),1,cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(frame, '{}'.format(personas[0]), (x,y -30), 1, 1.4,(0,255,0),1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



