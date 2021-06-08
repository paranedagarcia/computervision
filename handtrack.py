# hand tracking

import cv2
import time
import mediapipe as mp
import time
import imutils

cap = cv2.VideoCapture(3)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    res, img = cap.read()
    if res == False: break
    img = imutils.resize(img, width=720)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    response = hands.process(imgRGB)

    if response.multi_hand_landmarks:
        for handland in response.multi_hand_landmarks: # para cada mano
            # id y posición de cada landmark
            for id, lm in enumerate(handland.landmark):
                #print(id, lm)
                h, w, c = img.shape
                #posicion del centro
                cx, cy = int(lm.x*w), int(lm.y*h )
                #print(id, cx, cy)
                
                # destacar todos los puntos
                #cv2.circle(img, (cx, cy),5, (0,0,255), cv2.FILLED)
                # destacar un punto especifico
                if id == 0:
                    cv2.circle(img, (cx, cy),10, (0,250,250), cv2.FILLED)

            mpDraw.draw_landmarks(img, handland, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Hand", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
