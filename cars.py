#car detection movie based

import cv2
import sys

face_cascade = cv2.CascadeClassifier('data/haarcascade_car.xml')

video_capture = cv2.VideoCapture("cars.mp4")


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    autos = face_cascade.detectMultiScale(gray,
      scaleFactor = 1.1,
      minNeighbors = 8, 
      minSize = (20,20),
      maxSize = (200,200)
    )
# minNeighbors, for gouped boxes and detect object
# minSize,
# maxSize, 

    # Draw a rectangle around the faces
    for (x, y, w, h) in autos:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

    #key = cv2.waitKey(0)
    #if key == 27:
    #    break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()