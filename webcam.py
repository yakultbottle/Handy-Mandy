import numpy as np
import handGesture as htm
import time
import cv2

# Camera parameters
wCam, hCam = 320, 240
frameR = 100  # Frame Reduction
smoothening = 7

# Previous and current location for smoothing
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

url = 'http://172.20.10.11'

# Initialize the webcam
cap = cv2.VideoCapture(url + ":81/stream")
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        palmOrientation = detector.detectPalmOrientation()
        fingers = detector.fingersUp(palmOrientation)
        gesture = detector.detectGestures(fingers, palmOrientation)

        if gesture:
            cv2.putText(img, gesture, (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # Frame rate calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Display
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
