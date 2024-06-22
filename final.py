import numpy as np
import track_hand as htm
import time
import cv2

# Camera parameters
wCam, hCam = 1280, 720
frameR = 100  # Frame Reduction
smoothening = 7

# Previous and current location for smoothing
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)

while True:
    # Find hand landmarks
    fingers = [0, 0, 0, 0, 0]
    success, img = cap.read()

    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Get the tip of the index and middle fingers
    if len(lmList) != 0:
        # Check which fingers are up
        fingers = detector.fingersUp()
        # Check palm facing camera or not
        palm_orientation = detector.detectPalmOrientation()

        # if fingers == [1, 1, 1, 1, 1]:
        #     cv2.putText(img, "Open Palm", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # if fingers == [0, 0, 0, 0, 0]:
        #     cv2.putText(img, "Closed Fist", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        if fingers == [1, 1, 1, 1, 1]:
            if palm_orientation == "facing towards":
                cv2.putText(img, "Open Palm (Facing Towards)", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            elif palm_orientation == "facing away":
                cv2.putText(img, "Open Palm (Facing Away)", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        if fingers == [0, 0, 0, 0, 0]:
            cv2.putText(img, "Closed Fist", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        
        # # Only index finger
        # if fingers[1] == 1 and fingers[2] == 0:
        #     # Convert coordinates
        #     x3 = np.interp(x1, (frameR, wCam - frameR), (0, wCam))
        #     y3 = np.interp(y1, (frameR, hCam - frameR), (0, hCam))
        #     # Smoothen values
        #     clocX = plocX + (x3 - plocX) / smoothening
        #     clocY = plocY + (y3 - plocY) / smoothening

        #     # Draw a circle at the tip of the index finger
        #     cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        #     plocX, plocY = clocX, clocY
        
        # # Both index and middle fingers are up: Clicking mode
        # if fingers[1] == 1 and fingers[2] == 1:
        #     # Find distance between fingers
        #     length, img, lineInfo = detector.findDistance(8, 12, img)
        #     # Indicate click when fingers are close
        #     if length < 40:
        #         cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)

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
