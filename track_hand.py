import cv2
import mediapipe as mp
# import time
# import math
import numpy as np
 
 
class handDetector():
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, 
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.handLabel = None
 
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        # Checks and updates which hand is being raised
        if self.results.multi_handedness:
            self.handLabel = self.results.multi_handedness[0].classification[0].label
            cv2.putText(img, self.handLabel, (50, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # Finds the hand landmarks, and draws them on screen
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
 
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
 
            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)
 
        return self.lmList, bbox
 
    # Returns an array of bools representing open/closed. 
    # 1 represents open, and 0 represents closed
    # eg [0, 0, 1, 0, 0] is a middle finger
    def fingersUp(self):
        fingers = []

        # Thumb
        # Checks whether the tip of thumb is more "left" or "right" than the base of the thumb
        # Based on left/right hand, decides whether thumb is open/closed
        # Note, Left/Right is weird depending on mirror of webcam. Deal with it once we get to esp32. 
        if self.handLabel == "Left":
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        elif self.handLabel == "Right":
            if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
 
        # Fingers
        # Checks that tip of finger is higher than the base of the finger
        # Will not work properly if the camera is sideways or upside-down
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
 
        # totalFingers = fingers.count(1)
 
        return fingers

    def detectPalmOrientation(self):
        if not self.lmList:
            return None
        palm_direction = None

        # Compare wrist to middle finger MCP joint
        if self.lmList[0][2] > self.lmList[9][2]:
            palm_direction = "facing away"
        else:
            palm_direction = "facing towards"

        return palm_direction
    
#     def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
#         x1, y1 = self.lmList[p1][1:]
#         x2, y2 = self.lmList[p2][1:]
#         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
 
#         if draw:
#             cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
#             cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
#             cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
#             cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
#         length = math.hypot(x2 - x1, y2 - y1)
 
#         return length, img, [x1, y1, x2, y2, cx, cy]
 
 
# def main():
#     pTime = 0
#     cTime = 0
#     cap = cv2.VideoCapture(0)
#     detector = handDetector()
#     while True:
        
#         success, img = cap.read()
        
#         img = detector.findHands(img)
#         lmList, bbox = detector.findPosition(img)
#         if len(lmList) != 0:
#             print(lmList[4])
 
#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime
#         fingers = detector.fingersUp()
 
#         cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
#                     (255, 0, 255), 3)
 
#         cv2.imshow("Image", img)
#         cv2.waitKey(1)
 
 
# if __name__ == "__main__":
#     main()