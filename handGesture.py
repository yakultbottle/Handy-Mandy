import cv2
import mediapipe as mp
# import numpy as np
 
 
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
        self.handLabel = None  # Attribute to store hand label

    def findHands(self, img, draw=True):
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)
        # self.results = self.hands.process(imgRGB)
        
        if self.results.multi_handedness:
            self.handLabel = self.results.multi_handedness[0].classification[0].label  # Extract hand label
            cv2.putText(img, self.handLabel, (50, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

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
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
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

    def fingersUp(self, palmOrientation):
        fingers = []
        # Thumb
        if self.handLabel == "Right":
            if palmOrientation == "facing towards":
                if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
                    fingers.append(1)
                    # print(f"Right, towards, stick out")
                else:
                    fingers.append(0)
                    # print(f"Right, towards, stick in")
            else:  # facing away
                if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
                    fingers.append(0)
                    # print(f"Right, away, stick in")
                else:
                    fingers.append(1)
                    # print(f"Right, away, stick out")
        elif self.handLabel == "Left":
            if palmOrientation == "facing towards":
                if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                    fingers.append(1)
                    # print(f"Left, towards, stick out")
                else:
                    fingers.append(0)
                    # print(f"Left, towards, stick in")
            else:  # facing away
                if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                    fingers.append(0)
                    # print(f"Left, away, stick in")
                else:
                    fingers.append(1)
                    # print(f"Left, away, stick out")

        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def detectPalmOrientation(self):
        if not self.lmList or not self.handLabel:
            return None

        thumb_tip_x = self.lmList[4][1]
        pinky_tip_x = self.lmList[20][1]

        palm_direction = None

        if self.handLabel == "Right":
            if thumb_tip_x < pinky_tip_x:
                palm_direction = "facing towards"
            else:
                palm_direction = "facing away"
        elif self.handLabel == "Left":
            if thumb_tip_x > pinky_tip_x:
                palm_direction = "facing towards"
            else:
                palm_direction = "facing away"

        # print(f"Palm Orientation: {palm_direction}")
        return palm_direction

    def detectGestures(self, fingers, palmOrientation):
        gesture = None

        if fingers == [1, 1, 1, 1, 1]:
            if palmOrientation == "facing towards":
                gesture = "Open Palm (Stop)"
            elif palmOrientation == "facing away":
                gesture = "Open Palm (Go)"

        elif fingers == [1, 0, 0, 0, 0]:
            # Check thumb direction for a closed fist with thumb sticking out
            if self.handLabel == "Right":
                if self.lmList[4][1] > self.lmList[3][1]:
                    gesture = "Closed Fist (Thumb Right)"
                else:
                    gesture = "Closed Fist (Thumb Left)"
            elif self.handLabel == "Left":
                if self.lmList[4][1] < self.lmList[3][1]:
                    gesture = "Closed Fist (Thumb Left)"
                else:
                    gesture = "Closed Fist (Thumb Right)"
        
        elif fingers == [0, 0, 1, 0, 0]:
            gesture = "Middle finger"

        return gesture
