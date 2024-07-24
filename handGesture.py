import cv2
import mediapipe as mp


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
        self.results = self.hands.process(img)
        
        if self.results.multi_handedness:
            self.handLabel = self.results.multi_handedness[0].classification[0].label
            cv2.putText(img, self.handLabel, (50, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        xList = []
        yList = []
        bbox = []
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

        return self.lmList, bbox

    def fingersUp(self, palmOrientation):
        fingers = []
        # Thumb
        if self.handLabel == "Right":
            if palmOrientation == "facing towards":
                fingers.append(int(self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]))
            else:
                fingers.append(int(self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]))
        elif self.handLabel == "Left":
            if palmOrientation == "facing towards":
                fingers.append(int(self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]))
            else:
                fingers.append(int(self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]))

        # Fingers
        for id in range(1, 5):
            fingers.append(int(self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]))

        return fingers

    def detectPalmOrientation(self):
        if not self.lmList or not self.handLabel:
            return None

        thumb_tip_x = self.lmList[4][1]
        pinky_tip_x = self.lmList[20][1]

        if self.handLabel == "Right":
            palm_direction = "facing towards" if thumb_tip_x < pinky_tip_x else "facing away"
        else:
            palm_direction = "facing towards" if thumb_tip_x > pinky_tip_x else "facing away"

        return palm_direction

    def detectGestures(self, fingers, palmOrientation):
        if fingers == [1, 1, 1, 1, 1]:
            return "Stop" if palmOrientation == "facing towards" else "Go"
        elif fingers == [1, 0, 0, 0, 0]:
            if self.handLabel == "Right":
                return "Right" if self.lmList[4][1] > self.lmList[3][1] else "Left"
            else:
                return "Left" if self.lmList[4][1] < self.lmList[3][1] else "Right"
        elif fingers == [0, 1, 1, 0, 0]:
            return "Dance"
        elif fingers == [0, 0, 1, 0, 0]:
            return "Middle Finger"
        return None
