import requests
import handGesture as htm
import time
import cv2

# Camera resolutions dictionary
resolutions = {
    10: (1600, 1200),  # UXGA
    9:  (1280, 1024),  # SXGA
    8:  (1024, 768),   # XGA
    7:  (800, 600),    # SVGA
    6:  (640, 480),    # VGA
    5:  (400, 296),    # CIF
    4:  (320, 240),    # QVGA
    3:  (240, 176),    # HQVGA
    0:  (160, 120)     # QQVGA
}

def send_gesture(url: str, gesture: str):
    try:
        response = requests.post(url + "/gesture", gesture)
        if response.status_code == 200:
            print("Gesture sent successfully:", gesture)
        else:
            print("Failed to send gesture")
    except Exception as e:
        print("SEND_GESTURE: something went wrong")

resolution_index = 8
wCam, hCam = resolutions[resolution_index]

url = 'http://172.20.10.13'

# Initialize the webcam
cap = cv2.VideoCapture(url + ":81/stream")
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1, modelComplexity=0)  # Lower model complexity for better performance

pTime = 0  # Initialize pTime for FPS calculation

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 0)  # Up/Down flip
    img = detector.findHands(img, draw=False)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        palmOrientation = detector.detectPalmOrientation()
        fingers = detector.fingersUp(palmOrientation)
        gesture = detector.detectGestures(fingers, palmOrientation)

        if gesture:
            send_gesture(url, gesture)
            
            cv2.putText(img, gesture, (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.putText(img, "Hand Detected", (50, 200), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    else:
        cv2.putText(img, "No Hand Detected", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

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
