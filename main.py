# import requests
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

resolution_index = 8
wCam, hCam = resolutions[resolution_index]

# def set_resolution(url: str, index: int=1, verbose: bool=False):
#     global wCam, hCam
#     try:
#         if verbose:
#             res_text = "\n".join([f"{key}: {val[0]}x{val[1]}" for key, val in resolutions.items()])
#             print("available resolutions\n{}".format(res_text))

#         if index in resolutions:
#             response = requests.get(url + "/control?var=framesize&val={}".format(index))
#             if response.status_code == 200:
#                 wCam, hCam = resolutions[index]
#                 print("Resolution set to index", index, ":", wCam, "x", hCam)
#             else:
#                 print("Failed to set resolution")
#         else:
#             print("Wrong index")
#     except Exception as e:
#         print("SET_RESOLUTION: something went wrong", e)

# def set_quality(url: str, value: int=1, verbose: bool=False):
#     try:
#         if value >= 10 and value <= 63:
#             response = requests.get(url + "/control?var=quality&val={}".format(value))
#             if response.status_code == 200:
#                 print("Quality set to", value)
#             else:
#                 print("Failed to set quality")
#         else:
#             print("Invalid quality value")
#     except Exception as e:
#         print("SET_QUALITY: something went wrong", e)

# def set_awb(url: str, awb: int=1):
#     try:
#         awb = not awb
#         response = requests.get(url + "/control?var=awb&val={}".format(1 if awb else 0))
#         if response.status_code == 200:
#             print("AWB set to", "enabled" if awb else "disabled")
#         else:
#             print("Failed to set AWB")
#     except Exception as e:
#         print("SET_AWB: something went wrong", e)
#     return awb

# def set_default_settings(url: str):
#     print("Setting default settings...")
#     set_resolution(url, index=resolution_index)  # Use default resolution index
#     set_quality(url, value=10)   # Quality is from 10-63, lower number is higher quality
#     awb = 1                      # Assume AWB is initially enabled
#     awb = set_awb(url, awb)      # Toggle AWB
#     print("Default settings applied.")

url = 'http://172.20.10.13'
# set_default_settings(url)

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
            # # Send the gesture to ESP32
            # try:
            #     response = requests.post(url + '/gesture', json={"gesture": gesture})
            #     if response.status_code == 200:
            #         print("Gesture sent:", gesture)
            #     else:
            #         print("Failed to send gesture")
            # except Exception as e:
            #     print("SEND_GESTURE: something went wrong", e)
            
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
