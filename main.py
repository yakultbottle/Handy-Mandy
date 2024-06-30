# import numpy as np
import requests
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

def set_resolution(url: str, index: int=1, verbose: bool=False):
    try:
        if verbose:
            resolutions = "10: UXGA(1600x1200)\n9: SXGA(1280x1024)\n8: XGA(1024x768)\n7: SVGA(800x600)\n6: VGA(640x480)\n5: CIF(400x296)\n4: QVGA(320x240)\n3: HQVGA(240x176)\n0: QQVGA(160x120)"
            print("available resolutions\n{}".format(resolutions))

        if index in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
            response = requests.get(url + "/control?var=framesize&val={}".format(index))
            if response.status_code == 200:
                print("Resolution set to index", index)
            else:
                print("Failed to set resolution")
        else:
            print("Wrong index")
    except Exception as e:
        print("SET_RESOLUTION: something went wrong", e)

def set_quality(url: str, value: int=1, verbose: bool=False):
    try:
        if value >= 10 and value <=63:
            response = requests.get(url + "/control?var=quality&val={}".format(value))
            if response.status_code == 200:
                print("Quality set to", value)
            else:
                print("Failed to set quality")
        else:
            print("Invalid quality value")
    except Exception as e:
        print("SET_QUALITY: something went wrong", e)

def set_awb(url: str, awb: int=1):
    try:
        awb = not awb
        response = requests.get(url + "/control?var=awb&val={}".format(1 if awb else 0))
        if response.status_code == 200:
            print("AWB set to", "enabled" if awb else "disabled")
        else:
            print("Failed to set AWB")
    except Exception as e:
        print("SET_AWB: something went wrong", e)
    return awb

def set_default_settings(url: str):
    print("Setting default settings...")
    # 10 –> UXGA(1600×1200)
    #  9 –> SXGA(1280×1024)
    #  8 –> XGA(1024×768)
    #  7 –> SVGA(800×600)
    #  6 –> VGA(640×480)
    #  5 —> CIF(400×296)
    #  4 –> QVGA(320×240)
    #  3 –> HQVGA(240×176)
    #  0 –> QQVGA(160×120)
    set_resolution(url, index=6) # VGA
    set_quality(url, value=30)   # Quality is from 10-63, lower number is higher quality
    awb = 1                      # Assume AWB is initially enabled
    awb = set_awb(url, awb)      # Toggle AWB
    print("Default settings applied.")

url = 'http://172.20.10.11'
set_default_settings(url)

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
