# Handy Mandy

## Webcam Gesture Recognition Model Instructions

This document provides instructions for setting up and running the gesture recognition model on your webcam

## Prerequisites

- Python 3.7 or higher (Python 3.11.9 was used)
  - [Download Python](https://www.python.org/downloads/)

## Instructions for running the model

(Optional) It is recommended to install all Python dependencies in a virtual environment[^1], but you can skip this step

[^1]: [Best practice](https://stackoverflow.com/questions/41972261/what-is-a-virtualenv-and-why-should-i-use-one) is to use virtual environments to manage dependency conflicts between projects

Windows Powershell:
```powershell
python -m venv myenv
myenv\Scripts\activate
cd "myenv"
```

MacOS/Linux bash:
```bash
python3 -m venv myenv
source myenv/bin/activate
cd "myenv"
```

1. Unzip the folder into virtual environment

2. Install all dependencies

Windows Powershell:
```
cd "Handy Mandy"
pip install -r .\requirements.txt
```

MacOS/Linux bash:
```bash
cd "Handy Mandy"
pip install -r ./requirements.txt
```
3. Adjust the settings on the model

Webcam dimensions(Line 7, webcam.py) 
```
wCam, hCam = 1280, 720
```

4. Run the model (**Note:** this will not ask for permission before running your webcam!)

Windows Powershell:
```powershell
python3 .\webcam.py
```

MacOS/Linux bash:
```bash
python3 ./webcam.py
```

5. If left/right hand is not accurately determined, check the following settings in the model:

Comment out Line 24, webcam.py
```
img = cv2.flip(img, 1)
```