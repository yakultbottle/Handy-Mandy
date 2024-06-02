# Handy Mandy

## Webcam Gesture Recognition Model Instructions

This document provides instructions for setting up and running the gesture recognition model on your webcam, as well as instructions for training the model from scratch. 

## Prerequisites

- Python 3.7 or higher (Python 3.11.9 was used)
  - [Download Python](https://www.python.org/downloads/)

## Instructions for running the model

(Optional) It is recommended to install all Python dependencies in a virtual environment[^1], but you can skip this step

[^1]: [Best practice](https://stackoverflow.com/questions/41972261/what-is-a-virtualenv-and-why-should-i-use-one) is to use virtual environments to manage dependency conflicts between projects

Windows Powershell:
```powershell
python3 -m venv myenv
myenv\Scripts\activate
cd "myenv"
```

MacOS/Linux bash:
```bash
python3 -m venv myenv
source myenv/bin/activate
cd "myenv"
```

1. Clone the repo in the virtual environment or any other suitable place (this might take a while)
   
```bash
git clone git@github.com:yakultbottle/Handy-Mandy.git
```

2. Install all dependencies

Windows Powershell:
```
cd "Handy Mandy"
pip install -r .\requirements_run.txt
```

MacOS/Linux bash:
```bash
pip install -r ./requirements_run.txt
```
3. Run the model (**Note:** this will not ask for permission before running your webcam!)

Windows Powershell:
```powershell
python3 .\gesture_recognition.py
```

MacOS/Linux bash:
```bash
python3 ./gesture_recognition.py
```

## Instructions on training the model
This assumes you have already followed the instructions on running the model above
1. Download the [Hand Gesture Recognition Database](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) from Kaggle(note: it is huge, this will take forever)
2. When you first unzip the folder, you will get something like this:
```
/archive
  /leapGestRecog
    /00
      ..
    /01
      ..
    /02
      ..
    ..
    /leapGestRecog
```
There is a duplicate folder inside the first leapGestRecog folder that is not needed. Place the main leapGestRecog folder into the project directory like so(without the duplicate inside):
```
/Handy Mandy
  /leapGestRecog
    /00
      ..
    /01
      ..
    ..
  /gesture_recognition.py
```
3. Install all dependencies
```
pip install -r requirements_train.txt
```
4. Run the training model(will take a long time, like possibly 2h+)
Windows Powershell:
```powershell
python3 .\training.py
```

MacOS/Linux bash:
```bash
python3 ./training.py
```
