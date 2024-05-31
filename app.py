from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('train_signs.h5')
camera = cv2.VideoCapture(0)  # Change this if you have a different camera source

def predict_gesture(frame):
    # Preprocess the frame for the model
    img = cv2.resize(frame, (64, 64))  # Adjust based on your model's input size
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Predict the gesture
    prediction = model.predict(img)
    gesture = np.argmax(prediction)
    return gesture

def gen_frames():
    while True:
        success, frame = camera.read()  # Read the camera frame
        if not success:
            break
        else:
            gesture = predict_gesture(frame)
            # Add text or other annotations to the frame
            cv2.putText(frame, f'Gesture: {gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

