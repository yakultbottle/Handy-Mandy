import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('gesture_recognition_model.keras')

# Dictionary mapping gesture number to gesture label
gesture_labels = {
    1: 'palm',
    2: 'l',
    3: 'fist',
    4: 'fist_side',
    5: 'thumb',
    6: 'index',
    7: 'ok',
    8: 'palm_side',
    9: 'c',
    10: 'down'
}

def predict_gesture(frame, top_n=3):
    # Preprocess the frame for the model
    img = cv2.resize(frame, (64, 64))  # Resize the frame to match the input size of the model
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values

    # Predict the gesture
    predictions = model.predict(img)[0]

    # Get top n predictions and their corresponding labels and confidence levels
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    top_predictions = predictions[top_indices]
    top_labels = [gesture_labels.get(idx + 1, 'Unknown') for idx in top_indices]

    return top_labels, top_predictions

def main():
    # Open the webcam
    camera = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = camera.read()
        if not ret:
            break

        # Perform gesture prediction
        top_labels, top_predictions = predict_gesture(frame, top_n=3)

        # Display the frame with the predicted gesture label and confidence level
        for i, (label, confidence) in enumerate(zip(top_labels, top_predictions)):
            text = f'{label}: {confidence:.2f}'
            cv2.putText(frame, text, (10, 50 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Webcam Feed', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

