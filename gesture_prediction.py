import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model('gesture_recognition_model.keras')

# Define the gesture names
gesture_names = [
    'palm',
    'l',
    'fist',
    'fist_side',
    'thumb',
    'index',
    'ok',
    'palm_side',
    'c',
    'down'
]

def preprocess_image(image):
    # Resize the image to match the model's input shape
    resized_image = cv2.resize(image, (64, 64))
    # Normalize the pixel values to the range [0, 1]
    resized_image = resized_image / 255.0
    return resized_image

def predict_image(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Reshape the image to match the input shape of the model
    processed_image = np.expand_dims(processed_image, axis=0)
    # Perform prediction
    pred_array = model.predict(processed_image)
    # Get the predicted gesture label
    result = gesture_names[np.argmax(pred_array)]
    # Calculate the confidence score
    score = float("%0.2f" % (np.max(pred_array) * 100))
    print(f'Result: {result}, Score: {score}')
    return result, score

def main():
    # Open the webcam
    camera = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not camera.isOpened():
        print("Error: Unable to open the webcam.")
        return

    while True:
        # Read a frame from the webcam
        ret, frame = camera.read()

        # Check if the frame is read successfully
        if not ret:
            print("Error: Unable to read frame from the webcam.")
            break

        # Print the shape of the frame before resizing
        print(f"Frame shape before resize: {frame.shape}")

        # Perform gesture prediction
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            print("Error: Invalid frame shape. Skipping prediction.")
            continue

        result, score = predict_image(frame)

        # Display the frame with the predicted gesture label and confidence level
        text = f'{result}: {score}%'
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Webcam Feed', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

