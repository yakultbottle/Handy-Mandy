import cv2
import requests
import numpy as np
from keras.models import load_model

# URL for ESP32 video stream
url = 'http://192.168.0.172/capture'  # Ensure the correct endpoint for the image capture

# Load the saved model
model = load_model('gesture_recognition_model.keras')

# Define the gesture names
gesture_names = [
    'palm front',
    'L',
    'fist front',
    'fist side',
    'thumbs up',
    'index',
    'ok',
    'palm side',
    'C',
    'thumbs down'
]

# Initialize region and thresholding parameters
cap_region_y_end = 0.8  # Adjust the region as needed
cap_region_x_begin = 0.5  # Adjust the region as needed
blurValue = 41
threshold = 127

def preprocess_image(image):
    # Clip the ROI
    img = image[0:int(cap_region_y_end * image.shape[0]),
                int(cap_region_x_begin * image.shape[1]):image.shape[1]]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert binary image to RGB format
    rgb_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    
    # Resize the image to match the model's input shape
    resized_image = cv2.resize(rgb_image, (64, 64))
    
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
    # Get the predicted gesture labels and their confidence scores
    predictions = sorted(zip(gesture_names, pred_array[0]), key=lambda x: x[1], reverse=True)
    return predictions

def main():
    while True:
        # Read a frame from the ESP32-CAM
        response = requests.get(url)
        if response.status_code == 200:
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            print("Failed to get frame from ESP32-CAM")
            continue

        # Perform gesture prediction
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            print("Error: Invalid frame shape. Skipping prediction.")
            continue

        predictions = predict_image(frame)
        result, score = predictions[0]

        # Preprocess the frame for display
        preprocessed_frame = preprocess_image(frame)
        preprocessed_frame_display = cv2.resize(preprocessed_frame, (frame.shape[1], frame.shape[0]))
        preprocessed_frame_display = (preprocessed_frame_display * 255).astype(np.uint8)

        # Create a blank canvas
        canvas_height = frame.shape[0] + 230
        canvas_width = frame.shape[1] * 2 + 20
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Place the original frame
        canvas[:frame.shape[0], :frame.shape[1]] = frame

        # Place the preprocessed frame to the right of the original frame
        canvas[:preprocessed_frame_display.shape[0], frame.shape[1] + 20:frame.shape[1] * 2 + 20] = preprocessed_frame_display

        # Add labels to the canvas
        cv2.putText(canvas, "Webcam", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Transformed", (frame.shape[1] + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw grey background for confidence levels
        confidence_bg_start_x = frame.shape[1] + 10
        confidence_bg_start_y = preprocessed_frame_display.shape[0] + 10
        confidence_bg_end_x = canvas_width
        confidence_bg_end_y = canvas_height - 50
        canvas[confidence_bg_start_y:confidence_bg_end_y, confidence_bg_start_x:confidence_bg_end_x] = (38, 27, 26)  # RGB: (38, 27, 26)

        # Add confidence levels text
        cv2.putText(canvas, "Confidence Levels", (frame.shape[1] + 20, preprocessed_frame_display.shape[0] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Add predictions to the canvas with reduced font size and gap
        for i, (gesture, confidence) in enumerate(predictions[:3]):
            text = f'{i+1}. {gesture}: {confidence*100:.2f}%'
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(canvas, text, (frame.shape[1] + 20, preprocessed_frame_display.shape[0] + 80 + 40 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # Draw grey rectangle at the bottom
        cv2.rectangle(canvas, (0, canvas_height - 50), (canvas_width, canvas_height), (66, 46, 41), -1)  # RGB: (66, 46, 41)

        # Add a small gap between "Press 'q' to quit" and the image above
        quit_text = "Press 'q' to quit"
        text_size = cv2.getTextSize(quit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (canvas_width - text_size[0]) // 2
        text_y = canvas_height - 20
        cv2.putText(canvas, quit_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Display the canvas
        cv2.imshow('Gesture Recognition', canvas)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

