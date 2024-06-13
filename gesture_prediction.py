import cv2
import numpy as np
import asyncio
import websockets
from keras.models import load_model

# URL for ESP32 video stream
uri = 'ws://192.168.0.172/ws'  # Ensure the correct endpoint for the WebSocket connection

# Load the saved model
model = load_model('gesture_recognition_model.keras')

# Define the gesture names
gesture_names = [
    'palm front', 'L', 'fist front', 'fist side', 'thumbs up',
    'index', 'ok', 'palm side', 'C', 'thumbs down'
]

def preprocess_image(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)
    resized_image = cv2.resize(rgb_image, (64, 64))
    resized_image = resized_image / 255.0
    return resized_image

def predict_image(image):
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    pred_array = model.predict(processed_image)
    predictions = sorted(zip(gesture_names, pred_array[0]), key=lambda x: x[1], reverse=True)
    return predictions

async def receive_video():
    async with websockets.connect(uri) as websocket:
        while True:
            data = await websocket.recv()
            img_array = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame.shape[0] == 0 or frame.shape[1] == 0:
                print("Error: Invalid frame shape. Skipping prediction.")
                continue
            predictions = predict_image(frame)
            result, score = predictions[0]

            preprocessed_frame = preprocess_image(frame)
            preprocessed_frame_display = cv2.resize(preprocessed_frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            preprocessed_frame_display = (preprocessed_frame_display * 255).astype(np.uint8)

            canvas_height = max(frame.shape[0], preprocessed_frame_display.shape[0] + 150)
            canvas_width = frame.shape[1] + preprocessed_frame_display.shape[1] + 20
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            canvas[:frame.shape[0], :frame.shape[1]] = frame
            canvas[:preprocessed_frame_display.shape[0], frame.shape[1] + 10:frame.shape[1] + 10 + preprocessed_frame_display.shape[1]] = preprocessed_frame_display

            cv2.putText(canvas, "Webcam", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Transformed", (frame.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Image", (frame.shape[1] + 20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            confidence_bg_start_x = frame.shape[1] + 10
            confidence_bg_start_y = preprocessed_frame_display.shape[0] + 10
            confidence_bg_end_x = canvas_width
            confidence_bg_end_y = canvas_height - 50
            canvas[confidence_bg_start_y:confidence_bg_end_y, confidence_bg_start_x:confidence_bg_end_x] = (38, 27, 26)

            cv2.putText(canvas, "Confidence Levels", (frame.shape[1] + 20, preprocessed_frame_display.shape[0] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            for i, (gesture, confidence) in enumerate(predictions[:3]):
                text = f'{i+1}. {gesture}: {confidence*100:.2f}%'
                color = (0, 255, 0) if i == 0 else (255, 255, 255)
                cv2.putText(canvas, text, (frame.shape[1] + 20, preprocessed_frame_display.shape[0] + 80 + 40 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            cv2.rectangle(canvas, (0, canvas_height - 50), (canvas_width, canvas_height), (66, 46, 41), -1)

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
    asyncio.get_event_loop().run_until_complete(receive_video())

