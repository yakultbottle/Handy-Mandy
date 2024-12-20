#include "esp_camera.h"
#include <WiFi.h>
// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"

#define CAMERA_MODEL_AI_THINKER  // Has PSRAM
#include "camera_pins.h"
#include <ESP32Servo.h>
#include "servo_code.h"

// ===========================
// Enter your WiFi credentials
// ===========================
#include "wifi_config.h"

extern char *received_gesture;

void startCameraServer();
void setupLedFlash(int pin);

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_XGA;     // Set to XGA (1024x768)
  config.pixel_format = PIXFORMAT_JPEG;  // For streaming
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // If PSRAM IC present, init with higher JPEG quality and more frame buffers
  if (config.pixel_format == PIXFORMAT_JPEG && psramFound()) {
    config.jpeg_quality = 10;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
  }

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);        // flip it back
    s->set_brightness(s, 1);   // up the brightness just a bit
    s->set_saturation(s, -2);  // lower the saturation
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif

#if defined(CAMERA_MODEL_ESP32S3_EYE)
  s->set_vflip(s, 1);
#endif

// Setup LED Flash if LED pin is defined in camera_pins.h
#if defined(LED_GPIO_NUM)
  setupLedFlash(LED_GPIO_NUM);
#endif

  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  startCameraServer();
  setupServo();
  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");


  // // Create a new task for controlling servos
  // xTaskCreate(controlServos, "ServoControlTask", 2048, NULL, 1, NULL);
}

void loop() {
  // Print the received gesture if it's not NULL
  // if (received_gesture) {
  //   Serial.printf("Received gesture: %s\n", received_gesture);
  //   received_gesture = NULL; // Clear the received gesture after printing
  // }

  // // Do nothing. Everything is done in another task by the web server
  // // vTaskDelay(10000 / portTICK_PERIOD_MS);  // Non-blocking delay

  if (received_gesture) {
    Serial.printf("Received gesture: %s\n", received_gesture);
    if (strcmp(received_gesture, "Go") == 0) {
      Serial.println("Executing Go Command");
      forward();
      received_gesture = NULL;
    } else if (strcmp(received_gesture, "Stop") == 0) {
      Serial.println("Executing Stop Command");
      stop();
      received_gesture = NULL;
    } else if (strcmp(received_gesture, "Left") == 0) {
      Serial.println("Executing Left Command");
      left();
      received_gesture = NULL;
    } else if (strcmp(received_gesture, "Right") == 0) {
      Serial.println("Executing Right Command");
      right();
      received_gesture = NULL;
    } 
    else if (strcmp(received_gesture, "Dance") == 0) {
      Serial.println("Executing Dance Command");
      dance();
      received_gesture = NULL;
    }else {
      Serial.println("Unknown Gesture");
      received_gesture = NULL;
    }
  }
}

// // Task for controlling servos
// void controlServos(void *pvParameters) {
//   while (true) {
//     // Add servo control logic here
//     // For example, read from a queue to get commands and execute them

//     // Non-blocking delay to prevent task from hogging the CPU
//     vTaskDelay(100 / portTICK_PERIOD_MS);
//   }
// }
