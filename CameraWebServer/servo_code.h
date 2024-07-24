#ifndef SERVO_CODE_H
#define SERVO_CODE_H
#include <ESP32Servo.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

Servo servorightleg;  // create servo object to control a servo
Servo servorightthigh;
Servo servoleftleg;  // create servo object to control a servo
Servo servoleftthigh;
Servo servoleftarm;
Servo servorightarm;

int right_thigh_pos = 50;  //Neutral:50, Forward: 0, Backward: 100
int right_leg_pos = 50;
int right_thigh_pin = 13;
int right_leg_pin = 12;

int left_thigh_pos = 50;  //Neutral:50, Forward: 100, Backward: 0
int left_leg_pos = 50;
int left_thigh_pin = 15;
int left_leg_pin = 14;

int ledPin = 2;

int posL = 50;  //dummy variables for incrementations
int posR = 50;
int state_L = 2;  //state of robot, F = 1, B = 0, N = 2
int state_R = 2;

const int angle_F = 40;  //angle at max walking
int num_F = 1;           //num of times for F loop

const int angle_B = 25;
int num_B = 2;  //num of times for B loop

int delay2 = 20;           //delay between each leg shuffle
const int delayTime = 20;  //decrease for faster walking

void move_right_forward(int iterations) {
  int i = 0;
  while (i < iterations) {
    servorightthigh.write(posR);
    servorightleg.write(posR);
    vTaskDelay(pdMS_TO_TICKS(delayTime));
    posR -= 1;
    i += 1;
  }
}

void move_right_neutral(int iterations, int state) {
  int i = 0;
  if (state == 1) {
    while (i < iterations) {
      servorightthigh.write(posR);
      servorightleg.write(posR);
      vTaskDelay(pdMS_TO_TICKS(delayTime));
      posR += 1;
      i += 1;
    }
  } else if (state == 0) {
    while (i < iterations) {
      servorightthigh.write(posR);
      servorightleg.write(posR);
      vTaskDelay(pdMS_TO_TICKS(delayTime));
      posR -= 1;
      i += 1;
    }
  }
}

void move_right_backward(int iterations) {
  int i = 0;
  while (i < iterations) {
    servorightthigh.write(posR);
    servorightleg.write(posR);
    vTaskDelay(pdMS_TO_TICKS(delayTime));
    posR += 1;
    i += 1;
  }
}

void move_left_forward(int iterations) {
  int i = 0;
  while (i < iterations) {
    servoleftthigh.write(posL);
    servoleftleg.write(posL);
    vTaskDelay(pdMS_TO_TICKS(delayTime));
    posL += 1;
    i += 1;
  }
}

void move_left_neutral(int iterations, int state) {
  int i = 0;
  if (state == 1) {
    while (i < iterations) {
      servoleftthigh.write(posL);
      servoleftleg.write(posL);
      vTaskDelay(pdMS_TO_TICKS(delayTime));
      posL -= 1;
      i += 1;
    }
  } else if (state == 0) {
    while (i < iterations) {
      servoleftthigh.write(posL);
      servoleftleg.write(posL);
      vTaskDelay(pdMS_TO_TICKS(delayTime));
      posL += 1;
      i += 1;
    }
  }
}

void move_left_backward(int iterations) {
  int i = 0;
  while (i < iterations) {
    servoleftthigh.write(posL);
    servoleftleg.write(posL);
    vTaskDelay(pdMS_TO_TICKS(delayTime));
    posL -= 1;
    i += 1;
  }
}

void move_leftleg_F(int iterations) {
  move_left_forward(iterations);
  state_L = 1;
  move_left_neutral(iterations, state_L);
  state_L = 2;
}

void move_rightleg_F(int iterations) {
  move_right_forward(iterations);
  state_R = 1;
  move_right_neutral(iterations, state_R);
  state_R = 2;
}

void move_leftleg_B(int iterations) {
  move_left_backward(iterations);
  state_L = 0;
  move_left_neutral(iterations, state_L);
  state_L = 2;
}

void move_rightleg_B(int iterations) {
  move_right_backward(iterations);
  state_R = 0;
  move_right_neutral(iterations, state_R);
  state_R = 2;
}

void F_loop(int num_of_times) {
  int i = 0;
  while (i < num_of_times) {
    move_leftleg_F(angle_F);
    vTaskDelay(pdMS_TO_TICKS(delay2));
    move_rightleg_F(angle_F);
    vTaskDelay(pdMS_TO_TICKS(delay2));
    i += 1;
  }
}

void forward() {
  F_loop(num_F);
  // move_rightleg_F(angle_F);
  // vTaskDelay(pdMS_TO_TICKS(delay2));
}

void B_loop(int num_of_times) {
  int i = 0;
  while (i < num_of_times) {
    move_rightleg_B(angle_B + 10);
    vTaskDelay(pdMS_TO_TICKS(delay2));
    move_leftleg_B(angle_B);
    vTaskDelay(pdMS_TO_TICKS(delay2));
    i += 1;
  }
}

void backward() {
  B_loop(num_B);
  move_rightleg_B(angle_B + 10);
  vTaskDelay(pdMS_TO_TICKS(delay2));
}

void left() {
  move_rightleg_F(40);
  vTaskDelay(pdMS_TO_TICKS(delay2));
  move_leftleg_B(25);
  vTaskDelay(pdMS_TO_TICKS(delay2));
}

void right() {
  move_leftleg_F(40);
  vTaskDelay(pdMS_TO_TICKS(delay2));
  move_rightleg_B(25);
  vTaskDelay(pdMS_TO_TICKS(delay2));
}

void stop() {
  // Code to stop the robot
}

void setupServo() {
  servorightthigh.attach(right_thigh_pin);
  servorightleg.attach(right_leg_pin);
  servorightthigh.write(right_thigh_pos);
  servorightleg.write(right_leg_pos);

  servoleftthigh.attach(left_thigh_pin);
  servoleftleg.attach(left_leg_pin);
  servoleftthigh.write(left_thigh_pos);
  servoleftleg.write(left_leg_pos);

  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, HIGH);
}

#endif