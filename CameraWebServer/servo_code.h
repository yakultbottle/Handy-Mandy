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
Servo servolefthand;
Servo servorightarm;
Servo servorighthand;


int left_arm_pos = 90;
int left_arm_pin = 3;
int left_hand_pos = 90;
int left_hand_pin;


int right_arm_pos = 90;
int right_arm_pin = 1;
int right_hand_pos = 90;
int right_hand_pin;

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
int command = -1;          //0 = forward, 1 = left, 2 = right
int dance_num = 5;

void move_right_forward(int iterations) {
  int i = 0;
  int total_delay = 0;  // accumulate the total delay time
  while (i < iterations) {
    servorightthigh.write(posR);
    servorightleg.write(posR);

    //only move arms when moving forward
    if (command == 0) {
      // Move the arms if total_delay is less than 300ms
      if (total_delay < 300) {
        servoleftarm.write(80);   // rotate left arm backward
        servorightarm.write(80);  // rotate right arm forward
      } else {
        servoleftarm.write(90);   // stop left arm
        servorightarm.write(90);  // stop right arm
      }
    }

    vTaskDelay(pdMS_TO_TICKS(delayTime));
    total_delay += delayTime;
    posR -= 1;
    i += 1;
  }
}

void move_right_neutral(int iterations, int state) {  // only can call from forward
  int i = 0;
  int total_delay = 0;

  if (state == 1) {
    // start: posR = 0(forward)
    while (i < iterations) {
      servorightthigh.write(posR);
      servorightleg.write(posR);

      if (command == 0) {
        // Move the arms back to neutral if total_delay is less than 300ms
        if (total_delay < 300) {
          servoleftarm.write(100);   // moving back to neutral
          servorightarm.write(100);  // moving back to neutral
        } else {
          servoleftarm.write(90);   // stop left arm
          servorightarm.write(90);  // stop right arm
        }
      }

      vTaskDelay(pdMS_TO_TICKS(delayTime));
      total_delay += delayTime;

      posR += 1;
      i += 1;
    }
  } else if (state == 0) {
    // start: posR = 100
    while (i < iterations) {
      servorightthigh.write(posR);
      servorightleg.write(posR);
      vTaskDelay(pdMS_TO_TICKS(delayTime));
      posR -= 1;
      i += 1;
    }
    // end: posR = 50
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
  int total_delay = 0;  // accumulate the total delay time

  while (i < iterations) {
    servoleftthigh.write(posL);
    servoleftleg.write(posL);

    // Move the arms if total_delay is less than 300ms
    if (total_delay < 300) {
      servoleftarm.write(100);   // move left arm to 100
      servorightarm.write(100);  // move right arm to 100
    } else {
      servoleftarm.write(90);   // stop left arm at neutral
      servorightarm.write(90);  // stop right arm at neutral
    }

    vTaskDelay(pdMS_TO_TICKS(delayTime));
    total_delay += delayTime;

    posL += 1;
    i += 1;
  }

  // Ensure the arms are in the neutral position at the end
  servoleftarm.write(90);
  servorightarm.write(90);

  // end: posL = 100
}


void move_left_neutral(int iterations, int state) {  // only can call from forward
  int i = 0;
  int total_delay = 0;  // accumulate the total delay time

  if (state == 1) {
    // start: posL = 100(forward)
    while (i < iterations) {
      servoleftthigh.write(posL);
      servoleftleg.write(posL);

      // Move the arms back to 80 if total_delay is less than 300ms
      if (total_delay < 300) {
        servoleftarm.write(80);   // move left arm forward
        servorightarm.write(80);  // move right arm backward
      } else {
        servoleftarm.write(90);   // stop left arm at neutral
        servorightarm.write(90);  // stop right arm at neutral
      }

      vTaskDelay(pdMS_TO_TICKS(delayTime));
      total_delay += delayTime;

      posL -= 1;
      i += 1;
    }

    // Ensure the arms are in the neutral position at the end
    servoleftarm.write(90);
    servorightarm.write(90);

    // end: posL = 50
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
  command = 0;
  F_loop(num_F);
  command = -1;
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
  command = 1;
  move_rightleg_F(40);
  vTaskDelay(pdMS_TO_TICKS(delay2));
  move_leftleg_B(25);
  vTaskDelay(pdMS_TO_TICKS(delay2));
  command = -1;
}

void right() {
  command = 2;
  move_leftleg_F(40);
  vTaskDelay(pdMS_TO_TICKS(delay2));
  move_rightleg_B(25);
  vTaskDelay(pdMS_TO_TICKS(delay2));
  command = -1;
}

void stop() {
  // Code to stop the robot
}

void dance() {
  command = 3;
  int i = 0;
  while (i < 20) {
    servorightthigh.write(posR);
    servorightleg.write(posR);
    servoleftthigh.write(posL);
    servoleftleg.write(posL);
    vTaskDelay(pdMS_TO_TICKS(delayTime));
    posR -= 1;
    posL += 1;
    i += 1;
  }

  servoleftarm.write(80);  // Start of arm setup
  servorightarm.write(100);
  vTaskDelay(pdMS_TO_TICKS(500));
  servoleftarm.write(90);
  servorightarm.write(90);  // End of arm setup
  vTaskDelay(pdMS_TO_TICKS(50));

  servoleftarm.write(80);  // Left arm up, right arm down
  servorightarm.write(80);
  vTaskDelay(pdMS_TO_TICKS(100));
  servoleftarm.write(90);
  servorightarm.write(90);
  vTaskDelay(pdMS_TO_TICKS(50));

  int j = 0;
  while (j < dance_num) { // Dance!!
    servoleftarm.write(100);  // Left arm down, right arm up
    servorightarm.write(100);
    vTaskDelay(pdMS_TO_TICKS(200));
    servoleftarm.write(90);
    servorightarm.write(90);
    vTaskDelay(pdMS_TO_TICKS(50));
    servoleftarm.write(80);  // Left arm up, right arm down
    servorightarm.write(80);
    vTaskDelay(pdMS_TO_TICKS(200));
    servoleftarm.write(90);
    servorightarm.write(90);
    j += 1;
  }
  // Back to neutral
  vTaskDelay(pdMS_TO_TICKS(300));
  servoleftarm.write(100);
  servorightarm.write(80);
  vTaskDelay(pdMS_TO_TICKS(100));
  servorightarm.write(90);
  vTaskDelay(pdMS_TO_TICKS(350));
  servoleftarm.write(90);

  i = 0;
  while (i < 20) {
    servorightthigh.write(posR);
    servorightleg.write(posR);
    servoleftthigh.write(posL);
    servoleftleg.write(posL);
    vTaskDelay(pdMS_TO_TICKS(delayTime));
    posR += 1;
    posL -= 1;
    i += 1;
  }

  command = -1;
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


  servoleftarm.attach(left_arm_pin);
  servolefthand.attach(left_hand_pin);
  servoleftarm.write(left_arm_pos);
  servolefthand.write(left_hand_pos);

  servorightarm.attach(right_arm_pin);
  servorighthand.attach(right_hand_pin);
  servorightarm.write(right_arm_pos);
  servorighthand.write(right_hand_pos);

  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, HIGH);
}

#endif