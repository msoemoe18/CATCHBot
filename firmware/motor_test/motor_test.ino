// One-motor test for Cytron MDD10A channel 1
const int PWM_PIN = 25;      // MDD10A PWM1: how fast
const int DIR_PIN = 26;      // MDD10A DIR1: which direction
const int PWM_FREQ = 20000;  // 20 kHz, above human hearing so the motor doesn't whine
const int PWM_BITS = 8;      // speed values from 0 to 255

// speed from -255 (full backward) to 255 (full forward)
void setMotor(int speed) {
  digitalWrite(DIR_PIN, speed >= 0 ? HIGH : LOW);
  ledcWrite(PWM_PIN, abs(speed));
}

void setup() {
  Serial.begin(115200);
  pinMode(DIR_PIN, OUTPUT);
  ledcAttach(PWM_PIN, PWM_FREQ, PWM_BITS);
  setMotor(0);
  Serial.println("Motor test starting in 3 seconds...");
  delay(3000);
}

void loop() {
  Serial.println("Forward, slow");
  setMotor(80);
  delay(2000);

  Serial.println("Stop");
  setMotor(0);
  delay(1000);

  Serial.println("Backward, slow");
  setMotor(-80);
  delay(2000);

  Serial.println("Stop");
  setMotor(0);
  delay(3000);
}