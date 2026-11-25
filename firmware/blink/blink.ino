const int LED_PIN = 2;  // the blue LED on most ESP32 DevKit boards

void setup() {
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);
  Serial.println("ESP32 is alive!");
}

void loop() {
  digitalWrite(LED_PIN, HIGH);
  Serial.println("LED on");
  delay(500);

  digitalWrite(LED_PIN, LOW);
  Serial.println("LED off");
  delay(500);
}