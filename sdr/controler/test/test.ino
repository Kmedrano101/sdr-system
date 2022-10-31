#include <EasyUltrasonic.h>
#include <ArduinoJson.hpp>
#include <ArduinoJson.h>
#define TRIGA 2
#define ECHOA 3
#define TRIGB 4
#define ECHOB 5
#define MOTOR_A_IN1 8
#define MOTOR_A_IN2 9
#define MOTOR_B_IN1 10
#define MOTOR_B_IN2 11

EasyUltrasonic sonar_A; // Create the ultrasonic object
EasyUltrasonic sonar_B; // Create the ultrasonic object

int S_U1 = 0; // Sensor Ultrasonic  Value 1
int S_U2 = 0; // Sensor Ultrasonic  Value 2
byte SU_Values[2];
String data_TX, data_RX;
int data_values[2];
float peso = 0;

String serializeObject(byte s_state1, byte s_state2, float peso)
{
    String json;
    StaticJsonDocument<300> doc;
    doc["S_P1"] = s_state1;
    doc["S_P2"] = s_state2;
    doc["S_W"] = peso;
    serializeJson(doc, json);
    return json;
}
void deserializeObject(String json)
{
    //String json = "{\"text\":\"myText\",\"id\":10,\"status\":true,\"value\":3.14}";
    StaticJsonDocument<300> doc;
    DeserializationError error = deserializeJson(doc, json);
    if (error) { return; }
    data_values[0] = doc["C_P1"];
    data_values[1] = doc["T_P1"];
    Serial.println("Ok");
}
void setup() {
  Serial.begin(9600); // Open the serial port
  pinMode(MOTOR_A_IN1, OUTPUT);
  pinMode(MOTOR_A_IN2, OUTPUT);
  pinMode(MOTOR_B_IN1, OUTPUT);
  pinMode(MOTOR_B_IN2, OUTPUT);
  sonar_A.attach(TRIGA, ECHOA); // Attaches the ultrasonic sensor on the specified pins on the ultrasonic object
  sonar_B.attach(TRIGB, ECHOB); // Attaches the ultrasonic sensor on the specified pins on the ultrasonic object
}

void loop() {
  S_U1 = int(sonar_A.getDistanceCM()); // Read the distance in centimeters
  S_U2 = int(sonar_B.getDistanceCM()); // Read the distance in centimeters
  if (S_U1 <= 15){
    SU_Values[0] = 1;
  }
  else{
    SU_Values[0] = 0;
  }
  if (S_U2 <= 15){
    SU_Values[1] = 1;
  }
  else{
    SU_Values[1] = 0;
  }
  data_TX = serializeObject(SU_Values[0], SU_Values[1], peso);
  if (SU_Values[0] == 1 or SU_Values[1] == 1){
    Serial.println(data_TX);
  }
  while (Serial.available() > 0){
    data_RX = Serial.readStringUntil('\n');
    deserializeObject(data_RX);
  }
  //delay(300);
}
