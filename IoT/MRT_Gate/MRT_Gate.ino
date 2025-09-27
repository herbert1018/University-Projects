// 安裝函式庫：Adafruit SSD1306，RFID_MFRC522v2 ESP32Servo PubSubClient
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <MFRC522v2.h>
#include <MFRC522DriverSPI.h>
#include <MFRC522DriverPinSimple.h>
#include <MFRC522Debug.h>
#include <ESP32Servo.h>
#include <WiFi.h>
#include <PubSubClient.h>
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SS_PIN 5
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
// === WiFi設定 ===
const char* ssid = "草";          // 修改為您的WiFi名稱
const char* password = "wwwwwwwwww";   // 修改為您的WiFi密碼

// === MQTT設定 ===
const char* mqtt_server = "broker.emqx.io";  // 修改為您的MQTT伺服器IP
const int mqtt_port = 1883;
const char* mqtt_user = "";          // MQTT用戶名(如果需要)
const char* mqtt_password = "";  // MQTT密碼(如果需要)
const char* client_id = "ESP32_DoorSystem";

// === MQTT主題 ===
const char* topic_sensor = "access_control/sensor/json";      // 人流感測器資料
const char* topic_intrusion = "access_control/intrusion/json"; // 強闖偵測資料  
const char* topic_door_cmd = "access_control/door_control";    // 門控制命令接收

// === 引腳定義 ===
const int reedPin = 15, greenLed = 27, redLed = 26;
const int trustBtn = 17, doorBtn = 14, buzzerPin = 16, doorServoPin = 2;

// === 狀態變數 ===
bool trust = false, isAlarming = false, doorOpen = false;
bool lastReedState = HIGH, currentReedState = HIGH;
int peopleCount = 0, inCount = 0, outCount = 0;  // 人流計數器

// === 防彈跳相關 ===
int trustBtnState = HIGH, lastTrustBtnState = HIGH;
int doorBtnState = HIGH, lastDoorBtnState = HIGH;
unsigned long lastTrustDebounce = 0, lastDoorDebounce = 0;

// === 警報蜂鳴器 ===
int toneFreqs[2] = {600, 1200}, toneIndex = 0, lastToneIndex = -1;
unsigned long buzzerPhaseStart = 0;

// === MQTT相關 ===
WiFiClient espClient;
PubSubClient client(espClient);
unsigned long lastMqttReconnect = 0;
unsigned long lastSensorReport = 0;
const unsigned long sensorReportInterval = 5000; // 5秒發送一次感測器資料

// === Servo 物件 ===
Servo doorServo;
MFRC522DriverPinSimple ss_pin(SS_PIN);
MFRC522DriverSPI driver{ss_pin};
MFRC522 mfrc522{driver};
MFRC522::MIFARE_Key key;

const byte BLOCK = 1;
byte bufferSize = 18;
byte readBuffer[18];
byte writeBuffer[16];

String currentStation = "O9";  // 車站
byte fareTableSize = 36;
int balance=126;

struct Station {
  String code;
  int position;
};

Station stations[] = {
  {"R3", 1}, {"R4", 2}, {"R4A", 3}, {"R5", 4}, {"R6", 5},
  {"R7", 6}, {"R8", 7}, {"R9", 8}, {"R10", 9}, {"R11", 10},
  {"R12", 11}, {"R13", 12}, {"R14", 13}, {"R15", 14}, {"R16", 15},
  {"R17", 16}, {"R18", 17}, {"R19", 18}, {"R20", 19}, {"R21", 20},
  {"R22", 21}, {"R22A", 22}, {"R23", 23}, {"R24", 24}, {"RK1", 25},
  {"O1", 26}, {"O2", 27}, {"O4", 28}, {"O5", 29}, {"O6", 30},
  {"O7", 31}, {"O8", 32}, {"O9", 33}, {"O10", 34}, {"O11", 35},
  {"O12", 36}
};

void setup() {
  Serial.begin(115200);
  // 初始化引腳
  pinMode(reedPin, INPUT_PULLUP);
  pinMode(trustBtn, INPUT_PULLUP);
  pinMode(doorBtn, INPUT_PULLUP);
  pinMode(greenLed, OUTPUT);
  pinMode(redLed, OUTPUT);
  pinMode(buzzerPin, OUTPUT);
  
  digitalWrite(buzzerPin, LOW);
  digitalWrite(greenLed, trust ? HIGH : LOW);
  digitalWrite(redLed, LOW);
  
  doorServo.attach(doorServoPin);
  delay(300);
  doorServo.write(0);
  
  // 初始化WiFi和MQTT
  setupWiFi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(mqttCallback);
  
  // 獲取初始磁簧開關狀態
  lastReedState = digitalRead(reedPin);
  
  Serial.println("系統初始化完成");
  testDoorFunction();
  
  // 發送初始狀態
  publishSensorData();
  SPI.begin(18, 19, 23, SS_PIN);
  mfrc522.PCD_Init();

  for (byte i = 0; i < 6; i++) key.keyByte[i] = 0xFF;

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED failed.");
    while (true);
  }

  displayMessage("Metro Card Reader", "Initializing...");
  delay(2000);
}

void loop() {
  displayCurrentStation();  // 非刷卡時持續顯示站名
  unsigned long currentTime = millis();
  
  // 確保MQTT連接
  if (!client.connected()) {
    reconnectMQTT();
  }
  client.loop();
  
  // 處理各種功能
  //handleTrustButton(currentTime);
  handleAlarmSystem(currentTime);
  //handleDoorButton(currentTime);
  handleReedSwitch();
  //handlePeopleCount(); // 處理人流計數
  
  // 定期發送感測器資料
  if (currentTime - lastSensorReport > sensorReportInterval) {
    publishSensorData();
    lastSensorReport = currentTime;
  }
  if (!mfrc522.PICC_IsNewCardPresent() || !mfrc522.PICC_ReadCardSerial()) {
    delay(500);
    return;
  }

  Serial.println("Card detected!");
  MFRC522Debug::PrintUID(Serial, mfrc522.uid);
  Serial.println();

  if (!authenticate()) {
    displayMessage("Auth Failed", "Try Again");
    return;
  }

  if (!readBlock(BLOCK, readBuffer)) {
    displayMessage("Read Failed", "");
    return;
  }

  String storedStation = getStationCodeFromData(readBuffer);
  balance = readBalance(readBuffer);

  if (balance <= 0) {
    displayMessage("Insufficient balance","Please top up!");
    playInsufficientBalanceSound();
    //儲值196
    balance = 196;
    writeExitData(balance);
    delay(3000);
    return;
  }
  if (storedStation == "") {
    writeEntryData(currentStation, balance);
    displayMessage("Entry @ " + currentStation, "Balance: " + String(balance));
    inCount++;  // 門關著時有人通過視為進入
    peopleCount++;
    playEntrySound();
  } else {
    int fare = calculateFare(storedStation, currentStation);
    int newBalance = balance - fare;
    writeExitData(newBalance);
    displayMessage("Exit @ " + currentStation, "Fare: " + String(fare) + " Remain: " + String(newBalance));
    outCount++; // 門開著時有人通過視為離開
    peopleCount = max(0, peopleCount - 1);
    playExitSound();
  }
  trust=true;
  toggleDoor();
  delay(3000);
  toggleDoor();
  trust=false;
  mfrc522.PICC_HaltA();
  mfrc522.PCD_StopCrypto1();
}

// ---------------- 自訂函式區 ----------------

void displayMessage(String line1, String line2) {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.println(line1);
  display.println(line2);
  display.display();
}

bool authenticate() {
  return mfrc522.PCD_Authenticate(0x60, BLOCK, &key, &(mfrc522.uid)) == 0;
}

bool readBlock(byte block, byte *buffer) {
  return mfrc522.MIFARE_Read(block, buffer, &bufferSize) == 0;
}

String getStationCodeFromData(byte *data) {
  if (data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x00) return "";
  return String((char)data[0]) + String((char)data[1]) + (data[2] != 0 ? String((char)data[2]) : "");
}

int findStationIndex(String code) {
  for (int i = 0; i < fareTableSize; i++) {
    if (stations[i].code == code) return stations[i].position;
  }
  return -1;
}

int calculateFare(String entry, String exit) {
  int start = findStationIndex(entry);
  int end = findStationIndex(exit);
  if (start == -1 || end == -1) return 60;
  int dist = abs(end - start);
  if (dist <= 2) return 20;
  int extra = ((dist - 2) + 1) / 2;
  return min(20 + extra * 10, 60);
}

void writeEntryData(String stationCode, byte balance) {
  memset(writeBuffer, 0, 16);
  writeBuffer[0] = stationCode[0];
  writeBuffer[1] = stationCode[1];
  writeBuffer[2] = stationCode.length() > 2 ? stationCode[2] : 0;
  writeBuffer[3] = balance;
  if (mfrc522.MIFARE_Write(BLOCK, writeBuffer, 16) == 0) {
    Serial.println("Entry written.");
    if (readBlock(BLOCK, readBuffer)) {
      Serial.print("BLOCK "); Serial.print(BLOCK); Serial.print(" Data: ");
      printBlockData(readBuffer);
    }
  }
}

void writeExitData(byte balance) {
  memset(writeBuffer, 0, 16);
  writeBuffer[3] = balance;
  if (mfrc522.MIFARE_Write(BLOCK, writeBuffer, 16) == 0) {
    Serial.println("Exit written, entry cleared.");
    if (readBlock(BLOCK, readBuffer)) {
      Serial.print("BLOCK "); Serial.print(BLOCK); Serial.print(" Data: ");
      printBlockData(readBuffer);
    }
  }
}

void displayCurrentStation() {
  displayMessage("Current Station:", currentStation);
}

// ➤ 顯示區塊內容（16 bytes 十六進位）
void printBlockData(byte *data) {
  for (int i = 0; i < 16; i++) {
    if (data[i] < 0x10) Serial.print("0");
    Serial.print(data[i], HEX);
    Serial.print(" ");
  }
  Serial.println();
}
int readBalance(byte *data) {
  int balance = data[3];
  if (balance > 196) {
    balance -= 256;  // 將197-255轉換為-59到-1
  }
  return balance;
}
void setupWiFi() {
  delay(10);
  Serial.println();
  Serial.print("連接到WiFi: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.println("WiFi連接成功!");
  Serial.print("IP地址: ");
  Serial.println(WiFi.localIP());
}

void reconnectMQTT() {
  unsigned long currentTime = millis();
  if (currentTime - lastMqttReconnect < 5000) return; // 5秒重連一次
  
  lastMqttReconnect = currentTime;
  
  if (client.connect(client_id, mqtt_user, mqtt_password)) {
    Serial.println("MQTT連接成功");
    // 訂閱門控制命令
    client.subscribe(topic_door_cmd);
    publishSensorData(); // 發送初始感測器資料
  } else {
    Serial.print("MQTT連接失敗, rc=");
    Serial.println(client.state());
  }
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  
  Serial.print("收到MQTT訊息 [");
  Serial.print(topic);
  Serial.print("]: ");
  Serial.println(message);
  
  // 處理門控制命令
  if (String(topic) == topic_door_cmd) {
    if (message == "open") {
      trust=true;
      if (!doorOpen) toggleDoor();
    } else if (message == "close") {
      if (doorOpen) toggleDoor();
      trust=false;
    } else if (message == "emergency_open") {
      // 緊急開門：強制開門並停止警報
      if (isAlarming) {
        isAlarming = false;
        noTone(buzzerPin);
        digitalWrite(redLed, LOW);
        digitalWrite(buzzerPin, LOW);
        Serial.println("🔇 緊急開門：警報停止");
      }
      if (!doorOpen) {
        doorOpen = true;
        doorServo.attach(doorServoPin);
        doorServo.write(90);
        Serial.println("🚪 緊急開門執行");
      }
    }
  }
}

void handleTrustButton(unsigned long currentTime) {
  int reading = digitalRead(trustBtn);
  if (reading != lastTrustBtnState) lastTrustDebounce = currentTime;
  
  if ((currentTime - lastTrustDebounce) > 50 && reading != trustBtnState) {
    trustBtnState = reading;
    if (trustBtnState == LOW) {
      trust = !trust;
      digitalWrite(greenLed, trust ? HIGH : LOW);
      Serial.println(trust ? "✓ 已授權" : "✗ 取消授權");
    }
  }
  lastTrustBtnState = reading;
}

void handleAlarmSystem(unsigned long currentTime) {
  bool shouldAlarm = (!trust && digitalRead(reedPin) == LOW);
  
  if (shouldAlarm) {
    if (!isAlarming) {
      isAlarming = true;
      buzzerPhaseStart = currentTime;
      toneIndex = 0;
      lastToneIndex = -1;
      Serial.println("🚨 警報啟動");
      
      // 發送強闖偵測MQTT訊息
      publishIntrusionAlert(true);
    }
    
    if (currentTime - buzzerPhaseStart >= 300) {
      toneIndex = (toneIndex + 1) % 2;
      buzzerPhaseStart = currentTime;
    }
    
    if (toneIndex != lastToneIndex) {
      tone(buzzerPin, toneFreqs[toneIndex]);
      digitalWrite(redLed, toneIndex ? HIGH : LOW);
      lastToneIndex = toneIndex;
    }
    
  } else if (isAlarming) {
    isAlarming = false;
    noTone(buzzerPin);
    digitalWrite(redLed, LOW);
    digitalWrite(buzzerPin, LOW);
    Serial.println("🔇 警報停止");
    
    // 發送警報停止MQTT訊息  
    publishIntrusionAlert(false);
  }
}

void handleDoorButton(unsigned long currentTime) {
  int reading = digitalRead(doorBtn);
  if (reading != lastDoorBtnState) lastDoorDebounce = currentTime;
  
  if ((currentTime - lastDoorDebounce) > 50 && reading != doorBtnState) {
    doorBtnState = reading;
    if (doorBtnState == LOW) toggleDoor();
  }
  lastDoorBtnState = reading;
}

void handleReedSwitch() {
  currentReedState = digitalRead(reedPin);
  //Serial.println(trust);
  if (currentReedState != lastReedState) {
    Serial.print("🔗 磁簧開關: ");
    Serial.println((currentReedState == LOW) ? "開啟" : "關閉");
    
    lastReedState = currentReedState;
  }
}

void toggleDoor() {
  doorOpen = !doorOpen;
  Serial.print("🚪 門控制: ");
  bool wasAlarming = isAlarming;
  
  if (wasAlarming) {
    noTone(buzzerPin);
    digitalWrite(buzzerPin, LOW);
  }
  
  doorServo.attach(doorServoPin);
  doorServo.write(doorOpen ? 90 : 0);
  Serial.println(doorOpen ? "開啟 (90°)" : "關閉 (0°)");
  
  delay(500);
  
  if (wasAlarming) buzzerPhaseStart = millis();
}

void testDoorFunction() {/*
  Serial.println("🔧 測試門控功能...");
  doorServo.write(45);
  delay(500);
  doorServo.write(0);
  delay(500);
  Serial.println("✅ 門控測試完成");*/
}

void publishMqttMessage(const char* topic, const char* message) {
  if (client.connected()) {
    client.publish(topic, message);
    Serial.print("📤 MQTT發送 [");
    Serial.print(topic);
    Serial.print("]: ");
    Serial.println(message);
  }
}

void handlePeopleCount() {
  // 簡單的人流計數邏輯（基於磁簧開關狀態變化）
  static bool lastCountState = HIGH;
  bool currentCountState = digitalRead(reedPin);
  
  if (currentCountState != lastCountState && currentCountState == LOW) {
    // 檢測到有人通過（門被開啟）
    if (doorOpen) {
      outCount++; // 門開著時有人通過視為離開
      peopleCount = max(0, peopleCount - 1);
    } else {
      inCount++;  // 門關著時有人通過視為進入
      peopleCount++;
    }
    
    Serial.print("📊 人流變化 - 當前:");
    Serial.print(peopleCount);
    Serial.print(" 進入:");
    Serial.print(inCount);
    Serial.print(" 離開:");
    Serial.println(outCount);
  }
  
  lastCountState = currentCountState;
}

void publishSensorData() {
  if (!client.connected()) return;
  
  // 創建人流感測器JSON資料
  String sensorJson = "{";
  sensorJson += "\"count\":" + String(peopleCount) + ",";
  sensorJson += "\"in_count\":" + String(inCount) + ",";
  sensorJson += "\"out_count\":" + String(outCount) + ",";
  sensorJson += "\"timestamp\":\"" + String(millis()) + "\",";
  sensorJson += "\"door_open\":" + String(doorOpen ? "true" : "false") + ",";
  sensorJson += "\"authorized\":" + String(trust ? "true" : "false");
  sensorJson += "}";
  
  publishMqttMessage(topic_sensor, sensorJson.c_str());
}

void publishIntrusionAlert(bool alertStatus) {
  if (!client.connected()) return;
  
  // 創建強闖偵測JSON資料
  String intrusionJson = "{";
  intrusionJson += "\"alert\":" + String(alertStatus ? "true" : "false") + ",";
  intrusionJson += "\"location\":\""+ currentStation + "\",";
  intrusionJson += "\"timestamp\":\"" + String(millis()) + "\",";
  intrusionJson += "\"door_status\":\"" + String(doorOpen ? "open" : "closed") + "\",";
  intrusionJson += "\"authorized\":" + String(trust ? "true" : "false");
  intrusionJson += "}";
  
  publishMqttMessage(topic_intrusion, intrusionJson.c_str());
}
void playEntrySound() {
  // 第一聲
  tone(buzzerPin, 1000, 150);  // 1000Hz, 150ms
  delay(200);
  
  // 第二聲
  tone(buzzerPin, 1200, 150);  // 1200Hz, 150ms
  delay(300);
}

// 出站音效 - 一聲長響
void playExitSound() {
  tone(buzzerPin, 800, 500);   // 800Hz, 500ms
  delay(600);
}
void playInsufficientBalanceSound() {
  tone(buzzerPin, 300, 300);   // 長響
  delay(350);
  tone(buzzerPin, 500, 100);   // 短響
  delay(150);
  tone(buzzerPin, 300, 300);   // 長響
  delay(400);
}