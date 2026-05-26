#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <MPU9250_asukiaaa.h>
#include <NimBLEDevice.h>
#include "esp_wifi.h"

#define SDA_PIN 21
#define SCL_PIN 19

// =======================
// WiFi тохиргоо
// =======================
const char* ssid = "SICT-WIFI";
const char* password = "Sict@wirelessNET";

// Laptop-ийн IP ба UDP port
const char* udpAddress = "172.16.158.210";
const int udpPort = 5005;

// =======================
// Sensor тохиргоо
// =======================
#define SAMPLE_INTERVAL_MS 10       // 10ms = 100Hz
#define SENSOR_QUEUE_SIZE 200       // 2 секунд орчим buffer

// =======================
// BLE Advertising тохиргоо
// =======================
const char* bleDeviceName = "ESP32_MPU9250";

// Advertising interval нь 0.625ms unit-тэй.
// 160 = 100ms, 320 = 200ms
#define BLE_ADV_MIN_INTERVAL 160
#define BLE_ADV_MAX_INTERVAL 320

WiFiUDP udp;
MPU9250_asukiaaa mySensor;

struct SensorPacket {
  uint32_t t;
  float ax;
  float ay;
  float az;
  float gx;
  float gy;
  float gz;
};

QueueHandle_t sensorQueue;

volatile uint32_t droppedSamples = 0;

// =======================
// WiFi setup
// =======================
void connectWiFi() {
  WiFi.mode(WIFI_STA);

  // Real-time UDP stream-д WiFi sleep унтраах нь чухал
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);

  WiFi.begin(ssid, password);

  Serial.print("WiFi holbogdoj baina");

  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi holbogdloo");
  Serial.print("ESP32 WiFi IP: ");
  Serial.println(WiFi.localIP());

  udp.begin(0);
}

// =======================
// Sensor setup
// =======================
void setupSensor() {
  Wire.begin(SDA_PIN, SCL_PIN);

  // 100kHz биш 400kHz бол sensor унших delay багасна
  Wire.setClock(400000);

  mySensor.setWire(&Wire);

  mySensor.beginAccel();
  mySensor.beginGyro();

  delay(500);

  Serial.println("MPU9250 ready");
}

// =======================
// BLE advertising setup
// =======================
void setupBLEAdvertising() {
  NimBLEDevice::init(bleDeviceName);

  // TX power тохируулах
  // ESP_PWR_LVL_N0, ESP_PWR_LVL_P3, ESP_PWR_LVL_P6, ESP_PWR_LVL_P9 гэх мэт
  NimBLEDevice::setPower(ESP_PWR_LVL_P3);

  NimBLEAdvertising* advertising = NimBLEDevice::getAdvertising();

  advertising->reset();

  // Laptop scan хийхэд нэр нь харагдана
  advertising->setName(bleDeviceName);

  // Зөвхөн scan-д харагдана, холбогдох шаардлагагүй
  advertising->setConnectableMode(BLE_GAP_CONN_MODE_NON);

  // General discoverable mode
  advertising->setDiscoverableMode(BLE_GAP_DISC_MODE_GEN);

  // Scan response хэрэггүй. Laptop RSSI/MAC-г advertisement packet-оос уншина.
  advertising->enableScanResponse(false);

  // Advertising interval
  advertising->setMinInterval(BLE_ADV_MIN_INTERVAL);
  advertising->setMaxInterval(BLE_ADV_MAX_INTERVAL);

  // TX power-г advertisement packet-д нэмнэ
  advertising->addTxPower();

  bool ok = advertising->start(0);   // 0 = forever

  if (ok) {
    Serial.println("BLE advertising started");
    Serial.print("BLE name: ");
    Serial.println(bleDeviceName);
  } else {
    Serial.println("BLE advertising failed");
  }
}

// =======================
// Sensor унших task
// =======================
void sensorTask(void* parameter) {
  TickType_t lastWakeTime = xTaskGetTickCount();
  const TickType_t interval = pdMS_TO_TICKS(SAMPLE_INTERVAL_MS);

  SensorPacket packet;

  while (true) {
    mySensor.accelUpdate();
    mySensor.gyroUpdate();

    packet.t = millis();
    packet.ax = mySensor.accelX();
    packet.ay = mySensor.accelY();
    packet.az = mySensor.accelZ();
    packet.gx = mySensor.gyroX();
    packet.gy = mySensor.gyroY();
    packet.gz = mySensor.gyroZ();

    if (xQueueSend(sensorQueue, &packet, 0) != pdTRUE) {
      // Queue дүүрсэн бол хамгийн хуучин sample-г хаяад шинэ sample хийж байна
      SensorPacket oldPacket;
      xQueueReceive(sensorQueue, &oldPacket, 0);

      if (xQueueSend(sensorQueue, &packet, 0) != pdTRUE) {
        droppedSamples++;
      } else {
        droppedSamples++;
      }
    }

    vTaskDelayUntil(&lastWakeTime, interval);
  }
}

// =======================
// UDP stream task
// =======================
void wifiStreamTask(void* parameter) {
  SensorPacket packet;
  char payload[180];

  uint32_t printCounter = 0;
  uint32_t lastReconnectTry = 0;

  while (true) {
    if (xQueueReceive(sensorQueue, &packet, portMAX_DELAY) == pdTRUE) {
      int len = snprintf(
        payload,
        sizeof(payload),
        "%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
        (unsigned long)packet.t,
        packet.ax,
        packet.ay,
        packet.az,
        packet.gx,
        packet.gy,
        packet.gz
      );

      if (WiFi.status() == WL_CONNECTED) {
        udp.beginPacket(udpAddress, udpPort);
        udp.write((const uint8_t*)payload, len);
        udp.endPacket();
      } else {
        uint32_t now = millis();

        if (now - lastReconnectTry > 3000) {
          lastReconnectTry = now;
          Serial.println("WiFi disconnected. Reconnecting...");
          WiFi.disconnect();
          WiFi.begin(ssid, password);
        }
      }

      printCounter++;

      // Serial хэвлэлтийг sample бүр дээр хийхгүй
      if (printCounter >= 50) {
        printCounter = 0;

        Serial.print("UDP: ");
        Serial.print(payload);

        Serial.print(" | Queue: ");
        Serial.print(uxQueueMessagesWaiting(sensorQueue));

        Serial.print(" | Dropped: ");
        Serial.println(droppedSamples);
      }
    }
  }
}

// =======================
// setup
// =======================
void setup() {
  Serial.begin(115200);
  delay(2000);

  setupSensor();
  connectWiFi();
  setupBLEAdvertising();

  sensorQueue = xQueueCreate(SENSOR_QUEUE_SIZE, sizeof(SensorPacket));

  if (sensorQueue == NULL) {
    Serial.println("Sensor queue uusgej chadsangui!");
    while (true) {
      delay(1000);
    }
  }

  // Sensor task: Core 1, high priority
  xTaskCreatePinnedToCore(
    sensorTask,
    "SensorTask",
    4096,
    NULL,
    4,
    NULL,
    1
  );

  // UDP stream task: Core 1
  xTaskCreatePinnedToCore(
    wifiStreamTask,
    "WiFiStreamTask",
    4096,
    NULL,
    3,
    NULL,
    1
  );

  Serial.println("All tasks started");
}

// =======================
// loop
// =======================
void loop() {
  // BLE advertising background дээр явна
  // Sensor болон UDP stream task дээр явна
  vTaskDelay(pdMS_TO_TICKS(1000));
}