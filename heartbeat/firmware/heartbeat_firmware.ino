// Building Heartbeat Monitor - firmware (Adafruit Feather ESP32-S2)
// Streams 3-axis acceleration (in g) from three floors as CSV over USB serial.
//   Floor 1: ADXL345      @ 0x53
//   Floor 2: ICM-20948    @ 0x69
//   Roof   : LSM6DSOX     @ 0x6A  (Arduino Modulino Movement)
// Talks to the chips directly over I2C (register maps from the datasheets), so no extra libraries.
#include <Wire.h>

#define ADXL_ADDR 0x53
#define ICM_ADDR  0x69
#define LSM_ADDR  0x6A

const uint32_t PERIOD_US = 5000;   // 200 samples per second

bool writeReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

uint8_t readReg(uint8_t addr, uint8_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0xFF;
  if (Wire.requestFrom((uint16_t)addr, (size_t)1, true) != 1) return 0xFF;
  return Wire.read();
}

bool readBytes(uint8_t addr, uint8_t reg, uint8_t *buf, uint8_t n) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint16_t)addr, (size_t)n, true) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

// ---------- ADXL345 (floor 1) ----------
bool adxlInit() {
  if (readReg(ADXL_ADDR, 0x00) != 0xE5) return false;   // DEVID
  writeReg(ADXL_ADDR, 0x2C, 0x0C);                      // 400 Hz output rate
  writeReg(ADXL_ADDR, 0x31, 0x09);                      // full resolution, +/-4 g
  writeReg(ADXL_ADDR, 0x2D, 0x08);                      // measurement mode
  return true;
}
bool adxlRead(float *g) {
  uint8_t b[6];
  if (!readBytes(ADXL_ADDR, 0x32, b, 6)) return false;
  for (int i = 0; i < 3; i++) g[i] = (int16_t)(b[2*i] | (b[2*i+1] << 8)) * 0.0039f;
  return true;
}

// ---------- ICM-20948 (floor 2) ----------
void icmBank(uint8_t bank) { writeReg(ICM_ADDR, 0x7F, bank << 4); }
bool icmInit() {
  icmBank(0);
  writeReg(ICM_ADDR, 0x06, 0x80);        // reset
  delay(100);
  writeReg(ICM_ADDR, 0x06, 0x01);        // wake, auto clock
  delay(50);
  writeReg(ICM_ADDR, 0x07, 0x00);        // accel + gyro on
  if (readReg(ICM_ADDR, 0x00) != 0xEA) return false;   // WHO_AM_I
  icmBank(2);
  writeReg(ICM_ADDR, 0x10, 0x00);        // sample divider high
  writeReg(ICM_ADDR, 0x11, 0x04);        // divider 4 -> about 225 Hz
  writeReg(ICM_ADDR, 0x14, 0x03);        // +/-4 g, low-pass filter on
  icmBank(0);
  return true;
}
bool icmRead(float *g) {
  uint8_t b[6];
  if (!readBytes(ICM_ADDR, 0x2D, b, 6)) return false;
  for (int i = 0; i < 3; i++) g[i] = (int16_t)((b[2*i] << 8) | b[2*i+1]) / 8192.0f;
  return true;
}

// ---------- LSM6DSOX (roof) ----------
bool lsmInit() {
  uint8_t id = readReg(LSM_ADDR, 0x0F);
  if (id != 0x6C) return false;          // WHO_AM_I for LSM6DSOX
  writeReg(LSM_ADDR, 0x12, 0x01);        // software reset
  delay(50);
  writeReg(LSM_ADDR, 0x12, 0x44);        // block data update + auto-increment
  writeReg(LSM_ADDR, 0x10, 0x68);        // accel 416 Hz, +/-4 g
  return true;
}
bool lsmRead(float *g) {
  uint8_t b[6];
  if (!readBytes(LSM_ADDR, 0x28, b, 6)) return false;
  for (int i = 0; i < 3; i++) g[i] = (int16_t)(b[2*i] | (b[2*i+1] << 8)) * 0.000122f;
  return true;
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);

#ifdef NEOPIXEL_I2C_POWER
  pinMode(NEOPIXEL_I2C_POWER, OUTPUT);
  digitalWrite(NEOPIXEL_I2C_POWER, HIGH);   // powers the STEMMA QT port
#endif
  delay(100);
  Wire.begin();
  Wire.setClock(400000);

  Serial.println(adxlInit() ? "# floor1 ADXL345 ok"   : "# floor1 ADXL345 FAILED");
  Serial.println(icmInit()  ? "# floor2 ICM-20948 ok" : "# floor2 ICM-20948 FAILED");
  Serial.println(lsmInit()  ? "# roof LSM6DSOX ok"    : "# roof LSM6DSOX FAILED");
  Serial.println("t_us,f1x,f1y,f1z,f2x,f2y,f2z,f3x,f3y,f3z,errors");
}

void loop() {
  static uint32_t next = micros();
  static float a[3] = {0}, b[3] = {0}, c[3] = {0};
  static uint32_t errors = 0;

  while ((int32_t)(micros() - next) < 0) { }   // wait for the next 5 ms tick
  next += PERIOD_US;
  uint32_t t = micros();

  if (!adxlRead(a)) errors++;
  if (!icmRead(b))  errors++;
  if (!lsmRead(c))  errors++;

  Serial.printf("%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%lu\n",
                (unsigned long)t, a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2],
                (unsigned long)errors);
}
