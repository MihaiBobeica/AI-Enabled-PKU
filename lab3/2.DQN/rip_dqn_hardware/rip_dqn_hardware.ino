/*
 * Rotary inverted pendulum: MATLAB-aligned DQN deployment firmware
 *
 * - 200 Hz control on STM32
 * - DQN architecture 6 -> 64 -> 64 -> 10, ReLU
 * - model selected in the Python panel and uploaded to STM32 RAM
 * - full-range swing-up and balance are both produced by the DQN
 * - difference+LPF state estimate at large angles (training-compatible)
 * - fixed discrete Luenberger observer initialized and softly blended near upright
 *
 * Observation order:
 *   sin(theta), cos(theta), theta_dot, sin(alpha), cos(alpha), alpha_dot
 * Angular velocities are raw rad/s values; no additional normalization.
 * Serial protocol version: 7
 */
#include <Arduino.h>
#include <HardwareTimer.h>
#include <math.h>
#include <string.h>

static const int PIN_POT = A0;
static const int PIN_ENC_A = 2;
static const int PIN_ENC_B = 3;
static const int PIN_IN1 = 4;
static const int PIN_IN2 = 5;
static const int PIN_STBY = 7;
static const int PIN_PWM = 10;

static const uint32_t CONTROL_HZ = 200;
static const uint32_t IDLE_TELEMETRY_DIVIDER = 10;
static const float DT = 0.005f;
static const float PI_F = 3.14159265358979323846f;
static const int32_t POT_MOD = 4096;
static const int PROTOCOL_VERSION = 8;

static const int MODEL_INPUT = 6;
static const int MODEL_H1 = 64;
static const int MODEL_H2 = 64;
static const int MODEL_OUTPUT = 10;

static const int MODE_DISABLED = 0;
static const int MODE_DQN_DIFF = 1;
static const int MODE_ESTIMATOR_BLEND = 2;
static const int MODE_DQN_LUENBERGER = 3;

struct ControllerConfig {
  float duration;
  float pwm_limit;
  float observer_enter_deg;
  float observer_exit_deg;
  float observer_blend_alpha;
  float velocity_lpf;
  int32_t pot_up;
  int32_t pot_down;
  float theta_rad_per_count;
  int motor_sign;
};

struct DQNModel {
  float w1[MODEL_H1][MODEL_INPUT];
  float b1[MODEL_H1];
  float w2[MODEL_H2][MODEL_H1];
  float b2[MODEL_H2];
  float w3[MODEL_OUTPUT][MODEL_H2];
  float b3[MODEL_OUTPUT];
  int16_t actions[MODEL_OUTPUT];
};

static const uint32_t MODEL_FLOAT_COUNT =
  MODEL_H1 * MODEL_INPUT + MODEL_H1 + MODEL_H2 * MODEL_H1 +
  MODEL_H2 + MODEL_OUTPUT * MODEL_H2 + MODEL_OUTPUT;
static const uint32_t MODEL_BLOB_BYTES =
  MODEL_FLOAT_COUNT * sizeof(float) + MODEL_OUTPUT * sizeof(int16_t);
static_assert(sizeof(DQNModel) == MODEL_BLOB_BYTES,
              "DQNModel layout must match the Python model blob");

static ControllerConfig g_cfg = {
  30.0f, 150.0f, 20.0f, 35.0f, 0.18f, 0.25f,
  3110, 990, 0.00583730846f, 1
};
static DQNModel g_model;
static bool g_model_loaded = false;
static bool g_model_loading = false;
static bool g_model_actions_received = false;
static uint32_t g_model_token = 0;
static uint32_t g_received_w1 = 0, g_received_b1 = 0, g_received_w2 = 0;
static uint32_t g_received_b2 = 0, g_received_w3 = 0, g_received_b3 = 0;
static char g_model_digest[24] = "none";
static float g_model_pwm_limit = 150.0f;


// Protocol-v8 acknowledged ASCII-hex upload state.  The model is transferred in
// individually acknowledged chunks so the host never overruns the small USB-CDC
// receive buffer used by some ST-Link virtual COM implementations.
static const uint32_t MODEL_CHUNK_MAX_BYTES = 128u;
static bool g_binary_session_active = false;
static bool g_binary_chunk_receiving = false;
static uint32_t g_binary_expected = 0;
static uint32_t g_binary_received = 0;
static uint32_t g_binary_crc_running = 0xFFFFFFFFu;
static uint32_t g_binary_crc_expected = 0;
static uint32_t g_binary_last_activity_ms = 0;
static uint32_t g_binary_chunk_offset = 0;
static uint32_t g_binary_chunk_expected = 0;
static uint32_t g_binary_chunk_received = 0;
static uint32_t g_binary_chunk_crc_expected = 0;
static uint32_t g_binary_chunk_crc_running = 0xFFFFFFFFu;
static uint8_t g_binary_chunk_buffer[MODEL_CHUNK_MAX_BYTES];
static char g_binary_pending_digest[24] = "none";

// Fixed 5 ms discrete Luenberger observer from the prior MPC firmware:
// xhat[k+1] = (Ad-LC)xhat[k] + Bd*u[k] + L*y[k], y=[theta,alpha].
static const float OBS_A_LC[4][4] = {
  { 0.82469568f,  0.00438721f,  -0.31915638f, -0.00085553f},
  {-1.75095359f,  0.94056709f, -0.04181246f,  0.00193564f},
  {-0.05215683f,  0.00005656f,  0.76549687f,  0.00439733f},
  {-1.38405407f,  0.07685667f, -15.83946960f,  0.95027468f}
};
static const float OBS_B_U[4] = {0.00001119f, 0.00396389f, -0.00001398f, -0.00583846f};
static const float OBS_L[4][2] = {
  {0.17530432f, 0.31833008f},
  {1.75095359f, -0.19974721f},
  {0.05215683f, 0.23646539f},
  {1.38405407f, 16.66931771f}
};
static const float OBS_VEL_CLIP = 500.0f;
static const float OBS_RESET_ERR_LIMIT = 35.0f * PI_F / 180.0f;

static volatile int32_t g_encoder = 0;
static volatile int32_t g_encoder_raw = 0;
static volatile int32_t g_encoder_zero = 0;
static volatile int g_pot_raw = 0;
static volatile bool g_tick = false;
static volatile bool g_armed = false;
static volatile bool g_config_valid = false;
static volatile uint32_t g_step = 0;
static volatile uint32_t g_run_id = 0;
static volatile int g_pwm = 0;
static volatile int g_mode = MODE_DISABLED;
static volatile float g_observer_blend = 0.0f;

static float g_prev_theta = 0.0f;
static float g_prev_alpha = 0.0f;
static float g_theta_dot_lpf = 0.0f;
static float g_alpha_dot_lpf = 0.0f;
static bool g_velocity_ready = false;
static bool g_observer_active = false;
static bool g_observer_ready = false;
static float g_xhat[4] = {0, 0, 0, 0};
static float g_previous_policy_pwm = 0.0f;
static float g_h1[MODEL_H1];
static float g_h2[MODEL_H2];
static float g_q[MODEL_OUTPUT];
static int g_last_action_index = -1;
static float g_last_qmax = 0.0f;

static uint32_t g_idle_divider = 0;
static char g_rx[768];
static size_t g_rx_len = 0;
static HardwareTimer* g_timer = nullptr;

static inline float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}
static inline uint32_t crc32UpdateByte(uint32_t crc, uint8_t value) {
  crc ^= (uint32_t)value;
  for (uint8_t bit = 0; bit < 8; ++bit) {
    uint32_t mask = (uint32_t)-(int32_t)(crc & 1u);
    crc = (crc >> 1) ^ (0xEDB88320u & mask);
  }
  return crc;
}
static inline float wrapToPi(float x) {
  while (x > PI_F) x -= 2.0f * PI_F;
  while (x < -PI_F) x += 2.0f * PI_F;
  return x;
}
static inline bool finite4(const float x[4]) {
  return isfinite(x[0]) && isfinite(x[1]) && isfinite(x[2]) && isfinite(x[3]);
}
static inline int32_t mod4096(int32_t x) {
  x %= POT_MOD; if (x < 0) x += POT_MOD; return x;
}
static inline int32_t cwDistanceDecreasing(int32_t ref, int32_t x) {
  return mod4096(mod4096(ref) - mod4096(x));
}

static float alphaFromPot(int32_t raw, const ControllerConfig& cfg) {
  int32_t up = mod4096(cfg.pot_up), down = mod4096(cfg.pot_down), x = mod4096(raw);
  int32_t upToDown = cwDistanceDecreasing(up, down);
  if (upToDown <= 0 || upToDown >= POT_MOD) return 0.0f;
  int32_t upToX = cwDistanceDecreasing(up, x);
  if (upToX <= upToDown) return PI_F * (float)upToX / (float)upToDown;
  int32_t remainder = POT_MOD - upToDown;
  if (remainder <= 0) return 0.0f;
  return -PI_F * (float)(POT_MOD - upToX) / (float)remainder;
}

static inline void motorStandby(bool standby) { digitalWrite(PIN_STBY, standby ? LOW : HIGH); }
static inline void motorBrake() {
  digitalWrite(PIN_IN1, HIGH); digitalWrite(PIN_IN2, HIGH); analogWrite(PIN_PWM, 0); motorStandby(true);
}
static inline void motorDrive(int pwm) {
  pwm = constrain(pwm, -255, 255); motorStandby(false);
  if (pwm == 0) { digitalWrite(PIN_IN1, LOW); digitalWrite(PIN_IN2, LOW); analogWrite(PIN_PWM, 0); }
  else if (pwm > 0) { digitalWrite(PIN_IN1, HIGH); digitalWrite(PIN_IN2, LOW); analogWrite(PIN_PWM, pwm); }
  else { digitalWrite(PIN_IN1, LOW); digitalWrite(PIN_IN2, HIGH); analogWrite(PIN_PWM, -pwm); }
}

void encoderA() {
  bool a = digitalRead(PIN_ENC_A), b = digitalRead(PIN_ENC_B);
  if (a) g_encoder += b ? 1 : -1; else g_encoder += b ? -1 : 1;
}
void encoderB() {
  bool a = digitalRead(PIN_ENC_A), b = digitalRead(PIN_ENC_B);
  if (b) g_encoder += a ? -1 : 1; else g_encoder += a ? 1 : -1;
}
void controlTick() {
  g_pot_raw = analogRead(PIN_POT); g_encoder_raw = g_encoder; g_tick = true;
}

static int splitCsv(const String& line, String fields[], int capacity) {
  int count = 0, start = 0, length = line.length();
  for (int i = 0; i <= length && count < capacity; ++i) {
    if (i == length || line.charAt(i) == ',') {
      fields[count++] = line.substring(start, i); start = i + 1;
    }
  }
  return count;
}

static bool validateConfig(const ControllerConfig& c) {
  if (!(c.duration >= 0.1f && c.duration <= 600.0f)) return false;
  if (!(c.pwm_limit >= 1.0f && c.pwm_limit <= 255.0f)) return false;
  if (!(c.observer_enter_deg > 0.0f && c.observer_enter_deg < c.observer_exit_deg && c.observer_exit_deg < 180.0f)) return false;
  if (!(c.observer_blend_alpha > 0.0f && c.observer_blend_alpha <= 1.0f)) return false;
  if (!(c.velocity_lpf > 0.0f && c.velocity_lpf <= 1.0f)) return false;
  if (!(c.pot_up >= 0 && c.pot_up < POT_MOD && c.pot_down >= 0 && c.pot_down < POT_MOD && c.pot_up != c.pot_down)) return false;
  if (!(fabsf(c.theta_rad_per_count) > 1.0e-8f && fabsf(c.theta_rad_per_count) < 1.0f)) return false;
  if (!(c.motor_sign == 1 || c.motor_sign == -1)) return false;
  return true;
}

static void observerReset() {
  for (int i = 0; i < 4; ++i) g_xhat[i] = 0.0f;
  g_observer_ready = false; g_observer_active = false;
}
static void observerInitialize(float theta, float thetaDot, float alpha, float alphaDot) {
  g_xhat[0] = theta; g_xhat[1] = thetaDot; g_xhat[2] = wrapToPi(alpha); g_xhat[3] = alphaDot; g_observer_ready = true;
}
static void observerUpdate(float previousInput, float theta, float alpha) {
  if (!g_observer_ready) { observerInitialize(theta, g_theta_dot_lpf, alpha, g_alpha_dot_lpf); return; }
  float nextState[4];
  for (int i = 0; i < 4; ++i) {
    float value = OBS_B_U[i] * previousInput + OBS_L[i][0] * theta + OBS_L[i][1] * alpha;
    for (int j = 0; j < 4; ++j) value += OBS_A_LC[i][j] * g_xhat[j];
    nextState[i] = value;
  }
  if (!finite4(nextState) || fabsf(wrapToPi(nextState[2] - alpha)) > OBS_RESET_ERR_LIMIT) {
    observerInitialize(theta, g_theta_dot_lpf, alpha, g_alpha_dot_lpf); return;
  }
  nextState[1] = clampf(nextState[1], -OBS_VEL_CLIP, OBS_VEL_CLIP);
  nextState[3] = clampf(nextState[3], -OBS_VEL_CLIP, OBS_VEL_CLIP);
  nextState[2] = wrapToPi(nextState[2]);
  for (int i = 0; i < 4; ++i) g_xhat[i] = nextState[i];
}

static bool dqnForward(const float obs[MODEL_INPUT], int* actionIndex, float* qmax) {
  if (!g_model_loaded) return false;
  for (int i = 0; i < MODEL_H1; ++i) {
    float z = g_model.b1[i];
    for (int j = 0; j < MODEL_INPUT; ++j) z += g_model.w1[i][j] * obs[j];
    g_h1[i] = z > 0.0f ? z : 0.0f;
  }
  for (int i = 0; i < MODEL_H2; ++i) {
    float z = g_model.b2[i];
    for (int j = 0; j < MODEL_H1; ++j) z += g_model.w2[i][j] * g_h1[j];
    g_h2[i] = z > 0.0f ? z : 0.0f;
  }
  int best = 0;
  for (int i = 0; i < MODEL_OUTPUT; ++i) {
    float z = g_model.b3[i];
    for (int j = 0; j < MODEL_H2; ++j) z += g_model.w3[i][j] * g_h2[j];
    g_q[i] = z;
    if (i == 0 || z > g_q[best]) best = i;
  }
  if (!isfinite(g_q[best])) return false;
  if (actionIndex) *actionIndex = best;
  if (qmax) *qmax = g_q[best];
  return true;
}

static float* modelArray(const String& name, uint32_t* length) {
  if (name == "W1") { *length = MODEL_H1 * MODEL_INPUT; return &g_model.w1[0][0]; }
  if (name == "B1") { *length = MODEL_H1; return &g_model.b1[0]; }
  if (name == "W2") { *length = MODEL_H2 * MODEL_H1; return &g_model.w2[0][0]; }
  if (name == "B2") { *length = MODEL_H2; return &g_model.b2[0]; }
  if (name == "W3") { *length = MODEL_OUTPUT * MODEL_H2; return &g_model.w3[0][0]; }
  if (name == "B3") { *length = MODEL_OUTPUT; return &g_model.b3[0]; }
  *length = 0; return nullptr;
}
static uint32_t* modelCounter(const String& name) {
  if (name == "W1") return &g_received_w1;
  if (name == "B1") return &g_received_b1;
  if (name == "W2") return &g_received_w2;
  if (name == "B2") return &g_received_b2;
  if (name == "W3") return &g_received_w3;
  if (name == "B3") return &g_received_b3;
  return nullptr;
}
static bool modelComplete() {
  return g_model_actions_received &&
    g_received_w1 == (uint32_t)(MODEL_H1 * MODEL_INPUT) && g_received_b1 == MODEL_H1 &&
    g_received_w2 == (uint32_t)(MODEL_H2 * MODEL_H1) && g_received_b2 == MODEL_H2 &&
    g_received_w3 == (uint32_t)(MODEL_OUTPUT * MODEL_H2) && g_received_b3 == MODEL_OUTPUT;
}


static bool binaryModelValuesValid() {
  const float* values = reinterpret_cast<const float*>(&g_model);
  for (uint32_t i = 0; i < MODEL_FLOAT_COUNT; ++i) {
    if (!isfinite(values[i])) return false;
  }
  for (int i = 0; i < MODEL_OUTPUT; ++i) {
    if (g_model.actions[i] < -255 || g_model.actions[i] > 255) return false;
  }
  return true;
}

static void printHex8(uint32_t value) {
  if (value < 0x10000000u) Serial.print("0");
  if (value < 0x01000000u) Serial.print("0");
  if (value < 0x00100000u) Serial.print("0");
  if (value < 0x00010000u) Serial.print("0");
  if (value < 0x00001000u) Serial.print("0");
  if (value < 0x00000100u) Serial.print("0");
  if (value < 0x00000010u) Serial.print("0");
  Serial.print((unsigned long)value, HEX);
}

static void clearBinarySession(bool invalidateModel) {
  g_binary_session_active = false;
  g_binary_chunk_receiving = false;
  g_binary_expected = 0;
  g_binary_received = 0;
  g_binary_chunk_offset = 0;
  g_binary_chunk_expected = 0;
  g_binary_chunk_received = 0;
  g_model_loading = false;
  if (invalidateModel) g_model_loaded = false;
}

static void abortBinaryUpload(const char* reason) {
  clearBinarySession(true);
  Serial.print("ERR,MODEL_");
  Serial.println(reason);
}

static uint32_t crc32OfBytes(const uint8_t* data, uint32_t length) {
  uint32_t crc = 0xFFFFFFFFu;
  for (uint32_t i = 0; i < length; ++i) crc = crc32UpdateByte(crc, data[i]);
  return crc ^ 0xFFFFFFFFu;
}

static int hexNibble(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
  if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
  return -1;
}

static bool decodeHexBytes(const String& text, uint8_t* out, uint32_t length) {
  if ((uint32_t)text.length() != 2u * length) return false;
  for (uint32_t i = 0; i < length; ++i) {
    int hi = hexNibble(text.charAt((int)(2u * i)));
    int lo = hexNibble(text.charAt((int)(2u * i + 1u)));
    if (hi < 0 || lo < 0) return false;
    out[i] = (uint8_t)((hi << 4) | lo);
  }
  return true;
}

static void acknowledgeHexChunk(
    uint32_t offset, uint32_t endOffset, uint32_t length, uint32_t chunkCrc) {
  Serial.print("ACK,MODEL_HEX_CHUNK,");
  Serial.print(g_model_token);
  Serial.print(",");
  Serial.print(offset);
  Serial.print(",");
  Serial.print(endOffset);
  Serial.print(",");
  Serial.print(length);
  Serial.print(",");
  printHex8(chunkCrc);
  Serial.println();
}


static void finishBinaryUpload(uint32_t endCrc) {
  if (!g_binary_session_active) {
    Serial.println("ERR,MODEL_HEX_END_STATE");
    return;
  }
  uint32_t finalCrc = g_binary_crc_running ^ 0xFFFFFFFFu;
  if (g_binary_received != g_binary_expected ||
      g_binary_received != MODEL_BLOB_BYTES) {
    Serial.print("ERR,MODEL_HEX_SIZE,");
    Serial.print(g_model_token);
    Serial.print(",");
    Serial.println(g_binary_received);
    return;
  }
  if (finalCrc != g_binary_crc_expected || finalCrc != endCrc) {
    abortBinaryUpload("HEX_CRC");
    return;
  }
  if (!binaryModelValuesValid()) {
    abortBinaryUpload("HEX_VALUES");
    return;
  }

  strncpy(g_model_digest, g_binary_pending_digest, sizeof(g_model_digest) - 1);
  g_model_digest[sizeof(g_model_digest) - 1] = '\0';
  g_model_loading = false;
  g_model_loaded = true;
  g_binary_session_active = false;
  Serial.print("ACK,MODEL_HEX_DONE,");
  Serial.print(g_model_token);
  Serial.print(",");
  Serial.print(g_model_digest);
  Serial.print(",");
  printHex8(finalCrc);
  Serial.println();
}

static void stopControl() {
  noInterrupts(); g_armed = false; g_pwm = 0; g_mode = MODE_DISABLED; g_observer_blend = 0.0f; interrupts();
  g_previous_policy_pwm = 0.0f; g_last_action_index = -1; g_last_qmax = 0.0f; observerReset(); motorBrake();
}
static void startControl(uint32_t runId) {
  if (!g_config_valid) { Serial.println("ERR,CONFIG_REQUIRED"); return; }
  if (!g_model_loaded) { Serial.println("ERR,MODEL_REQUIRED"); return; }
  if (g_armed) {
    if (runId == g_run_id) { Serial.print("ARMED,"); Serial.println(g_run_id); }
    else Serial.println("ERR,ALREADY_RUNNING");
    return;
  }
  noInterrupts();
  g_encoder_zero = g_encoder; g_step = 0; g_run_id = runId; g_pwm = 0; g_mode = MODE_DQN_DIFF; g_observer_blend = 0.0f; g_armed = true;
  interrupts();
  g_prev_theta = 0.0f; g_prev_alpha = 0.0f; g_theta_dot_lpf = 0.0f; g_alpha_dot_lpf = 0.0f; g_velocity_ready = false;
  g_previous_policy_pwm = 0.0f; observerReset(); motorStandby(false);
  Serial.print("ARMED,"); Serial.println(runId);
}

static void printStatus() {
  ControllerConfig c; noInterrupts(); c = g_cfg; bool armed = g_armed; bool valid = g_config_valid; interrupts();
  Serial.print("STATUS,"); Serial.print(valid ? 1 : 0); Serial.print(","); Serial.print(g_model_loaded ? 1 : 0); Serial.print(","); Serial.print(armed ? 1 : 0);
  Serial.print(","); Serial.print(c.pot_up); Serial.print(","); Serial.print(c.pot_down); Serial.print(","); Serial.print(c.theta_rad_per_count, 10);
  Serial.print(","); Serial.print(c.motor_sign); Serial.print(","); Serial.println(g_model_digest);
}

static void handleModelHexBegin(const String& line) {
  String f[10];
  int n = splitCsv(line, f, 10);
  if (n != 7 || f[1].toInt() != PROTOCOL_VERSION || g_armed) {
    Serial.println("ERR,MODEL_HEX_BEGIN");
    return;
  }
  uint32_t token = (uint32_t)f[2].toInt();
  uint32_t bytes = (uint32_t)f[3].toInt();
  float pwmLimit = f[4].toFloat();
  uint32_t expectedCrc = (uint32_t)strtoul(f[6].c_str(), nullptr, 16);
  if (bytes != MODEL_BLOB_BYTES ||
      !(pwmLimit >= 1.0f && pwmLimit <= 255.0f)) {
    Serial.println("ERR,MODEL_HEX_FORMAT");
    return;
  }

  stopControl();
  clearBinarySession(true);
  memset(&g_model, 0, sizeof(g_model));
  g_model_token = token;
  g_model_pwm_limit = pwmLimit;
  g_model_loading = true;
  g_model_loaded = false;
  g_model_actions_received = false;
  g_binary_expected = bytes;
  g_binary_received = 0;
  g_binary_crc_running = 0xFFFFFFFFu;
  g_binary_crc_expected = expectedCrc;
  g_binary_last_activity_ms = millis();
  f[5].toCharArray(g_binary_pending_digest,
                   sizeof(g_binary_pending_digest));
  g_binary_session_active = true;
  Serial.print("ACK,MODEL_HEX_READY,");
  Serial.print(token);
  Serial.print(",");
  Serial.println(bytes);
}

static void handleModelHexChunk(const String& line) {
  String f[9];
  int n = splitCsv(line, f, 9);
  if (n != 7 || f[1].toInt() != PROTOCOL_VERSION ||
      !g_binary_session_active || !g_model_loading ||
      (uint32_t)f[2].toInt() != g_model_token) {
    Serial.println("ERR,MODEL_HEX_CHUNK_STATE");
    return;
  }

  uint32_t offset = (uint32_t)f[3].toInt();
  uint32_t length = (uint32_t)f[4].toInt();
  uint32_t expectedCrc = (uint32_t)strtoul(f[5].c_str(), nullptr, 16);
  if (length == 0 || length > MODEL_CHUNK_MAX_BYTES ||
      offset + length > g_binary_expected ||
      (uint32_t)f[6].length() != 2u * length) {
    Serial.println("ERR,MODEL_HEX_CHUNK_FORMAT");
    return;
  }

  // Idempotent retry after a lost ACK.
  if (offset < g_binary_received && offset + length <= g_binary_received) {
    const uint8_t* modelBytes = reinterpret_cast<const uint8_t*>(&g_model);
    uint32_t storedCrc = crc32OfBytes(modelBytes + offset, length);
    if (storedCrc == expectedCrc) {
      acknowledgeHexChunk(offset, offset + length, length, storedCrc);
    } else {
      Serial.println("ERR,MODEL_HEX_DUPLICATE_CRC");
    }
    g_binary_last_activity_ms = millis();
    return;
  }

  if (offset != g_binary_received) {
    Serial.print("ERR,MODEL_HEX_OFFSET,");
    Serial.print(g_model_token);
    Serial.print(",");
    Serial.println(g_binary_received);
    return;
  }

  if (!decodeHexBytes(f[6], g_binary_chunk_buffer, length)) {
    Serial.println("ERR,MODEL_HEX_DECODE");
    return;
  }
  uint32_t actualCrc = crc32OfBytes(g_binary_chunk_buffer, length);
  if (actualCrc != expectedCrc) {
    Serial.print("ERR,MODEL_HEX_CHUNK_CRC,");
    Serial.print(g_model_token);
    Serial.print(",");
    Serial.println(offset);
    return;
  }

  uint8_t* destination = reinterpret_cast<uint8_t*>(&g_model);
  memcpy(destination + offset, g_binary_chunk_buffer, length);
  for (uint32_t i = 0; i < length; ++i) {
    g_binary_crc_running = crc32UpdateByte(
        g_binary_crc_running, g_binary_chunk_buffer[i]);
  }
  g_binary_received += length;
  g_binary_last_activity_ms = millis();
  acknowledgeHexChunk(offset, g_binary_received, length, actualCrc);
}

static void handleModelHexEnd(const String& line) {
  String f[6];
  int n = splitCsv(line, f, 6);
  if (n != 4 || f[1].toInt() != PROTOCOL_VERSION ||
      (uint32_t)f[2].toInt() != g_model_token) {
    Serial.println("ERR,MODEL_HEX_END");
    return;
  }
  uint32_t endCrc = (uint32_t)strtoul(f[3].c_str(), nullptr, 16);
  finishBinaryUpload(endCrc);
}

static void handleModelHexAbort(const String& line) {
  String f[5];
  int n = splitCsv(line, f, 5);
  uint32_t token = n >= 3 ? (uint32_t)f[2].toInt() : g_model_token;
  clearBinarySession(true);
  Serial.print("ACK,MODEL_HEX_ABORT,");
  Serial.println(token);
}


static void handleModelBegin(const String& line) {
  String f[12]; int n = splitCsv(line, f, 12);
  if (n != 8 || f[1].toInt() != PROTOCOL_VERSION || g_armed) { Serial.println("ERR,MODEL_BEGIN"); return; }
  uint32_t token = (uint32_t)f[2].toInt();
  if (f[3].toInt()!=MODEL_INPUT || f[4].toInt()!=MODEL_H1 || f[5].toInt()!=MODEL_H2 || f[6].toInt()!=MODEL_OUTPUT) { Serial.println("ERR,MODEL_DIMENSIONS"); return; }
  memset(&g_model, 0, sizeof(g_model));
  g_model_token=token; g_model_loading=true; g_model_loaded=false; g_model_actions_received=false; g_model_pwm_limit=f[7].toFloat();
  g_received_w1=g_received_b1=g_received_w2=g_received_b2=g_received_w3=g_received_b3=0;
  strcpy(g_model_digest,"loading");
  Serial.print("ACK,MODEL_BEGIN,"); Serial.println(token);
}
static void handleModelActions(const String& line) {
  String f[16]; int n=splitCsv(line,f,16);
  if (n != 2 + MODEL_OUTPUT || !g_model_loading || (uint32_t)f[1].toInt()!=g_model_token) { Serial.println("ERR,MODEL_ACTIONS"); return; }
  for (int i=0;i<MODEL_OUTPUT;++i) g_model.actions[i]=(int16_t)f[i+2].toInt();
  g_model_actions_received=true; Serial.print("ACK,MODEL_ACTIONS,"); Serial.println(g_model_token);
}
static void handleModelSet(const String& line) {
  String f[32]; int n=splitCsv(line,f,32);
  if (n < 5 || !g_model_loading || (uint32_t)f[1].toInt()!=g_model_token) { Serial.println("ERR,MODEL_SET"); return; }
  String name=f[2]; uint32_t start=(uint32_t)f[3].toInt(), length=0; float* array=modelArray(name,&length); uint32_t* counter=modelCounter(name);
  if (!array || !counter || start>=length) { Serial.println("ERR,MODEL_ARRAY"); return; }
  uint32_t index=start;
  for (int i=4;i<n && index<length;++i,++index) array[index]=f[i].toFloat();
  if (index>*counter) *counter=index;
  Serial.print("ACK,MODEL_SET,"); Serial.print(g_model_token); Serial.print(","); Serial.print(name); Serial.print(","); Serial.print(start); Serial.print(","); Serial.println(index);
}
static void handleModelEnd(const String& line) {
  String f[5]; int n=splitCsv(line,f,5);
  if (n < 3 || !g_model_loading || (uint32_t)f[1].toInt()!=g_model_token) { Serial.println("ERR,MODEL_END"); return; }
  if (!modelComplete()) { Serial.println("ERR,MODEL_INCOMPLETE"); return; }
  String digest=f[2]; digest.toCharArray(g_model_digest,sizeof(g_model_digest));
  g_model_loading=false; g_model_loaded=true;
  Serial.print("ACK,MODEL_END,"); Serial.print(g_model_token); Serial.print(","); Serial.println(g_model_digest);
}

static void applyConfig(const String& line) {
  String f[16]; int n=splitCsv(line,f,16);
  if (n != 13 || f[1].toInt()!=PROTOCOL_VERSION) { Serial.println("ERR,CONFIG_FORMAT"); return; }
  uint32_t token=(uint32_t)f[2].toInt(); ControllerConfig c;
  c.duration=f[3].toFloat(); c.pwm_limit=f[4].toFloat(); c.observer_enter_deg=f[5].toFloat(); c.observer_exit_deg=f[6].toFloat();
  c.observer_blend_alpha=f[7].toFloat(); c.velocity_lpf=f[8].toFloat(); c.pot_up=f[9].toInt(); c.pot_down=f[10].toInt(); c.theta_rad_per_count=f[11].toFloat(); c.motor_sign=f[12].toInt();
  if (!validateConfig(c)) { Serial.println("ERR,CONFIG_VALUE"); return; }
  if (g_armed) stopControl(); noInterrupts(); g_cfg=c; g_config_valid=true; interrupts();
  Serial.print("ACK,CONFIG,"); Serial.println(token);
}

static void handleCommand(const String& raw) {
  String line=raw; line.trim(); if (line.length()==0) return;
  if (line=="HELLO") { Serial.println("READY,RIP_DQN_HW,8"); return; }
  if (line=="STATUS") { printStatus(); return; }
  if (line.startsWith("MODEL_HEX_BEGIN,")) { handleModelHexBegin(line); return; }
  if (line.startsWith("MODEL_HEX_CHUNK,")) { handleModelHexChunk(line); return; }
  if (line.startsWith("MODEL_HEX_END,")) { handleModelHexEnd(line); return; }
  if (line.startsWith("MODEL_HEX_ABORT,")) { handleModelHexAbort(line); return; }
  if (line.startsWith("MODEL_BEGIN,")) { handleModelBegin(line); return; }
  if (line.startsWith("MODEL_ACTIONS,")) { handleModelActions(line); return; }
  if (line.startsWith("MODEL_SET,")) { handleModelSet(line); return; }
  if (line.startsWith("MODEL_END,")) { handleModelEnd(line); return; }
  if (line.startsWith("CONFIG,")) { applyConfig(line); return; }
  if (line.startsWith("GO,")) { startControl((uint32_t)line.substring(3).toInt()); return; }
  if (line=="STOP" || line.startsWith("STOP,")) { uint32_t id=g_run_id; stopControl(); Serial.print("STOPPED,"); Serial.println(id); return; }
  if (line=="CALUP") {
    if (g_armed) { Serial.println("ERR,CAL_WHILE_RUNNING"); return; }
    int rawPot=g_pot_raw; noInterrupts(); g_cfg.pot_up=rawPot; interrupts(); Serial.print("CAL,UP,"); Serial.println(rawPot); return;
  }
  if (line=="CALDOWN") {
    if (g_armed) { Serial.println("ERR,CAL_WHILE_RUNNING"); return; }
    int rawPot=g_pot_raw; noInterrupts(); g_cfg.pot_down=rawPot; interrupts(); Serial.print("CAL,DOWN,"); Serial.println(rawPot); return;
  }
  Serial.println("ERR,UNKNOWN_COMMAND");
}

static void readSerialInput() {
  while (Serial.available() > 0) {
    int raw = Serial.read();
    if (raw < 0) break;
    char c = (char)raw;
    if (c == '\n') {
      g_rx[g_rx_len] = '\0';
      handleCommand(String(g_rx));
      g_rx_len = 0;
    } else if (c != '\r') {
      if (g_rx_len + 1 < sizeof(g_rx)) {
        g_rx[g_rx_len++] = c;
      } else {
        g_rx_len = 0;
        Serial.println("ERR,LINE_TOO_LONG");
      }
    }
  }

  uint32_t now = millis();
  if (g_binary_session_active &&
      (uint32_t)(now - g_binary_last_activity_ms) > 30000u) {
    abortBinaryUpload("HEX_SESSION_TIMEOUT");
  }
}


static void printTelemetry(uint32_t runId,uint32_t step,float theta,float thetaDot,float alpha,float alphaDot,int pwm,int mode,float blend,int pot,int32_t enc,int action,float qmax) {
  Serial.print("TEL,"); Serial.print(runId); Serial.print(","); Serial.print(step); Serial.print(","); Serial.print(theta,7); Serial.print(","); Serial.print(thetaDot,7);
  Serial.print(","); Serial.print(alpha,7); Serial.print(","); Serial.print(alphaDot,7); Serial.print(","); Serial.print(pwm); Serial.print(","); Serial.print(mode);
  Serial.print(","); Serial.print(blend,7); Serial.print(","); Serial.print(pot); Serial.print(","); Serial.print(enc); Serial.print(","); Serial.print(action); Serial.print(","); Serial.println(qmax,7);
}
static void printMonitor(float theta,float thetaDot,float alpha,float alphaDot,int pot,int32_t enc) {
  Serial.print("MON,"); Serial.print(theta,7); Serial.print(","); Serial.print(thetaDot,7); Serial.print(","); Serial.print(alpha,7); Serial.print(","); Serial.print(alphaDot,7); Serial.print(","); Serial.print(pot); Serial.print(","); Serial.println(enc);
}

static void processControlTick() {
  ControllerConfig c; int pot; int32_t enc,encZero; bool armed; uint32_t step,runId;
  noInterrupts(); c=g_cfg; pot=g_pot_raw; enc=g_encoder_raw; encZero=g_encoder_zero; armed=g_armed; step=g_step; runId=g_run_id; interrupts();
  float theta=(float)(enc-encZero)*c.theta_rad_per_count; float alpha=alphaFromPot(pot,c);
  if (g_binary_session_active) {
    motorBrake();
    return;
  }
  if (!g_velocity_ready) {
    g_prev_theta=theta; g_prev_alpha=alpha; g_theta_dot_lpf=0.0f; g_alpha_dot_lpf=0.0f; g_velocity_ready=true;
  } else {
    float rawThetaDot=(theta-g_prev_theta)/DT; float rawAlphaDot=wrapToPi(alpha-g_prev_alpha)/DT;
    g_theta_dot_lpf += c.velocity_lpf*(rawThetaDot-g_theta_dot_lpf); g_alpha_dot_lpf += c.velocity_lpf*(rawAlphaDot-g_alpha_dot_lpf);
    g_prev_theta=theta; g_prev_alpha=alpha;
  }
  if (!armed) {
    motorBrake(); if (++g_idle_divider>=IDLE_TELEMETRY_DIVIDER) { g_idle_divider=0; printMonitor(theta,g_theta_dot_lpf,alpha,g_alpha_dot_lpf,pot,enc); } return;
  }
  float runTime=(float)step*DT;
  if (runTime>=c.duration) { stopControl(); Serial.print("DONE,"); Serial.println(runId); return; }
  float absAlpha=fabsf(alpha), enter=c.observer_enter_deg*PI_F/180.0f, exitAngle=c.observer_exit_deg*PI_F/180.0f;
  if (!g_observer_active && absAlpha<=enter) {
    g_observer_active=true; observerInitialize(theta,g_theta_dot_lpf,alpha,g_alpha_dot_lpf);
  } else if (g_observer_active && absAlpha>=exitAngle) {
    g_observer_active=false; g_observer_ready=false;
  }
  if (g_observer_active) observerUpdate(g_previous_policy_pwm,theta,alpha);
  float target=g_observer_active?1.0f:0.0f;
  g_observer_blend += c.observer_blend_alpha*(target-g_observer_blend); g_observer_blend=clampf(g_observer_blend,0.0f,1.0f);
  float b=g_observer_blend;
  float thetaCtrl=(1.0f-b)*theta+b*g_xhat[0];
  float thetaDotCtrl=(1.0f-b)*g_theta_dot_lpf+b*g_xhat[1];
  float alphaCtrl=wrapToPi(alpha+b*wrapToPi(g_xhat[2]-alpha));
  float alphaDotCtrl=(1.0f-b)*g_alpha_dot_lpf+b*g_xhat[3];
  float obs[MODEL_INPUT]={sinf(thetaCtrl),cosf(thetaCtrl),thetaDotCtrl,sinf(alphaCtrl),cosf(alphaCtrl),alphaDotCtrl};
  int actionIndex=-1; float qmax=0.0f;
  if (!dqnForward(obs,&actionIndex,&qmax)) { stopControl(); Serial.println("ERR,MODEL_INFERENCE"); return; }
  float command=(float)g_model.actions[actionIndex];
  float limit=fminf(c.pwm_limit,g_model_pwm_limit>0.0f?g_model_pwm_limit:c.pwm_limit); command=clampf(command,-limit,limit);
  int policyPwm=(int)lroundf(command); g_previous_policy_pwm=(float)policyPwm;
  int pwmApplied=c.motor_sign*policyPwm; pwmApplied=constrain(pwmApplied,-(int)c.pwm_limit,(int)c.pwm_limit); motorDrive(pwmApplied);
  int mode=(b<=0.01f)?MODE_DQN_DIFF:((b>=0.99f)?MODE_DQN_LUENBERGER:MODE_ESTIMATOR_BLEND);
  noInterrupts(); g_step=step+1; g_pwm=pwmApplied; g_mode=mode; interrupts();
  g_last_action_index=actionIndex; g_last_qmax=qmax;
  printTelemetry(runId,step,theta,thetaDotCtrl,alpha,alphaDotCtrl,pwmApplied,mode,b,pot,enc,actionIndex,qmax);
}

void setup() {
  Serial.begin(921600);
  pinMode(PIN_POT,INPUT); pinMode(PIN_ENC_A,INPUT); pinMode(PIN_ENC_B,INPUT); pinMode(PIN_IN1,OUTPUT); pinMode(PIN_IN2,OUTPUT); pinMode(PIN_STBY,OUTPUT); pinMode(PIN_PWM,OUTPUT);
  analogReadResolution(12); analogWriteResolution(8); motorBrake(); attachInterrupt(digitalPinToInterrupt(PIN_ENC_A),encoderA,CHANGE); attachInterrupt(digitalPinToInterrupt(PIN_ENC_B),encoderB,CHANGE);
  g_timer=new HardwareTimer(TIM2); g_timer->setOverflow(CONTROL_HZ,HERTZ_FORMAT); g_timer->attachInterrupt(controlTick); g_timer->setInterruptPriority(1,0); g_timer->resume();
  Serial.println("READY,RIP_DQN_HW,8");
}
void loop() {
  readSerialInput(); if (!g_tick) return; noInterrupts(); g_tick=false; interrupts(); processControlTick();
}
