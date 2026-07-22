/*
 * Rotary inverted pendulum hybrid controller
 *
 * Far from upright: MPC-style phase/energy-pumping swing-up.
 * Near upright: distilled compact7 PPO balance actor, 7 -> 64 -> 64 -> 1, Tanh.
 * The Python panel uploads the original float32 student actor and the exact
 * observation-normalization statistics extracted from ppo_model_weights.h.
 * Control, sensing and telemetry run at 200 Hz.
 *
 * Serial protocol version: 11
 */
#include <Arduino.h>
#include <HardwareTimer.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

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
static const int PROTOCOL_VERSION = 11;

static const int MODEL_INPUT = 7;
static const int MODEL_H1 = 64;
static const int MODEL_H2 = 64;
static const int MODEL_OUTPUT = 1;

static const int MODE_DISABLED = 0;
static const int MODE_ENERGY_SWING = 1;
static const int MODE_BLEND = 2;
static const int MODE_PPO_BALANCE = 3;

static const float THETA_LIMIT = 12.0f * PI_F;
static const float THETA_DOT_LIMIT = 45.0f;
static const float ALPHA_DOT_LIMIT = 40.0f;

struct ControllerConfig {
  float duration;
  float safety_pwm_limit;
  float swing_pwm;
  float kick_time;
  float ppo_enter_deg;
  float ppo_exit_deg;
  float blend_alpha;
  float velocity_lpf;
  int32_t pot_up;
  int32_t pot_down;
  float theta_rad_per_count;
  int motor_sign;
};

struct PPOFloatModel {
  float w0[MODEL_H1][MODEL_INPUT];
  float b0[MODEL_H1];
  float w1[MODEL_H2][MODEL_H1];
  float b1[MODEL_H2];
  float w2[MODEL_OUTPUT][MODEL_H2];
  float b2[MODEL_OUTPUT];
  float obs_mean[MODEL_INPUT];
  float obs_inv_std[MODEL_INPUT];
  float clip_obs;
  float model_pwm_scale;
};

static const uint32_t MODEL_BLOB_BYTES =
    (MODEL_H1 * MODEL_INPUT + MODEL_H1 +
     MODEL_H2 * MODEL_H1 + MODEL_H2 +
     MODEL_OUTPUT * MODEL_H2 + MODEL_OUTPUT +
     MODEL_INPUT + MODEL_INPUT + 1 + 1) * sizeof(float);
static_assert(sizeof(PPOFloatModel) == MODEL_BLOB_BYTES,
              "PPOFloatModel layout must match Python float_blob");

static ControllerConfig g_cfg = {
  30.0f, 150.0f, 120.0f, 0.10f,
  15.0f, 25.0f, 0.18f, 0.25f,
  3110, 990, 0.00583730846f, 1
};
static PPOFloatModel g_model;
static bool g_model_loaded = false;
static bool g_model_loading = false;
static uint32_t g_model_token = 0;
static char g_model_digest[24] = "none";

// Acknowledged ASCII-hex upload state.
static const uint32_t MODEL_CHUNK_MAX_BYTES = 128u;
static bool g_binary_session_active = false;
static uint32_t g_binary_expected = 0;
static uint32_t g_binary_received = 0;
static uint32_t g_binary_crc_running = 0xFFFFFFFFu;
static uint32_t g_binary_crc_expected = 0;
static uint32_t g_binary_last_activity_ms = 0;
static uint8_t g_binary_chunk_buffer[MODEL_CHUNK_MAX_BYTES];
static char g_binary_pending_digest[24] = "none";

// Fixed 5 ms Luenberger observer from the supplied MPC firmware.
static const float OBS_A_LC[4][4] = {
  { 0.82469568f,  0.00438721f,  -0.31915638f, -0.00085553f},
  {-1.75095359f,  0.94056709f, -0.04181246f,  0.00193564f},
  {-0.05215683f,  0.00005656f,  0.76549687f,  0.00439733f},
  {-1.38405407f,  0.07685667f, -15.83946960f,  0.95027468f}
};
static const float OBS_B_U[4] = {
  0.00001119f, 0.00396389f, -0.00001398f, -0.00583846f
};
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
static volatile float g_blend = 0.0f;

static float g_prev_theta = 0.0f;
static float g_prev_alpha = 0.0f;
static float g_theta_unwrapped = 0.0f;
static float g_theta_dot_lpf = 0.0f;
static float g_alpha_dot_lpf = 0.0f;
static bool g_velocity_ready = false;

static bool g_ppo_active = false;
static bool g_observer_ready = false;
static float g_xhat[4] = {0, 0, 0, 0};
static float g_previous_controller_pwm = 0.0f;

static float g_h1[MODEL_H1];
static float g_h2[MODEL_H2];
static float g_last_action_norm = 0.0f;
static float g_last_policy_raw = 0.0f;

static uint32_t g_idle_divider = 0;
static char g_rx[768];
static size_t g_rx_len = 0;
static HardwareTimer* g_timer = nullptr;

static inline float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static inline float wrapToPi(float x) {
  while (x > PI_F) x -= 2.0f * PI_F;
  while (x < -PI_F) x += 2.0f * PI_F;
  return x;
}

static inline bool finite4(const float x[4]) {
  return isfinite(x[0]) && isfinite(x[1]) && isfinite(x[2]) && isfinite(x[3]);
}

static inline uint32_t crc32UpdateByte(uint32_t crc, uint8_t value) {
  crc ^= (uint32_t)value;
  for (uint8_t bit = 0; bit < 8; ++bit) {
    uint32_t mask = (uint32_t)-(int32_t)(crc & 1u);
    crc = (crc >> 1) ^ (0xEDB88320u & mask);
  }
  return crc;
}

static inline int32_t mod4096(int32_t x) {
  x %= POT_MOD;
  if (x < 0) x += POT_MOD;
  return x;
}

static inline int32_t cwDistanceDecreasing(int32_t ref, int32_t x) {
  return mod4096(mod4096(ref) - mod4096(x));
}

static float alphaFromPot(int32_t raw, const ControllerConfig& cfg) {
  int32_t up = mod4096(cfg.pot_up);
  int32_t down = mod4096(cfg.pot_down);
  int32_t x = mod4096(raw);
  int32_t upToDown = cwDistanceDecreasing(up, down);
  if (upToDown <= 0 || upToDown >= POT_MOD) return 0.0f;
  int32_t upToX = cwDistanceDecreasing(up, x);
  if (upToX <= upToDown) {
    return PI_F * (float)upToX / (float)upToDown;
  }
  int32_t remainder = POT_MOD - upToDown;
  if (remainder <= 0) return 0.0f;
  return -PI_F * (float)(POT_MOD - upToX) / (float)remainder;
}

static inline void motorStandby(bool standby) {
  digitalWrite(PIN_STBY, standby ? LOW : HIGH);
}

static inline void motorBrake() {
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, HIGH);
  analogWrite(PIN_PWM, 0);
  motorStandby(true);
}

static inline void motorDrive(int pwm) {
  pwm = constrain(pwm, -255, 255);
  motorStandby(false);
  if (pwm == 0) {
    digitalWrite(PIN_IN1, LOW);
    digitalWrite(PIN_IN2, LOW);
    analogWrite(PIN_PWM, 0);
  } else if (pwm > 0) {
    digitalWrite(PIN_IN1, HIGH);
    digitalWrite(PIN_IN2, LOW);
    analogWrite(PIN_PWM, pwm);
  } else {
    digitalWrite(PIN_IN1, LOW);
    digitalWrite(PIN_IN2, HIGH);
    analogWrite(PIN_PWM, -pwm);
  }
}

void encoderA() {
  bool a = digitalRead(PIN_ENC_A);
  bool b = digitalRead(PIN_ENC_B);
  if (a) g_encoder += b ? 1 : -1;
  else g_encoder += b ? -1 : 1;
}

void encoderB() {
  bool a = digitalRead(PIN_ENC_A);
  bool b = digitalRead(PIN_ENC_B);
  if (b) g_encoder += a ? -1 : 1;
  else g_encoder += a ? 1 : -1;
}

void controlTick() {
  g_pot_raw = analogRead(PIN_POT);
  g_encoder_raw = g_encoder;
  g_tick = true;
}

static int splitCsv(const String& line, String fields[], int capacity) {
  int count = 0;
  int start = 0;
  int length = line.length();
  for (int i = 0; i <= length && count < capacity; ++i) {
    if (i == length || line.charAt(i) == ',') {
      fields[count++] = line.substring(start, i);
      start = i + 1;
    }
  }
  return count;
}

static bool validateConfig(const ControllerConfig& c) {
  if (!(c.duration >= 0.1f && c.duration <= 600.0f)) return false;
  if (!(c.safety_pwm_limit >= 1.0f && c.safety_pwm_limit <= 255.0f)) return false;
  if (!(c.swing_pwm >= 0.0f && c.swing_pwm <= c.safety_pwm_limit)) return false;
  if (!(c.kick_time >= 0.0f && c.kick_time <= 10.0f)) return false;
  if (!(c.ppo_enter_deg > 0.0f && c.ppo_enter_deg < c.ppo_exit_deg && c.ppo_exit_deg < 180.0f)) return false;
  if (!(c.blend_alpha > 0.0f && c.blend_alpha <= 1.0f)) return false;
  if (!(c.velocity_lpf > 0.0f && c.velocity_lpf <= 1.0f)) return false;
  if (!(c.pot_up >= 0 && c.pot_up < POT_MOD && c.pot_down >= 0 && c.pot_down < POT_MOD && c.pot_up != c.pot_down)) return false;
  if (!(fabsf(c.theta_rad_per_count) > 1.0e-8f && fabsf(c.theta_rad_per_count) < 1.0f)) return false;
  if (!(c.motor_sign == 1 || c.motor_sign == -1)) return false;
  return true;
}

static void observerReset() {
  for (int i = 0; i < 4; ++i) g_xhat[i] = 0.0f;
  g_observer_ready = false;
}

static void observerInitialize(float theta, float alpha) {
  g_xhat[0] = theta;
  g_xhat[1] = g_theta_dot_lpf;
  g_xhat[2] = wrapToPi(alpha);
  g_xhat[3] = g_alpha_dot_lpf;
  g_observer_ready = true;
}

static void observerUpdate(float previousInput, float theta, float alpha) {
  if (!g_observer_ready) {
    observerInitialize(theta, alpha);
    return;
  }
  float nextState[4];
  for (int i = 0; i < 4; ++i) {
    float value = OBS_B_U[i] * previousInput + OBS_L[i][0] * theta + OBS_L[i][1] * alpha;
    for (int j = 0; j < 4; ++j) value += OBS_A_LC[i][j] * g_xhat[j];
    nextState[i] = value;
  }
  if (!finite4(nextState) || fabsf(wrapToPi(nextState[2] - alpha)) > OBS_RESET_ERR_LIMIT) {
    observerInitialize(theta, alpha);
    return;
  }
  nextState[1] = clampf(nextState[1], -OBS_VEL_CLIP, OBS_VEL_CLIP);
  nextState[3] = clampf(nextState[3], -OBS_VEL_CLIP, OBS_VEL_CLIP);
  nextState[2] = wrapToPi(nextState[2]);
  for (int i = 0; i < 4; ++i) g_xhat[i] = nextState[i];
}

static void buildCompactObservation(const float state[4], float out[MODEL_INPUT]) {
  out[0] = sinf(state[0]);
  out[1] = cosf(state[0]);
  out[2] = clampf(state[1], -THETA_DOT_LIMIT, THETA_DOT_LIMIT);
  out[3] = sinf(state[2]);
  out[4] = cosf(state[2]);
  out[5] = clampf(state[3], -ALPHA_DOT_LIMIT, ALPHA_DOT_LIMIT);
  out[6] = clampf(g_previous_controller_pwm / g_model.model_pwm_scale, -1.0f, 1.0f);
}

static bool ppoForward(const float rawObs[MODEL_INPUT], float* actionNorm, float* policyRaw) {
  if (!g_model_loaded) return false;
  float x[MODEL_INPUT];
  for (int j = 0; j < MODEL_INPUT; ++j) {
    float z = (rawObs[j] - g_model.obs_mean[j]) * g_model.obs_inv_std[j];
    x[j] = clampf(z, -g_model.clip_obs, g_model.clip_obs);
  }
  for (int i = 0; i < MODEL_H1; ++i) {
    float acc = g_model.b0[i];
    for (int j = 0; j < MODEL_INPUT; ++j) acc += g_model.w0[i][j] * x[j];
    g_h1[i] = tanhf(acc);
  }
  for (int i = 0; i < MODEL_H2; ++i) {
    float acc = g_model.b1[i];
    for (int j = 0; j < MODEL_H1; ++j) acc += g_model.w1[i][j] * g_h1[j];
    g_h2[i] = tanhf(acc);
  }
  float raw = g_model.b2[0];
  for (int j = 0; j < MODEL_H2; ++j) raw += g_model.w2[0][j] * g_h2[j];
  if (!isfinite(raw)) return false;
  float action = tanhf(raw);
  if (policyRaw) *policyRaw = raw;
  if (actionNorm) *actionNorm = clampf(action, -1.0f, 1.0f);
  return true;
}

static bool modelValuesValid() {
  if (!(isfinite(g_model.clip_obs) && g_model.clip_obs > 0.0f && g_model.clip_obs <= 1.0e6f)) return false;
  if (!(isfinite(g_model.model_pwm_scale) && g_model.model_pwm_scale >= 1.0f && g_model.model_pwm_scale <= 255.0f)) return false;
  for (int i = 0; i < MODEL_H1; ++i) {
    if (!isfinite(g_model.b0[i])) return false;
    for (int j = 0; j < MODEL_INPUT; ++j) if (!isfinite(g_model.w0[i][j])) return false;
  }
  for (int i = 0; i < MODEL_H2; ++i) {
    if (!isfinite(g_model.b1[i])) return false;
    for (int j = 0; j < MODEL_H1; ++j) if (!isfinite(g_model.w1[i][j])) return false;
  }
  if (!isfinite(g_model.b2[0])) return false;
  for (int j = 0; j < MODEL_H2; ++j) if (!isfinite(g_model.w2[0][j])) return false;
  for (int i = 0; i < MODEL_INPUT; ++i) {
    if (!(isfinite(g_model.obs_mean[i]) && isfinite(g_model.obs_inv_std[i]) && g_model.obs_inv_std[i] > 0.0f)) return false;
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

static void clearBinarySession(bool invalidateModel) {
  g_binary_session_active = false;
  g_binary_expected = 0;
  g_binary_received = 0;
  g_model_loading = false;
  if (invalidateModel) g_model_loaded = false;
}

static void abortBinaryUpload(const char* reason) {
  clearBinarySession(true);
  Serial.print("ERR,MODEL_");
  Serial.println(reason);
}

static void acknowledgeHexChunk(uint32_t offset, uint32_t endOffset, uint32_t length, uint32_t chunkCrc) {
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

static void stopControl() {
  noInterrupts();
  g_armed = false;
  g_pwm = 0;
  g_mode = MODE_DISABLED;
  g_blend = 0.0f;
  interrupts();
  g_previous_controller_pwm = 0.0f;
  g_last_action_norm = 0.0f;
  g_last_policy_raw = 0.0f;
  g_ppo_active = false;
  observerReset();
  motorBrake();
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
  g_encoder_zero = g_encoder;
  g_step = 0;
  g_run_id = runId;
  g_pwm = 0;
  g_mode = MODE_ENERGY_SWING;
  g_blend = 0.0f;
  g_armed = true;
  interrupts();
  g_prev_theta = 0.0f;
  g_prev_alpha = 0.0f;
  g_theta_unwrapped = 0.0f;
  g_theta_dot_lpf = 0.0f;
  g_alpha_dot_lpf = 0.0f;
  g_velocity_ready = false;
  g_ppo_active = false;
  g_previous_controller_pwm = 0.0f;
  observerReset();
  motorStandby(false);
  Serial.print("ARMED,");
  Serial.println(runId);
}

static void printStatus() {
  ControllerConfig c;
  noInterrupts();
  c = g_cfg;
  bool armed = g_armed;
  bool valid = g_config_valid;
  interrupts();
  Serial.print("STATUS,");
  Serial.print(valid ? 1 : 0);
  Serial.print(",");
  Serial.print(g_model_loaded ? 1 : 0);
  Serial.print(",");
  Serial.print(armed ? 1 : 0);
  Serial.print(",");
  Serial.print(c.pot_up);
  Serial.print(",");
  Serial.print(c.pot_down);
  Serial.print(",");
  Serial.print(c.theta_rad_per_count, 10);
  Serial.print(",");
  Serial.print(c.motor_sign);
  Serial.print(",");
  Serial.println(g_model_digest);
}

static void handleModelHexBegin(const String& line) {
  String f[9];
  int n = splitCsv(line, f, 9);
  if (n != 6 || f[1].toInt() != PROTOCOL_VERSION || g_armed) {
    Serial.println("ERR,MODEL_HEX_BEGIN");
    return;
  }
  uint32_t token = (uint32_t)f[2].toInt();
  uint32_t bytes = (uint32_t)f[3].toInt();
  uint32_t expectedCrc = (uint32_t)strtoul(f[5].c_str(), nullptr, 16);
  if (bytes != MODEL_BLOB_BYTES || f[4].length() < 4) {
    Serial.println("ERR,MODEL_HEX_FORMAT");
    return;
  }
  stopControl();
  clearBinarySession(true);
  memset(&g_model, 0, sizeof(g_model));
  g_model_token = token;
  g_model_loading = true;
  g_model_loaded = false;
  g_binary_expected = bytes;
  g_binary_received = 0;
  g_binary_crc_running = 0xFFFFFFFFu;
  g_binary_crc_expected = expectedCrc;
  g_binary_last_activity_ms = millis();
  f[4].toCharArray(g_binary_pending_digest, sizeof(g_binary_pending_digest));
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
  const uint8_t* modelBytesConst = reinterpret_cast<const uint8_t*>(&g_model);
  if (offset < g_binary_received && offset + length <= g_binary_received) {
    uint32_t storedCrc = crc32OfBytes(modelBytesConst + offset, length);
    if (storedCrc == expectedCrc) acknowledgeHexChunk(offset, offset + length, length, storedCrc);
    else Serial.println("ERR,MODEL_HEX_DUPLICATE_CRC");
    g_binary_last_activity_ms = millis();
    return;
  }
  if (offset != g_binary_received) {
    Serial.print("ERR,MODEL_HEX_OFFSET,");
    Serial.println(g_binary_received);
    return;
  }
  if (!decodeHexBytes(f[6], g_binary_chunk_buffer, length)) {
    Serial.println("ERR,MODEL_HEX_DECODE");
    return;
  }
  uint32_t chunkCrc = crc32OfBytes(g_binary_chunk_buffer, length);
  if (chunkCrc != expectedCrc) {
    Serial.println("ERR,MODEL_HEX_CHUNK_CRC");
    return;
  }
  uint8_t* modelBytes = reinterpret_cast<uint8_t*>(&g_model);
  memcpy(modelBytes + offset, g_binary_chunk_buffer, length);
  for (uint32_t i = 0; i < length; ++i) {
    g_binary_crc_running = crc32UpdateByte(g_binary_crc_running, g_binary_chunk_buffer[i]);
  }
  g_binary_received += length;
  g_binary_last_activity_ms = millis();
  acknowledgeHexChunk(offset, offset + length, length, chunkCrc);
}

static void handleModelHexEnd(const String& line) {
  String f[6];
  int n = splitCsv(line, f, 6);
  if (n != 4 || f[1].toInt() != PROTOCOL_VERSION ||
      !g_binary_session_active || (uint32_t)f[2].toInt() != g_model_token) {
    Serial.println("ERR,MODEL_HEX_END_STATE");
    return;
  }
  uint32_t endCrc = (uint32_t)strtoul(f[3].c_str(), nullptr, 16);
  uint32_t finalCrc = g_binary_crc_running ^ 0xFFFFFFFFu;
  if (g_binary_received != g_binary_expected || g_binary_received != MODEL_BLOB_BYTES) {
    Serial.print("ERR,MODEL_HEX_SIZE,");
    Serial.println(g_binary_received);
    return;
  }
  if (finalCrc != g_binary_crc_expected || finalCrc != endCrc) {
    abortBinaryUpload("HEX_CRC");
    return;
  }
  if (!modelValuesValid()) {
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

static void handleModelHexAbort(const String& line) {
  String f[5];
  int n = splitCsv(line, f, 5);
  if (n < 3 || f[1].toInt() != PROTOCOL_VERSION) {
    Serial.println("ERR,MODEL_HEX_ABORT");
    return;
  }
  clearBinarySession(true);
  Serial.print("ACK,MODEL_HEX_ABORT,");
  Serial.println(f[2]);
}

static void applyConfig(const String& line) {
  String f[18];
  int n = splitCsv(line, f, 18);
  if (n != 15 || f[1].toInt() != PROTOCOL_VERSION) {
    Serial.println("ERR,CONFIG_FORMAT");
    return;
  }
  uint32_t token = (uint32_t)f[2].toInt();
  ControllerConfig c;
  c.duration = f[3].toFloat();
  c.safety_pwm_limit = f[4].toFloat();
  c.swing_pwm = f[5].toFloat();
  c.kick_time = f[6].toFloat();
  c.ppo_enter_deg = f[7].toFloat();
  c.ppo_exit_deg = f[8].toFloat();
  c.blend_alpha = f[9].toFloat();
  c.velocity_lpf = f[10].toFloat();
  c.pot_up = f[11].toInt();
  c.pot_down = f[12].toInt();
  c.theta_rad_per_count = f[13].toFloat();
  c.motor_sign = f[14].toInt();
  if (!validateConfig(c)) {
    Serial.println("ERR,CONFIG_VALUE");
    return;
  }
  if (g_armed) stopControl();
  noInterrupts();
  g_cfg = c;
  g_config_valid = true;
  interrupts();
  Serial.print("ACK,CONFIG,");
  Serial.println(token);
}

static void handleCommand(const String& raw) {
  String line = raw;
  line.trim();
  if (line.length() == 0) return;
  if (line == "HELLO") { Serial.println("READY,RIP_PPO64_HYBRID_HW_STABLE,11"); return; }
  if (line == "STATUS") { printStatus(); return; }
  if (line.startsWith("MODEL_HEX_BEGIN,")) { handleModelHexBegin(line); return; }
  if (line.startsWith("MODEL_HEX_CHUNK,")) { handleModelHexChunk(line); return; }
  if (line.startsWith("MODEL_HEX_END,")) { handleModelHexEnd(line); return; }
  if (line.startsWith("MODEL_HEX_ABORT,")) { handleModelHexAbort(line); return; }
  if (line.startsWith("CONFIG,")) { applyConfig(line); return; }
  if (line.startsWith("GO,")) { startControl((uint32_t)line.substring(3).toInt()); return; }
  if (line == "STOP" || line.startsWith("STOP,")) {
    uint32_t id = g_run_id;
    stopControl();
    Serial.print("STOPPED,");
    Serial.println(id);
    return;
  }
  if (line == "CALUP") {
    if (g_armed) { Serial.println("ERR,CAL_WHILE_RUNNING"); return; }
    int rawPot = g_pot_raw;
    noInterrupts(); g_cfg.pot_up = rawPot; interrupts();
    Serial.print("CAL,UP,"); Serial.println(rawPot);
    return;
  }
  if (line == "CALDOWN") {
    if (g_armed) { Serial.println("ERR,CAL_WHILE_RUNNING"); return; }
    int rawPot = g_pot_raw;
    noInterrupts(); g_cfg.pot_down = rawPot; interrupts();
    Serial.print("CAL,DOWN,"); Serial.println(rawPot);
    return;
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
      if (g_rx_len + 1 < sizeof(g_rx)) g_rx[g_rx_len++] = c;
      else {
        g_rx_len = 0;
        if (g_binary_session_active) abortBinaryUpload("LINE_TOO_LONG");
        else Serial.println("ERR,LINE_TOO_LONG");
      }
    }
  }
  uint32_t now = millis();
  if (g_binary_session_active && (uint32_t)(now - g_binary_last_activity_ms) > 60000u) {
    abortBinaryUpload("HEX_SESSION_TIMEOUT");
  }
}

static void printTelemetry(uint32_t runId, uint32_t step, float theta, float thetaDot,
                           float alpha, float alphaDot, int pwm, int mode, float blend,
                           int pot, int32_t enc, float actionNorm, float policyRaw) {
  Serial.print("TEL,"); Serial.print(runId); Serial.print(","); Serial.print(step);
  Serial.print(","); Serial.print(theta, 7); Serial.print(","); Serial.print(thetaDot, 7);
  Serial.print(","); Serial.print(alpha, 7); Serial.print(","); Serial.print(alphaDot, 7);
  Serial.print(","); Serial.print(pwm); Serial.print(","); Serial.print(mode);
  Serial.print(","); Serial.print(blend, 7); Serial.print(","); Serial.print(pot);
  Serial.print(","); Serial.print(enc); Serial.print(","); Serial.print(actionNorm, 7);
  Serial.print(","); Serial.println(policyRaw, 7);
}

static void printMonitor(float theta, float thetaDot, float alpha, float alphaDot,
                         int pot, int32_t enc) {
  Serial.print("MON,"); Serial.print(theta, 7); Serial.print(","); Serial.print(thetaDot, 7);
  Serial.print(","); Serial.print(alpha, 7); Serial.print(","); Serial.print(alphaDot, 7);
  Serial.print(","); Serial.print(pot); Serial.print(","); Serial.println(enc);
}

static void processControlTick() {
  ControllerConfig c;
  int pot;
  int32_t enc, encZero;
  bool armed;
  uint32_t step, runId;
  noInterrupts();
  c = g_cfg;
  pot = g_pot_raw;
  enc = g_encoder_raw;
  encZero = g_encoder_zero;
  armed = g_armed;
  step = g_step;
  runId = g_run_id;
  interrupts();

  float thetaMeasured = (float)(enc - encZero) * c.theta_rad_per_count;
  float alpha = alphaFromPot(pot, c);
  if (g_binary_session_active) {
    motorBrake();
    return;
  }

  if (!g_velocity_ready) {
    g_prev_theta = thetaMeasured;
    g_theta_unwrapped = thetaMeasured;
    g_prev_alpha = alpha;
    g_theta_dot_lpf = 0.0f;
    g_alpha_dot_lpf = 0.0f;
    g_velocity_ready = true;
  } else {
    float dtheta = wrapToPi(thetaMeasured - g_prev_theta);
    g_theta_unwrapped += dtheta;
    float rawThetaDot = dtheta / DT;
    float rawAlphaDot = wrapToPi(alpha - g_prev_alpha) / DT;
    g_theta_dot_lpf += c.velocity_lpf * (rawThetaDot - g_theta_dot_lpf);
    g_alpha_dot_lpf += c.velocity_lpf * (rawAlphaDot - g_alpha_dot_lpf);
    g_prev_theta = thetaMeasured;
    g_prev_alpha = alpha;
  }

  if (!armed) {
    motorBrake();
    if (++g_idle_divider >= IDLE_TELEMETRY_DIVIDER) {
      g_idle_divider = 0;
      printMonitor(g_theta_unwrapped, g_theta_dot_lpf, alpha, g_alpha_dot_lpf, pot, enc);
    }
    return;
  }

  float runTime = (float)step * DT;
  if (runTime >= c.duration) {
    stopControl();
    Serial.print("DONE,");
    Serial.println(runId);
    return;
  }

  if (!isfinite(g_theta_unwrapped) || !isfinite(alpha) ||
      !isfinite(g_theta_dot_lpf) || !isfinite(g_alpha_dot_lpf) ||
      fabsf(g_theta_unwrapped) > THETA_LIMIT ||
      fabsf(g_theta_dot_lpf) > 2.0f * THETA_DOT_LIMIT ||
      fabsf(g_alpha_dot_lpf) > 2.0f * ALPHA_DOT_LIMIT) {
    stopControl();
    Serial.println("ERR,SAFETY_STATE");
    return;
  }

  float absAlpha = fabsf(wrapToPi(alpha));
  float enter = c.ppo_enter_deg * PI_F / 180.0f;
  float exitAngle = c.ppo_exit_deg * PI_F / 180.0f;
  if (!g_ppo_active && absAlpha <= enter) {
    g_ppo_active = true;
    observerInitialize(g_theta_unwrapped, alpha);
  } else if (g_ppo_active && absAlpha >= exitAngle) {
    g_ppo_active = false;
    observerReset();
  }

  float controlState[4] = {g_theta_unwrapped, g_theta_dot_lpf, alpha, g_alpha_dot_lpf};
  if (g_ppo_active) {
    observerUpdate(g_previous_controller_pwm, g_theta_unwrapped, alpha);
    for (int i = 0; i < 4; ++i) controlState[i] = g_xhat[i];
  }
  controlState[2] = wrapToPi(controlState[2]);

  float compactObs[MODEL_INPUT];
  buildCompactObservation(controlState, compactObs);
  float actionNorm = 0.0f;
  float policyRaw = 0.0f;
  if (!ppoForward(compactObs, &actionNorm, &policyRaw)) {
    stopControl();
    Serial.println("ERR,MODEL_INFERENCE");
    return;
  }
  float policyPwm = actionNorm * g_model.model_pwm_scale;

  float swing;
  if (runTime < c.kick_time) {
    swing = c.swing_pwm;
  } else {
    float phase = g_alpha_dot_lpf * cosf(alpha);
    float direction = phase >= 0.0f ? 1.0f : -1.0f;
    swing = -c.swing_pwm * direction;
  }

  float target = g_ppo_active ? 1.0f : 0.0f;
  float blend = g_blend + c.blend_alpha * (target - g_blend);
  blend = clampf(blend, 0.0f, 1.0f);
  float command = (1.0f - blend) * swing + blend * policyPwm;
  command = clampf(command, -c.safety_pwm_limit, c.safety_pwm_limit);
  g_previous_controller_pwm = command;

  int controllerPwm = (int)lroundf(command);
  int pwmApplied = c.motor_sign * controllerPwm;
  pwmApplied = constrain(pwmApplied, -(int)c.safety_pwm_limit, (int)c.safety_pwm_limit);
  motorDrive(pwmApplied);

  int mode = blend <= 0.01f ? MODE_ENERGY_SWING :
             (blend >= 0.99f ? MODE_PPO_BALANCE : MODE_BLEND);
  noInterrupts();
  g_step = step + 1;
  g_pwm = pwmApplied;
  g_mode = mode;
  g_blend = blend;
  interrupts();
  g_last_action_norm = actionNorm;
  g_last_policy_raw = policyRaw;
  printTelemetry(runId, step, g_theta_unwrapped, controlState[1], alpha,
                 controlState[3], pwmApplied, mode, blend, pot, enc,
                 actionNorm, policyRaw);
}

void setup() {
  Serial.begin(921600);
  pinMode(PIN_POT, INPUT);
  pinMode(PIN_ENC_A, INPUT);
  pinMode(PIN_ENC_B, INPUT);
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_STBY, OUTPUT);
  pinMode(PIN_PWM, OUTPUT);
  analogReadResolution(12);
  analogWriteResolution(8);
  motorBrake();
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_A), encoderA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_B), encoderB, CHANGE);
  g_timer = new HardwareTimer(TIM2);
  g_timer->setOverflow(CONTROL_HZ, HERTZ_FORMAT);
  g_timer->attachInterrupt(controlTick);
  g_timer->setInterruptPriority(1, 0);
  g_timer->resume();
  Serial.println("READY,RIP_PPO64_HYBRID_HW_STABLE,11");
}

void loop() {
  readSerialInput();
  if (!g_tick) return;
  noInterrupts();
  g_tick = false;
  interrupts();
  processControlTick();
}
