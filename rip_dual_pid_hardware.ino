#include <Arduino.h>
#include <HardwareTimer.h>
#include <math.h>

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
static const int MODE_DISABLED = 0;
static const int MODE_SWING = 1;
static const int MODE_BLEND = 2;
static const int MODE_PID = 3;

struct ControllerConfig {
  float duration;
  float kp_alpha;
  float ki_alpha;
  float kd_alpha;
  float kp_theta;
  float ki_theta;
  float kd_theta;
  float pwm_limit;
  float swing_pwm;
  float kick_time;
  float enter_deg;
  float exit_deg;
  float blend_alpha;
  float alpha_i_limit;
  float theta_i_limit;
  float velocity_lpf;
  int32_t pot_up;
  int32_t pot_down;
  float theta_rad_per_count;
  int motor_sign;
};

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
static ControllerConfig g_cfg = {
  10.0f,
  10.0f, 10.0f, 10.0f,
  10.0f, 10.0f, 10.0f,
  150.0f, 120.0f, 0.10f,
  15.0f, 25.0f, 0.18f,
  0.50f, 1.00f, 0.25f,
  3110, 990,
  0.00583730846f, 1
};

static float g_int_alpha = 0.0f;
static float g_int_theta = 0.0f;
static float g_prev_theta = 0.0f;
static float g_prev_alpha = 0.0f;
static float g_theta_dot_lpf = 0.0f;
static float g_alpha_dot_lpf = 0.0f;
static bool g_velocity_ready = false;
static bool g_upright = false;
static uint32_t g_idle_divider = 0;
static char g_rx[512];
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
  int32_t up_to_down = cwDistanceDecreasing(up, down);
  if (up_to_down <= 0 || up_to_down >= POT_MOD) return 0.0f;
  int32_t up_to_x = cwDistanceDecreasing(up, x);
  if (up_to_x <= up_to_down) {
    return PI_F * (float)up_to_x / (float)up_to_down;
  }
  int32_t remainder = POT_MOD - up_to_down;
  if (remainder <= 0) return 0.0f;
  return -PI_F * (float)(POT_MOD - up_to_x) / (float)remainder;
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
  if (!(c.pwm_limit >= 1.0f && c.pwm_limit <= 255.0f)) return false;
  if (!(c.swing_pwm >= 0.0f && c.swing_pwm <= c.pwm_limit)) return false;
  if (!(c.kick_time >= 0.0f && c.kick_time <= 10.0f)) return false;
  if (!(c.enter_deg > 0.0f && c.enter_deg < c.exit_deg && c.exit_deg < 180.0f)) return false;
  if (!(c.blend_alpha > 0.0f && c.blend_alpha <= 1.0f)) return false;
  if (!(c.alpha_i_limit >= 0.0f && c.theta_i_limit >= 0.0f)) return false;
  if (!(c.velocity_lpf > 0.0f && c.velocity_lpf <= 1.0f)) return false;
  if (!(c.pot_up >= 0 && c.pot_up < POT_MOD)) return false;
  if (!(c.pot_down >= 0 && c.pot_down < POT_MOD)) return false;
  if (c.pot_up == c.pot_down) return false;
  if (!(fabsf(c.theta_rad_per_count) > 1.0e-8f && fabsf(c.theta_rad_per_count) < 1.0f)) return false;
  if (!(c.motor_sign == 1 || c.motor_sign == -1)) return false;
  return true;
}

static void stopControl() {
  noInterrupts();
  g_armed = false;
  g_pwm = 0;
  g_mode = MODE_DISABLED;
  g_blend = 0.0f;
  interrupts();
  g_int_alpha = 0.0f;
  g_int_theta = 0.0f;
  g_upright = false;
  motorBrake();
}

static void startControl(uint32_t runId) {
  if (!g_config_valid) {
    Serial.println("ERR,CONFIG_REQUIRED");
    return;
  }
  if (g_armed) {
    if (runId == g_run_id) {
      Serial.print("ARMED,");
      Serial.println(g_run_id);
    } else {
      Serial.println("ERR,ALREADY_RUNNING");
    }
    return;
  }
  noInterrupts();
  g_encoder_zero = g_encoder;
  g_step = 0;
  g_run_id = runId;
  g_pwm = 0;
  g_mode = MODE_SWING;
  g_blend = 0.0f;
  g_armed = true;
  interrupts();
  g_int_alpha = 0.0f;
  g_int_theta = 0.0f;
  g_prev_theta = 0.0f;
  g_prev_alpha = 0.0f;
  g_theta_dot_lpf = 0.0f;
  g_alpha_dot_lpf = 0.0f;
  g_velocity_ready = false;
  g_upright = false;
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
  Serial.print(armed ? 1 : 0);
  Serial.print(",");
  Serial.print(c.pot_up);
  Serial.print(",");
  Serial.print(c.pot_down);
  Serial.print(",");
  Serial.print(c.theta_rad_per_count, 10);
  Serial.print(",");
  Serial.println(c.motor_sign);
}

static void applyConfig(const String& line) {
  String fields[24];
  int count = splitCsv(line, fields, 24);
  if (count != 23 || fields[1].toInt() != 2) {
    Serial.println("ERR,CONFIG_FORMAT");
    return;
  }
  uint32_t token = (uint32_t)fields[2].toInt();
  ControllerConfig c;
  c.duration = fields[3].toFloat();
  c.kp_alpha = fields[4].toFloat();
  c.ki_alpha = fields[5].toFloat();
  c.kd_alpha = fields[6].toFloat();
  c.kp_theta = fields[7].toFloat();
  c.ki_theta = fields[8].toFloat();
  c.kd_theta = fields[9].toFloat();
  c.pwm_limit = fields[10].toFloat();
  c.swing_pwm = fields[11].toFloat();
  c.kick_time = fields[12].toFloat();
  c.enter_deg = fields[13].toFloat();
  c.exit_deg = fields[14].toFloat();
  c.blend_alpha = fields[15].toFloat();
  c.alpha_i_limit = fields[16].toFloat();
  c.theta_i_limit = fields[17].toFloat();
  c.velocity_lpf = fields[18].toFloat();
  c.pot_up = fields[19].toInt();
  c.pot_down = fields[20].toInt();
  c.theta_rad_per_count = fields[21].toFloat();
  c.motor_sign = fields[22].toInt();
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
  if (line == "HELLO") {
    Serial.println("READY,RIP_DUAL_PID_HW,2");
    return;
  }
  if (line == "STATUS") {
    printStatus();
    return;
  }
  if (line.startsWith("GO,")) {
    startControl((uint32_t)line.substring(3).toInt());
    return;
  }
  if (line == "STOP" || line.startsWith("STOP,")) {
    uint32_t runId = g_run_id;
    stopControl();
    Serial.print("STOPPED,");
    Serial.println(runId);
    return;
  }
  if (line == "CALUP") {
    if (g_armed) {
      Serial.println("ERR,CAL_WHILE_RUNNING");
      return;
    }
    int rawPot = g_pot_raw;
    noInterrupts();
    g_cfg.pot_up = rawPot;
    interrupts();
    Serial.print("CAL,UP,");
    Serial.println(rawPot);
    return;
  }
  if (line == "CALDOWN") {
    if (g_armed) {
      Serial.println("ERR,CAL_WHILE_RUNNING");
      return;
    }
    int rawPot = g_pot_raw;
    noInterrupts();
    g_cfg.pot_down = rawPot;
    interrupts();
    Serial.print("CAL,DOWN,");
    Serial.println(rawPot);
    return;
  }
  if (line.startsWith("PREF,") || line.startsWith("PREF=")) {
    int p = line.indexOf(',');
    if (p < 0) p = line.indexOf('=');
    int value = line.substring(p + 1).toInt();
    if (value >= 0 && value < POT_MOD) {
      noInterrupts();
      g_cfg.pot_up = value;
      interrupts();
      Serial.print("CAL,UP,");
      Serial.println(value);
    } else {
      Serial.println("ERR,PREF");
    }
    return;
  }
  if (line.startsWith("PDOWN,") || line.startsWith("PDOWN=")) {
    int p = line.indexOf(',');
    if (p < 0) p = line.indexOf('=');
    int value = line.substring(p + 1).toInt();
    if (value >= 0 && value < POT_MOD) {
      noInterrupts();
      g_cfg.pot_down = value;
      interrupts();
      Serial.print("CAL,DOWN,");
      Serial.println(value);
    } else {
      Serial.println("ERR,PDOWN");
    }
    return;
  }
  if (line.startsWith("CONFIG,")) {
    applyConfig(line);
    return;
  }
  Serial.println("ERR,UNKNOWN_COMMAND");
}

static void readSerialLines() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
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
}

static void printTelemetry(
  uint32_t runId,
  uint32_t step,
  float theta,
  float thetaDot,
  float alpha,
  float alphaDot,
  int pwm,
  int mode,
  float blend,
  int pot,
  int32_t enc
) {
  Serial.print("TEL,");
  Serial.print(runId);
  Serial.print(",");
  Serial.print(step);
  Serial.print(",");
  Serial.print(theta, 7);
  Serial.print(",");
  Serial.print(thetaDot, 7);
  Serial.print(",");
  Serial.print(alpha, 7);
  Serial.print(",");
  Serial.print(alphaDot, 7);
  Serial.print(",");
  Serial.print(pwm);
  Serial.print(",");
  Serial.print(mode);
  Serial.print(",");
  Serial.print(blend, 7);
  Serial.print(",");
  Serial.print(pot);
  Serial.print(",");
  Serial.println(enc);
}

static void printMonitor(
  float theta,
  float thetaDot,
  float alpha,
  float alphaDot,
  int pot,
  int32_t enc
) {
  Serial.print("MON,");
  Serial.print(theta, 7);
  Serial.print(",");
  Serial.print(thetaDot, 7);
  Serial.print(",");
  Serial.print(alpha, 7);
  Serial.print(",");
  Serial.print(alphaDot, 7);
  Serial.print(",");
  Serial.print(pot);
  Serial.print(",");
  Serial.println(enc);
}

static void processControlTick() {
  ControllerConfig c;
  int pot;
  int32_t enc;
  int32_t encZero;
  bool armed;
  uint32_t step;
  uint32_t runId;
  noInterrupts();
  c = g_cfg;
  pot = g_pot_raw;
  enc = g_encoder_raw;
  encZero = g_encoder_zero;
  armed = g_armed;
  step = g_step;
  runId = g_run_id;
  interrupts();

  float theta = (float)(enc - encZero) * c.theta_rad_per_count;
  float alpha = alphaFromPot(pot, c);

  if (!g_velocity_ready) {
    g_prev_theta = theta;
    g_prev_alpha = alpha;
    g_theta_dot_lpf = 0.0f;
    g_alpha_dot_lpf = 0.0f;
    g_velocity_ready = true;
  } else {
    float rawThetaDot = (theta - g_prev_theta) / DT;
    float rawAlphaDot = wrapToPi(alpha - g_prev_alpha) / DT;
    g_theta_dot_lpf += c.velocity_lpf * (rawThetaDot - g_theta_dot_lpf);
    g_alpha_dot_lpf += c.velocity_lpf * (rawAlphaDot - g_alpha_dot_lpf);
    g_prev_theta = theta;
    g_prev_alpha = alpha;
  }

  float thetaDot = g_theta_dot_lpf;
  float alphaDot = g_alpha_dot_lpf;

  if (!armed) {
    motorBrake();
    g_idle_divider++;
    if (g_idle_divider >= IDLE_TELEMETRY_DIVIDER) {
      g_idle_divider = 0;
      printMonitor(theta, thetaDot, alpha, alphaDot, pot, enc);
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

  float enter = c.enter_deg * PI_F / 180.0f;
  float exitAngle = c.exit_deg * PI_F / 180.0f;
  float absAlpha = fabsf(alpha);

  if (!g_upright && absAlpha <= enter) {
    g_upright = true;
    g_int_alpha = 0.0f;
    g_int_theta = 0.0f;
  } else if (g_upright && absAlpha >= exitAngle) {
    g_upright = false;
    g_int_alpha = 0.0f;
    g_int_theta = 0.0f;
  }

  if (g_upright) {
    g_int_alpha = clampf(
      g_int_alpha + alpha * DT,
      -c.alpha_i_limit,
      c.alpha_i_limit
    );
    g_int_theta = clampf(
      g_int_theta + theta * DT,
      -c.theta_i_limit,
      c.theta_i_limit
    );
  }

  float swing;
  if (runTime < c.kick_time) {
    swing = c.swing_pwm;
  } else {
    float phase = alphaDot * cosf(alpha);
    float direction = phase >= 0.0f ? 1.0f : -1.0f;
    swing = -c.swing_pwm * direction;
  }

  float pid =
    c.kp_alpha * alpha +
    c.ki_alpha * g_int_alpha +
    c.kd_alpha * alphaDot +
    c.kp_theta * theta +
    c.ki_theta * g_int_theta +
    c.kd_theta * thetaDot;

  float target = g_upright ? 1.0f : 0.0f;
  g_blend += c.blend_alpha * (target - g_blend);
  g_blend = clampf(g_blend, 0.0f, 1.0f);

  float command = (1.0f - g_blend) * swing + g_blend * pid;
  command = clampf(command, -c.pwm_limit, c.pwm_limit);
  int controllerPwm = (int)lroundf(command);
  int pwmApplied = c.motor_sign * controllerPwm;
  pwmApplied = constrain(pwmApplied, -(int)c.pwm_limit, (int)c.pwm_limit);
  motorDrive(pwmApplied);

  int mode;
  if (g_blend <= 0.01f) mode = MODE_SWING;
  else if (g_blend >= 0.99f) mode = MODE_PID;
  else mode = MODE_BLEND;

  noInterrupts();
  g_step = step + 1;
  g_pwm = pwmApplied;
  g_mode = mode;
  interrupts();

  printTelemetry(
    runId,
    step,
    theta,
    thetaDot,
    alpha,
    alphaDot,
    pwmApplied,
    mode,
    g_blend,
    pot,
    enc
  );
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
  Serial.println("READY,RIP_DUAL_PID_HW,2");
}

void loop() {
  readSerialLines();
  if (!g_tick) return;
  noInterrupts();
  g_tick = false;
  interrupts();
  processControlTick();
}
