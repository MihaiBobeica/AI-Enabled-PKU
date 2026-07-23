/*
 * Class 9 rotary inverted pendulum firmware
 * Hybrid phase-pumping swing-up + constrained linear MPC stabilisation
 *
 * State order: x = [theta, theta_dot, alpha, alpha_dot]^T
 * Student-selectable state estimator:
 *   0 = angle difference + first-order low-pass velocity filter
 *   1 = fixed Luenberger observer from the previous MPC firmware
 *
 * Serial protocol version: 4
 */
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
static const int MPC_MAX_N = 20;

static const int MODE_DISABLED = 0;
static const int MODE_SWING = 1;
static const int MODE_BLEND = 2;
static const int MODE_MPC = 3;
static const int ESTIMATOR_DIFFERENTIAL = 0;
static const int ESTIMATOR_LUENBERGER = 1;

struct ControllerConfig {
  float duration;
  int horizon;
  float q_theta;
  float q_theta_dot;
  float q_alpha;
  float q_alpha_dot;
  float r_input;
  int pgd_iterations;
  int estimator_mode;
  float pwm_limit;
  float swing_pwm;
  float kick_time;
  float enter_deg;
  float exit_deg;
  float blend_alpha;
  float velocity_lpf;
  int32_t pot_up;
  int32_t pot_down;
  float theta_rad_per_count;
  int motor_sign;
};

static ControllerConfig g_cfg = {
  10.0f,
  8,
  1.0f, 0.05f, 80.0f, 2.0f,
  0.001f,
  16,
  ESTIMATOR_DIFFERENTIAL,
  150.0f, 120.0f, 0.10f,
  15.0f, 25.0f, 0.18f,
  0.25f,
  3110, 990,
  0.00583730846f, 1
};

// Exact 5 ms ZOH model used to generate the validated legacy MPC H/G matrices.
static const float MPC_A[4][4] = {
  {1.0f, 0.0048612691704992f, -0.0006112340140108f, 0.0000048745269118f},
  {0.0f, 0.9450502885145126f, -0.2419612332597028f, 0.0017237656430326f},
  {0.0f, 0.0002070777337968f, 1.0021161748142602f, 0.0049831142945805f},
  {0.0f, 0.0819731603634566f, 0.8422019829266282f, 0.9939886689544905f}
};
static const float MPC_B[4] = {
  0.0000100237507910f, 0.0039702942449759f, -0.0000149620355143f, -0.0059228257625436f
};

// Fixed Luenberger observer copied from the previously supplied MPC firmware.
static const float OBS_A_LC[4][4] = {
  {0.82469568f, 0.00438721f, -0.31915638f, -0.00085553f},
  {-1.75095359f, 0.94056709f, -0.04181246f, 0.00193564f},
  {-0.05215683f, 0.00005656f, 0.76549687f, 0.00439733f},
  {-1.38405407f, 0.07685667f, -15.83946960f, 0.95027468f}
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
static float g_theta_dot_lpf = 0.0f;
static float g_alpha_dot_lpf = 0.0f;
static bool g_velocity_ready = false;
static bool g_upright = false;
static float g_previous_controller_pwm = 0.0f;

static float g_xhat[4] = {0, 0, 0, 0};
static bool g_observer_ready = false;

static float g_mpc_h[MPC_MAX_N][MPC_MAX_N];
static float g_mpc_g[MPC_MAX_N][4];
static float g_mpc_u_sequence[MPC_MAX_N];
static float g_mpc_terminal_p[4][4];
static float g_mpc_pgd_step = 0.0f;
static bool g_mpc_ready = false;

// Work arrays are global to avoid a large temporary stack allocation during SAVE.
static float g_power_a[MPC_MAX_N + 1][4][4];
static float g_su[MPC_MAX_N][MPC_MAX_N][4];

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
  if (!(c.horizon >= 4 && c.horizon <= MPC_MAX_N)) return false;
  if (!(isfinite(c.q_theta) && c.q_theta >= 0.0f)) return false;
  if (!(isfinite(c.q_theta_dot) && c.q_theta_dot >= 0.0f)) return false;
  if (!(isfinite(c.q_alpha) && c.q_alpha >= 0.0f)) return false;
  if (!(isfinite(c.q_alpha_dot) && c.q_alpha_dot >= 0.0f)) return false;
  if (!(c.q_theta + c.q_theta_dot + c.q_alpha + c.q_alpha_dot > 0.0f)) return false;
  if (!(isfinite(c.r_input) && c.r_input > 0.0f && c.r_input <= 1000.0f)) return false;
  if (!(c.pgd_iterations >= 1 && c.pgd_iterations <= 100)) return false;
  if (!(c.estimator_mode == ESTIMATOR_DIFFERENTIAL || c.estimator_mode == ESTIMATOR_LUENBERGER)) return false;
  if (!(c.pwm_limit >= 1.0f && c.pwm_limit <= 255.0f)) return false;
  if (!(c.swing_pwm >= 0.0f && c.swing_pwm <= c.pwm_limit)) return false;
  if (!(c.kick_time >= 0.0f && c.kick_time <= 10.0f)) return false;
  if (!(c.enter_deg > 0.0f && c.enter_deg < c.exit_deg && c.exit_deg < 180.0f)) return false;
  if (!(c.blend_alpha > 0.0f && c.blend_alpha <= 1.0f)) return false;
  if (!(c.velocity_lpf > 0.0f && c.velocity_lpf <= 1.0f)) return false;
  if (!(c.pot_up >= 0 && c.pot_up < POT_MOD)) return false;
  if (!(c.pot_down >= 0 && c.pot_down < POT_MOD)) return false;
  if (c.pot_up == c.pot_down) return false;
  if (!(fabsf(c.theta_rad_per_count) > 1.0e-8f && fabsf(c.theta_rad_per_count) < 1.0f)) return false;
  if (!(c.motor_sign == 1 || c.motor_sign == -1)) return false;
  return true;
}

static void matrixIdentity4(float out[4][4]) {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) out[i][j] = (i == j) ? 1.0f : 0.0f;
  }
}

static void matrixMultiply4(const float a[4][4], const float b[4][4], float out[4][4]) {
  float temp[4][4];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      float value = 0.0f;
      for (int k = 0; k < 4; ++k) value += a[i][k] * b[k][j];
      temp[i][j] = value;
    }
  }
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) out[i][j] = temp[i][j];
}

static void matrixVector4(const float a[4][4], const float b[4], float out[4]) {
  for (int i = 0; i < 4; ++i) {
    float value = 0.0f;
    for (int j = 0; j < 4; ++j) value += a[i][j] * b[j];
    out[i] = value;
  }
}

static bool solveTerminalRiccati(const ControllerConfig& c, float pOut[4][4]) {
  float q[4] = {c.q_theta, c.q_theta_dot, c.q_alpha, c.q_alpha_dot};
  float p[4][4] = {{0}};
  for (int i = 0; i < 4; ++i) p[i][i] = q[i];

  for (int iteration = 0; iteration < 20000; ++iteration) {
    float pB[4] = {0};
    float pA[4][4] = {{0}};
    for (int i = 0; i < 4; ++i) {
      for (int k = 0; k < 4; ++k) pB[i] += p[i][k] * MPC_B[k];
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 4; ++k) pA[i][j] += p[i][k] * MPC_A[k][j];
      }
    }
    float denominator = c.r_input;
    for (int i = 0; i < 4; ++i) denominator += MPC_B[i] * pB[i];
    if (!isfinite(denominator) || denominator <= 1.0e-12f) return false;

    float atPA[4][4] = {{0}};
    float atPB[4] = {0};
    float btPA[4] = {0};
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 4; ++k) atPA[i][j] += MPC_A[k][i] * pA[k][j];
      }
      for (int k = 0; k < 4; ++k) atPB[i] += MPC_A[k][i] * pB[k];
    }
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) btPA[j] += MPC_B[k] * pA[k][j];
    }

    float nextP[4][4];
    float maxChange = 0.0f;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        float value = atPA[i][j] - atPB[i] * btPA[j] / denominator;
        if (i == j) value += q[i];
        nextP[i][j] = value;
      }
    }
    for (int i = 0; i < 4; ++i) {
      for (int j = i; j < 4; ++j) {
        float symmetric = 0.5f * (nextP[i][j] + nextP[j][i]);
        nextP[i][j] = symmetric;
        nextP[j][i] = symmetric;
      }
    }
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        if (!isfinite(nextP[i][j])) return false;
        float change = fabsf(nextP[i][j] - p[i][j]);
        if (change > maxChange) maxChange = change;
        p[i][j] = nextP[i][j];
      }
    }
    if (maxChange < 1.0e-4f) {
      for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) pOut[i][j] = p[i][j];
      return true;
    }
  }
  return false;
}

static bool buildMpcMatrices(const ControllerConfig& c) {
  if (!solveTerminalRiccati(c, g_mpc_terminal_p)) return false;
  int n = c.horizon;

  matrixIdentity4(g_power_a[0]);
  for (int k = 1; k <= n; ++k) matrixMultiply4(g_power_a[k - 1], MPC_A, g_power_a[k]);

  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int state = 0; state < 4; ++state) g_su[k][j][state] = 0.0f;
      if (j <= k) matrixVector4(g_power_a[k - j], MPC_B, g_su[k][j]);
    }
  }

  for (int i = 0; i < MPC_MAX_N; ++i) {
    g_mpc_u_sequence[i] = 0.0f;
    for (int j = 0; j < MPC_MAX_N; ++j) g_mpc_h[i][j] = 0.0f;
    for (int state = 0; state < 4; ++state) g_mpc_g[i][state] = 0.0f;
  }

  float qDiag[4] = {c.q_theta, c.q_theta_dot, c.q_alpha, c.q_alpha_dot};
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      float value = (i == j) ? c.r_input : 0.0f;
      int firstK = (i > j) ? i : j;
      for (int k = firstK; k < n; ++k) {
        if (k == n - 1) {
          for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
              value += g_su[k][i][a] * g_mpc_terminal_p[a][b] * g_su[k][j][b];
            }
          }
        } else {
          for (int a = 0; a < 4; ++a) value += g_su[k][i][a] * qDiag[a] * g_su[k][j][a];
        }
      }
      g_mpc_h[i][j] = 2.0f * value;
    }

    for (int state = 0; state < 4; ++state) {
      float value = 0.0f;
      for (int k = i; k < n; ++k) {
        if (k == n - 1) {
          for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
              value += g_su[k][i][a] * g_mpc_terminal_p[a][b] * g_power_a[k + 1][b][state];
            }
          }
        } else {
          for (int a = 0; a < 4; ++a) {
            value += g_su[k][i][a] * qDiag[a] * g_power_a[k + 1][a][state];
          }
        }
      }
      g_mpc_g[i][state] = 2.0f * value;
    }
  }

  float maxRowSum = 0.0f;
  for (int i = 0; i < n; ++i) {
    float rowSum = 0.0f;
    for (int j = 0; j < n; ++j) {
      if (!isfinite(g_mpc_h[i][j])) return false;
      rowSum += fabsf(g_mpc_h[i][j]);
    }
    if (rowSum > maxRowSum) maxRowSum = rowSum;
  }
  if (!isfinite(maxRowSum) || maxRowSum <= 1.0e-12f) return false;
  g_mpc_pgd_step = 0.95f / maxRowSum;
  if (!isfinite(g_mpc_pgd_step) || g_mpc_pgd_step <= 0.0f) return false;
  return true;
}

static void resetMpcSequence() {
  for (int i = 0; i < MPC_MAX_N; ++i) g_mpc_u_sequence[i] = 0.0f;
}

static float computeMpc(const float xInput[4], const ControllerConfig& c) {
  int n = c.horizon;
  float x[4] = {xInput[0], xInput[1], wrapToPi(xInput[2]), xInput[3]};
  float u[MPC_MAX_N];
  for (int i = 0; i < n - 1; ++i) u[i] = g_mpc_u_sequence[i + 1];
  u[n - 1] = g_mpc_u_sequence[n - 1];
  for (int i = 0; i < n; ++i) u[i] = clampf(u[i], -c.pwm_limit, c.pwm_limit);

  float linear[MPC_MAX_N];
  for (int i = 0; i < n; ++i) {
    linear[i] = 0.0f;
    for (int state = 0; state < 4; ++state) linear[i] += g_mpc_g[i][state] * x[state];
  }

  for (int iteration = 0; iteration < c.pgd_iterations; ++iteration) {
    float gradient[MPC_MAX_N];
    for (int i = 0; i < n; ++i) {
      float value = linear[i];
      for (int j = 0; j < n; ++j) value += g_mpc_h[i][j] * u[j];
      gradient[i] = value;
    }
    for (int i = 0; i < n; ++i) {
      u[i] = clampf(u[i] - g_mpc_pgd_step * gradient[i], -c.pwm_limit, c.pwm_limit);
    }
  }

  for (int i = 0; i < n; ++i) {
    if (!isfinite(u[i])) u[i] = 0.0f;
    g_mpc_u_sequence[i] = clampf(u[i], -c.pwm_limit, c.pwm_limit);
  }
  return g_mpc_u_sequence[0];
}

static void observerReset() {
  for (int i = 0; i < 4; ++i) g_xhat[i] = 0.0f;
  g_observer_ready = false;
}

static void observerInitialize(float theta, float alpha) {
  g_xhat[0] = theta;
  g_xhat[1] = 0.0f;
  g_xhat[2] = wrapToPi(alpha);
  g_xhat[3] = 0.0f;
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
  if (!finite4(nextState) || fabsf(nextState[2] - alpha) > OBS_RESET_ERR_LIMIT) {
    observerInitialize(theta, alpha);
    return;
  }
  nextState[1] = clampf(nextState[1], -OBS_VEL_CLIP, OBS_VEL_CLIP);
  nextState[3] = clampf(nextState[3], -OBS_VEL_CLIP, OBS_VEL_CLIP);
  nextState[2] = wrapToPi(nextState[2]);
  for (int i = 0; i < 4; ++i) g_xhat[i] = nextState[i];
}

static void stopControl() {
  noInterrupts();
  g_armed = false;
  g_pwm = 0;
  g_mode = MODE_DISABLED;
  g_blend = 0.0f;
  interrupts();
  g_upright = false;
  g_previous_controller_pwm = 0.0f;
  resetMpcSequence();
  observerReset();
  motorBrake();
}

static void startControl(uint32_t runId) {
  if (!g_config_valid || !g_mpc_ready) {
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
  g_prev_theta = 0.0f;
  g_prev_alpha = 0.0f;
  g_theta_dot_lpf = 0.0f;
  g_alpha_dot_lpf = 0.0f;
  g_velocity_ready = false;
  g_upright = false;
  g_previous_controller_pwm = 0.0f;
  resetMpcSequence();
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
  bool valid = g_config_valid && g_mpc_ready;
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
  Serial.print(c.motor_sign);
  Serial.print(",");
  Serial.print(c.horizon);
  Serial.print(",");
  Serial.println(c.estimator_mode);
}

static void applyConfig(const String& line) {
  String fields[24];
  int count = splitCsv(line, fields, 24);
  if (count != 23 || fields[1].toInt() != 4) {
    Serial.println("ERR,CONFIG_FORMAT");
    return;
  }
  uint32_t token = (uint32_t)fields[2].toInt();
  ControllerConfig c;
  c.duration = fields[3].toFloat();
  c.horizon = fields[4].toInt();
  c.q_theta = fields[5].toFloat();
  c.q_theta_dot = fields[6].toFloat();
  c.q_alpha = fields[7].toFloat();
  c.q_alpha_dot = fields[8].toFloat();
  c.r_input = fields[9].toFloat();
  c.pgd_iterations = fields[10].toInt();
  c.estimator_mode = fields[11].toInt();
  c.pwm_limit = fields[12].toFloat();
  c.swing_pwm = fields[13].toFloat();
  c.kick_time = fields[14].toFloat();
  c.enter_deg = fields[15].toFloat();
  c.exit_deg = fields[16].toFloat();
  c.blend_alpha = fields[17].toFloat();
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
  g_mpc_ready = false;
  if (!buildMpcMatrices(c)) {
    Serial.println("ERR,MPC_BUILD");
    return;
  }
  noInterrupts();
  g_cfg = c;
  g_config_valid = true;
  g_mpc_ready = true;
  interrupts();
  Serial.print("ACK,CONFIG,");
  Serial.println(token);
}

static void handleCommand(const String& raw) {
  String line = raw;
  line.trim();
  if (line.length() == 0) return;
  if (line == "HELLO") {
    Serial.println("READY,RIP_MPC_HW,4");
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

  float differentialState[4] = {theta, g_theta_dot_lpf, alpha, g_alpha_dot_lpf};

  if (!armed) {
    motorBrake();
    g_idle_divider++;
    if (g_idle_divider >= IDLE_TELEMETRY_DIVIDER) {
      g_idle_divider = 0;
      printMonitor(theta, g_theta_dot_lpf, alpha, g_alpha_dot_lpf, pot, enc);
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
  bool wasUpright = g_upright;
  if (!g_upright && absAlpha <= enter) {
    g_upright = true;
    resetMpcSequence();
    if (c.estimator_mode == ESTIMATOR_LUENBERGER) observerInitialize(theta, alpha);
  } else if (g_upright && absAlpha >= exitAngle) {
    g_upright = false;
    resetMpcSequence();
    observerReset();
  }
  (void)wasUpright;

  float controlState[4];
  if (c.estimator_mode == ESTIMATOR_LUENBERGER && g_upright) {
    observerUpdate(g_previous_controller_pwm, theta, alpha);
    for (int i = 0; i < 4; ++i) controlState[i] = g_xhat[i];
  } else {
    for (int i = 0; i < 4; ++i) controlState[i] = differentialState[i];
  }

  float swing;
  if (runTime < c.kick_time) {
    swing = c.swing_pwm;
  } else {
    float phase = g_alpha_dot_lpf * cosf(alpha);
    float direction = phase >= 0.0f ? 1.0f : -1.0f;
    swing = -c.swing_pwm * direction;
  }

  float mpc = (g_upright || g_blend > 1.0e-6f) ? computeMpc(controlState, c) : 0.0f;
  float target = g_upright ? 1.0f : 0.0f;
  g_blend += c.blend_alpha * (target - g_blend);
  g_blend = clampf(g_blend, 0.0f, 1.0f);

  float command = (1.0f - g_blend) * swing + g_blend * mpc;
  command = clampf(command, -c.pwm_limit, c.pwm_limit);
  int controllerPwm = (int)lroundf(command);
  g_previous_controller_pwm = (float)controllerPwm;
  int pwmApplied = c.motor_sign * controllerPwm;
  pwmApplied = constrain(pwmApplied, -(int)c.pwm_limit, (int)c.pwm_limit);
  motorDrive(pwmApplied);

  int mode;
  if (g_blend <= 0.01f) mode = MODE_SWING;
  else if (g_blend >= 0.99f) mode = MODE_MPC;
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
    controlState[1],
    alpha,
    controlState[3],
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
  Serial.println("READY,RIP_MPC_HW,4");
}

void loop() {
  readSerialLines();
  if (!g_tick) return;
  noInterrupts();
  g_tick = false;
  interrupts();
  processControlTick();
}
