#ifndef DQN_INFERENCE_TEMPLATE_H
#define DQN_INFERENCE_TEMPLATE_H

#include <stdint.h>
#include "model_weights.h"

// Input must match training exactly:
// [sin(theta), cos(theta), theta_dot_LPF,
//  sin(alpha), cos(alpha), alpha_dot_LPF]
// theta_dot/alpha_dot are raw rad/s estimates after LPF=0.25 and clipping.

static inline float dqn_relu(float x) { return x > 0.0f ? x : 0.0f; }

static inline int dqn_action_index(const float obs[MODEL_INPUT_DIM]) {
    float h1[MODEL_H1];
    float h2[MODEL_H2];
    float q[MODEL_OUTPUT_DIM];

    for (int o = 0; o < MODEL_H1; ++o) {
        float sum = DQN_b1[o];
        for (int i = 0; i < MODEL_INPUT_DIM; ++i) sum += DQN_W1[o][i] * obs[i];
        h1[o] = dqn_relu(sum);
    }
    for (int o = 0; o < MODEL_H2; ++o) {
        float sum = DQN_b2[o];
        for (int i = 0; i < MODEL_H1; ++i) sum += DQN_W2[o][i] * h1[i];
        h2[o] = dqn_relu(sum);
    }
    for (int o = 0; o < MODEL_OUTPUT_DIM; ++o) {
        float sum = DQN_b3[o];
        for (int i = 0; i < MODEL_H2; ++i) sum += DQN_W3[o][i] * h2[i];
        q[o] = sum;
    }

    int best = 0;
    for (int i = 1; i < MODEL_OUTPUT_DIM; ++i) {
        if (q[i] > q[best]) best = i;
    }
    return best;
}

static inline int16_t dqn_pwm(const float obs[MODEL_INPUT_DIM]) {
    return DQN_ACTIONS[dqn_action_index(obs)];
}

#endif
