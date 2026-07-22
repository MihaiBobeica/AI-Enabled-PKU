%% pendulum_dqn_singleNet_swing_then_upright_SINGLE_v1_stop10_DYNRN_fullstate_hardgate10.m
% Single-network DQN for Furuta rotary inverted pendulum:
% - One DQN learns swing-up + balance
% - Reward style: 10*cos(alpha) - vel^2 penalties - angle penalty
% - Noise OFF, Delay OFF
%
% 核心改动（最小修改版）：
%   1) 总开关仍然使用 env.use_obs_rn（不改变量名）
%   2) 10度以内 hard gate 保持不变
%   3) RN 不再修观测层
%   4) 环境推进永远先走纯物理模型
%   5) 若 gate 打开，则 RN 输出“下一步状态残差”，加到纯物理 x_phys_next 上
%   6) DQN 使用原始观测+LPF速度估计来构造控制输入状态
%
% RN 语义现在变为：
%   输入: 最近 seq_len 步 [theta, theta_dot, alpha, alpha_dot, u]
%   输出: 下一步状态残差
%         - alpha_only       -> [dalpha_next]
%         - alpha_alpha_dot  -> [dalpha_next, dalpha_dot_next]
%         - full_state       -> [dtheta_next, dtheta_dot_next, dalpha_next, dalpha_dot_next]
%
% 即：
%   x_true_next = x_phys_next + delta_x_rn
%
% 注意：
%   下面的 env.obs_rn.mat_path 现在必须指向“动力学残差 RN”的 mat

clear; clc; rng(42);

%% ======================= 0) Physical / Motor params =======================
C.g   = 9.8;
C.c_theta = 0.025;
C.c_alpha = 0.001;
C.k_t = 0.2310;
C.k_b = 0.1875;
C.k_u = 0.04706;
C.R   = 4.2857;
C.K1  = C.k_t*C.k_u/C.R;
C.K2  = C.k_t*C.k_b/C.R;

C.m1   = 0.20625;
C.m2   = 0.15845;
C.l1cg = 0.080305;
C.l1   = 0.151894;
C.l2cg = 0.066733;
C.I1z  = 0.00049228;
C.I2x  = 0.00036892;
C.I2y  = 2.3641e-05;
C.I2z  = 0.00036139;

%% ======================= 1) Env / limits / init =======================
env.dt = 0.005;

env.maxSteps = 2000;           % per-episode cap
env.theta_lim_true    = 12*pi; % unwrap hard guard (theta only)
env.thetaDot_lim_base = 45;    % done threshold
env.alphaDot_lim_base = 40;    % done threshold

% Delay OFF
env.use_delay     = false;
env.delay_mu_ms   = 0;
env.delay_sig_ms  = 0;
env.delay_lo_ms   = 0;
env.delay_hi_ms   = 0;

% Measurement noise OFF
meas.use_noise = false;
meas.theta_sigma = 0;
meas.alpha_sigma = 0;
meas.bias_target_sigma = 0;
meas.bias_ar = 0;
meas.bias_inj = 0;

% DOWNWARD init (alpha near pi)
env.init_true_down = @() [ ...
    (randn()*deg2rad(5)); ...
    0.0; ...
    wrapToPi(pi + randn()*deg2rad(8)); ...
    0.0 ];

%% ======================= 1.5) Dynamics RN config =======================
% ---------------------------
% 是否启用“动力学残差 RN”
% 注意：总开关变量名保持不变，不改成别的
% ---------------------------
env.use_obs_rn = false;

% ---------------------------
% 模式：
%   'alpha_only'       -> 只修 alpha_next
%   'alpha_alpha_dot'  -> 修 alpha_next, alpha_dot_next
%   'full_state'       -> 修 [theta_next, theta_dot_next, alpha_next, alpha_dot_next]
% ---------------------------
env.obs_rn.mode = 'alpha_alpha_dot';

% ---------------------------
% 训练时允许运行
% ---------------------------
env.obs_rn.runtime_enabled = true;
env.obs_rn.phase_name = 'dyn_rn_fullstate_hardgate10';

% ---------------------------
% 仅当 |alpha_true| <= 10 deg 时启用动力学 RN
% ---------------------------
env.obs_rn.alpha_gate_deg = 10.0;

% ---------------------------
% 动力学残差限幅
% 注意：这是“对下一步状态的修正量”
% ---------------------------
env.obs_rn.clip_theta_corr      = 0.20;          % rad
env.obs_rn.clip_theta_dot_corr  = 5.0;           % rad/s
env.obs_rn.clip_alpha_corr      = deg2rad(5.0);  % rad
env.obs_rn.clip_alpha_dot_corr  = 5.0;           % rad/s

% ---------------------------
% 四通道独立增益
% 默认先只强修摆杆通道
% ---------------------------
env.obs_rn.gain_theta     = 0.0;
env.obs_rn.gain_theta_dot = 0.0;
env.obs_rn.gain_alpha     = 1.0;
env.obs_rn.gain_alpha_dot = 1.0;

% ---------------------------
% 这里现在必须是“动力学残差 RN”的 mat
% ---------------------------
env.obs_rn.mat_path = '/Users/juzhixiang/Desktop/pendulum_project/Pendulum(Macbook+STM32)/RN_data_collect/dataset_dynrn_from_observer_seq5_20260401_154244/rn_seq5_observer_alphaadot_residual_run_20260401_212952/rn_model_export__best_rn_seq5_observer_alphaadot_residual.mat';
if env.use_obs_rn
    fprintf('\n[DYN-RN] Loading dynamics RN model from:\n%s\n', env.obs_rn.mat_path);
    env.obs_rn.model = load_rn_bundle_from_mat(env.obs_rn.mat_path);
    fprintf('[DYN-RN] Loaded. seq_len = %d, input_dim = %d, output_dim = %d, mode = %s\n', ...
        env.obs_rn.model.seq_len, env.obs_rn.model.input_dim, env.obs_rn.model.output_dim, env.obs_rn.mode);
    fprintf('[DYN-RN] Runtime enabled from the VERY BEGINNING.\n');
    fprintf('[DYN-RN] Hard gate: active only when |alpha_true| <= %.1f deg.\n', env.obs_rn.alpha_gate_deg);
    fprintf('[DYN-RN] Channel gains: theta=%.3f, theta_dot=%.3f, alpha=%.3f, alpha_dot=%.3f\n', ...
        env.obs_rn.gain_theta, env.obs_rn.gain_theta_dot, env.obs_rn.gain_alpha, env.obs_rn.gain_alpha_dot);
else
    env.obs_rn.model = [];
    fprintf('\n[DYN-RN] Disabled. Pure dynamics only.\n');
end

%% ======================= 2) Actions =======================
ACTIONS = int16([-150,-120,-90,-60,-30,30,60,90,120,150]);
actDim  = numel(ACTIONS);

%% ======================= 3) Velocity estimation =======================
lpf = 0.25;

%% ======================= 4) Reward (Python-style) =======================
RW.KCOS = 10.0;
RW.KALD = 0.001;
RW.KTHD = 0.0001;
RW.ANG_PEN_DEG = 15;
RW.ANG_PEN_VAL = 5.0;

R_step = @(alpha_wrap, thetaDot, alphaDot) ...
    ( RW.KCOS*cos(alpha_wrap) ...
      - RW.KALD*(alphaDot.^2) ...
      - RW.KTHD*(thetaDot.^2) ...
      - RW.ANG_PEN_VAL*double(abs(alpha_wrap) > deg2rad(RW.ANG_PEN_DEG)) );

%% ======================= 5) "Maintain" success metric =======================
MA.TH_DEG  = 15;
MA.ALD_LIM = 4.0;

%% ======================= 6) DQN =======================
stateDim = 6; % [sin(theta); cos(theta); thetaDot; sin(alpha); cos(alpha); alphaDot]

net.h = [64, 64];
net.W1 = 0.01*randn(net.h(1),stateDim); net.b1 = zeros(net.h(1),1);
net.W2 = 0.01*randn(net.h(2),net.h(1)); net.b2 = zeros(net.h(2),1);
net.W3 = 0.01*randn(actDim,net.h(2));   net.b3 = zeros(actDim,1);
tgt = net;
net_snap = net;

SNAP_EVERY_STEPS = 50;

%% ======================= 7) Optimizer (Adam) =======================
opt.lr = 5e-4; opt.beta1 = 0.9; opt.beta2 = 0.999; opt.eps = 1e-8; opt.t = 0;
[opt.mW1,opt.vW1] = deal(zeros(size(net.W1))); [opt.mb1,opt.vb1] = deal(zeros(size(net.b1)));
[opt.mW2,opt.vW2] = deal(zeros(size(net.W2))); [opt.mb2,opt.vb2] = deal(zeros(size(net.b2)));
[opt.mW3,opt.vW3] = deal(zeros(size(net.W3))); [opt.mb3,opt.vb3] = deal(zeros(size(net.b3)));

%% ======================= 8) Training config =======================
conf.gamma  = 0.95;
conf.batch  = 512;
conf.buf    = 30000;
conf.warm   = 2000;
conf.steps  = 5000000;

conf.eps0   = 1.0;
conf.eps1   = 0.001;
conf.edecay = 400000.0;

conf.sync   = 2000;
conf.save_every_steps = 20000;

conf.stop_avg_window = 10;
conf.stop_avg_thresh = 1400;

%% ======================= 9) Replay buffer =======================
B = replay_init(stateDim, conf.buf);

%% ======================= 10) Saving paths =======================
desk = getDesktopDir();
if env.use_obs_rn
    runDir = fullfile(desk, ['furuta_dqn_singleNet_DYNRN_', env.obs_rn.mode, '_hardgate10_avgstop_', datestr(now,'yyyymmdd_HHMMSS')]);
else
    runDir = fullfile(desk, ['furuta_dqn_singleNet_noDynRN_', datestr(now,'yyyymmdd_HHMMSS')]);
end
if ~exist(runDir,'dir'), mkdir(runDir); end

finalMat  = fullfile(runDir,'model_singleNet_stop10.mat');
trainPng  = fullfile(runDir,'training_curve.png');
rollPng   = fullfile(runDir,'greedy_rollout.png');
hdrPath   = fullfile(runDir,'model_weights.h');

%% ======================= 11) Plot =======================
movWin = 30;
stats.Steps = [];
stats.EpRet = [];
stats.EpRetAvg = [];
stats.EpLen = [];
stats.MaintainMaxStable = [];
stats.ObsRNEnabled = [];
stats.Phase = strings(0,1);

fig = figure('Name','SingleNet DQN Training (DYN-RN hard gate 10deg)','Color','w');
ax = axes(fig); hold(ax,'on'); grid(ax,'on');
xlabel(ax,'global_step'); ylabel(ax,'return per step');
hLine = plot(ax,nan,nan,'-k','LineWidth',1.6);
hDots = plot(ax,nan,nan,'.','Color',[0.2 0.5 1]);
legend(ax,{'moving avg (30 ep)','episode return/step'},'Location','best');

t_all = tic;
tim.step_ms = nan(1, conf.steps + 1000);

%% ======================= 12) Training state =======================
step_global = 0;
ep = 0;
alpha_bias = 0.0;

train_stop_reason = "not_finished";
user_interrupted = false;

fprintf('\n==================== SINGLE-NET DQN TRAIN ====================\n');
fprintf('[DYN-RN allow] env.use_obs_rn = %d, env.obs_rn.mode = %s\n', env.use_obs_rn, env.obs_rn.mode);
fprintf('[DYN-RN runtime] enabled from the beginning = %d\n', env.obs_rn.runtime_enabled);
fprintf('[DYN-RN hard gate] active only when |alpha_true| <= %.1f deg\n', env.obs_rn.alpha_gate_deg);
fprintf('[STOP] stop when last %d episodes mean(maintainMaxStable) > %.1f\n', ...
    conf.stop_avg_window, conf.stop_avg_thresh);

%% ======================= 13) Train loop =======================
try
    while step_global < conf.steps
        ep = ep + 1;

        epRet = 0; epSteps = 0;
        maintainCurStable = 0;
        maintainMaxStable = 0;

        % ---------- true initial state ----------
        x_true = double(env.init_true_down());

        % ---------- measurement ----------
        y_meas = make_measurement(x_true, meas, alpha_bias);
        theta_meas_prev = y_meas(1);
        alpha_meas_prev = y_meas(2);

        % unwrap + raw observer initialization
        theta_unwrap = theta_meas_prev;
        theta_unwrap_prev = theta_unwrap;

        thetaDot_raw = 0.0;
        alphaDot_raw = 0.0;
        thetaDot_est = 0.0;
        alphaDot_est = 0.0;

        % 上一个已执行动作
        u_cmd_prev = 0.0;

        % ---------- dynamics RN context ----------
        dyn_ctx = init_dyn_rn_ctx(env, x_true, 0.0);

        % ---------- initial control state (raw observation only) ----------
        theta_ctrl = theta_unwrap;
        thetaDot_ctrl = thetaDot_est;
        alpha_ctrl = wrapToPi(alpha_meas_prev);
        alphaDot_ctrl = alphaDot_est;

        s = make_state_sincos( ...
            theta_ctrl, ...
            clip(thetaDot_ctrl, env.thetaDot_lim_base), ...
            alpha_ctrl, ...
            clip(alphaDot_ctrl, env.alphaDot_lim_base));

        for t = 1:env.maxSteps
            if step_global >= conf.steps, break; end
            step_global = step_global + 1;
            wall_start = tic;

            eps = conf.eps1 + (conf.eps0-conf.eps1)*exp(-double(step_global)/conf.edecay);

            % ---------- select action ----------
            [aIdx, pwm_cmd] = select_action(net_snap, s, eps, actDim, ACTIONS);

            % ---------- pure physics + dynamics RN residual ----------
            [x_true_next, ~, dyn_ctx] = step_with_delay_dynamics_rn(x_true, u_cmd_prev, pwm_cmd, C, env, dyn_ctx);

            % ---------- measurement ----------
            y = make_measurement(x_true_next, meas, alpha_bias);
            theta_meas = y(1);
            alpha_meas = y(2);

            % ---------- raw observation / diff + LPF ----------
            dtheta_wrapped = wrapToPi(theta_meas - theta_meas_prev);
            theta_unwrap   = theta_unwrap + dtheta_wrapped;
            thetaDot_raw   = (theta_unwrap - theta_unwrap_prev) / env.dt;

            dalpha_wrapped = wrapToPi(alpha_meas - alpha_meas_prev);
            alphaDot_raw   = dalpha_wrapped / env.dt;

            thetaDot_est = (1-lpf)*thetaDot_est + lpf*thetaDot_raw;
            alphaDot_est = (1-lpf)*alphaDot_est + lpf*alphaDot_raw;

            theta_ctrl = theta_unwrap;
            thetaDot_ctrl = thetaDot_est;
            alpha_ctrl = wrapToPi(alpha_meas);
            alphaDot_ctrl = alphaDot_est;

            % ---------- done / reward / maintain all use raw estimated state ----------
            done = (abs(theta_ctrl) > env.theta_lim_true) || ...
                   (abs(thetaDot_ctrl) > env.thetaDot_lim_base) || ...
                   (abs(alphaDot_ctrl) > env.alphaDot_lim_base) || ...
                   any(~isfinite([theta_ctrl, alpha_ctrl, thetaDot_ctrl, alphaDot_ctrl]));

            thd_clip = clip(thetaDot_ctrl, env.thetaDot_lim_base);
            ald_clip = clip(alphaDot_ctrl, env.alphaDot_lim_base);
            r = R_step(alpha_ctrl, thd_clip, ald_clip);

            if (abs(alpha_ctrl) < deg2rad(MA.TH_DEG)) && (abs(alphaDot_ctrl) < MA.ALD_LIM)
                maintainCurStable = maintainCurStable + 1;
            else
                maintainCurStable = 0;
            end
            if maintainCurStable > maintainMaxStable
                maintainMaxStable = maintainCurStable;
            end

            s2 = make_state_sincos(theta_ctrl, thd_clip, alpha_ctrl, ald_clip);

            B = replay_push(B, s, aIdx, r, s2, done);

            if B.n >= conf.warm
                [net, opt] = dqn_learn_one(net, tgt, B, conf, opt);
                if mod(step_global, conf.sync) == 0
                    tgt = net;
                end
            end

            if mod(step_global, SNAP_EVERY_STEPS) == 0
                net_snap = net;
            end

            epRet   = epRet + r;
            epSteps = epSteps + 1;
            tim.step_ms(step_global) = toc(wall_start) * 1000.0;

            % ---------- update ----------
            x_true = x_true_next;
            theta_meas_prev = theta_meas;
            alpha_meas_prev = alpha_meas;

            theta_unwrap_prev = theta_unwrap;
            u_cmd_prev = pwm_cmd;
            s = s2;

            if done, break; end
        end

        epRetPerStep = epRet / max(1, epSteps);

        stats.Steps(end+1) = step_global;
        stats.EpRet(end+1) = epRetPerStep;
        stats.EpLen(end+1) = epSteps;
        stats.MaintainMaxStable(end+1) = maintainMaxStable;
        stats.ObsRNEnabled(end+1) = env.use_obs_rn;
        stats.Phase(end+1) = "DYN_RN_HARDGATE10";

        if numel(stats.EpRet) >= movWin
            stats.EpRetAvg(end+1) = mean(stats.EpRet(end-movWin+1:end));
        else
            stats.EpRetAvg(end+1) = mean(stats.EpRet);
        end

        set(hLine, 'XData', stats.Steps, 'YData', stats.EpRetAvg);
        set(hDots, 'XData', stats.Steps, 'YData', stats.EpRet);

        if numel(stats.MaintainMaxStable) >= conf.stop_avg_window
            recent_avg = mean(stats.MaintainMaxStable(end-conf.stop_avg_window+1:end));
        else
            recent_avg = mean(stats.MaintainMaxStable);
        end

        title(ax, sprintf(['[DQN] Ep %d | step %d/%d | movRet %.4f | epLen %d | ' ...
            'maintainMaxStable %d | last%dAvg %.2f | eps %.4f | elapsed %0.1fs | phase=%s | DYN-RN(use=%d,run=%d)'], ...
            ep, step_global, conf.steps, stats.EpRetAvg(end), epSteps, maintainMaxStable, ...
            conf.stop_avg_window, recent_avg, eps, toc(t_all), ...
            char(stats.Phase(end)), env.use_obs_rn, env.obs_rn.runtime_enabled));
        drawnow limitrate;

        fprintf('[DQN] Ep %4d | step=%7d | epRet/step=%.4f | eps=%.4f | epSteps=%d | maintainMaxStable=%d | last%dAvg=%.2f | phase=%s | DYN-RN(use=%d,run=%d)\n', ...
            ep, step_global, epRetPerStep, eps, epSteps, maintainMaxStable, ...
            conf.stop_avg_window, recent_avg, char(stats.Phase(end)), env.use_obs_rn, env.obs_rn.runtime_enabled);

        if numel(stats.MaintainMaxStable) >= conf.stop_avg_window
            recent_vals = stats.MaintainMaxStable(end-conf.stop_avg_window+1:end);
            recent_avg = mean(recent_vals);

            if recent_avg > conf.stop_avg_thresh
                fprintf('\n[STOP] Last %d episodes mean(maintainMaxStable) = %.3f > %.3f\n', ...
                    conf.stop_avg_window, recent_avg, conf.stop_avg_thresh);
                fprintf('[STOP] Training finished successfully.\n\n');
                train_stop_reason = "avg_maintain_success";
                break;
            end
        end

        if (mod(step_global, conf.save_every_steps) == 0) || (step_global >= conf.steps)
            save(finalMat, 'net','net_snap','tgt','stats','conf','ACTIONS','env','C','meas','tim', ...
                'train_stop_reason','-v7.3');
            exportgraphics(fig, trainPng, 'Resolution', 180);
            export_model_to_header_single(net, ACTIONS, hdrPath);
        end
    end

    if step_global >= conf.steps && train_stop_reason == "not_finished"
        train_stop_reason = "max_steps_reached";
    end

catch ME
    user_interrupted = true;
    train_stop_reason = "user_interrupted";
    fprintf('\n[INTERRUPTED] Training interrupted by user or runtime exception.\n');
    fprintf('[INTERRUPTED] Message: %s\n\n', ME.message);
end

%% ======================= 14) Final save =======================
save(finalMat, 'net','net_snap','tgt','stats','conf','ACTIONS','env','C','meas','tim', ...
    'train_stop_reason','user_interrupted','-v7.3');
exportgraphics(fig, trainPng, 'Resolution', 180);
export_model_to_header_single(net, ACTIONS, hdrPath);

fprintf('\nFinal MAT saved: %s\n', finalMat);
fprintf('Header written : %s\n', hdrPath);
fprintf('Run directory  : %s\n', runDir);
fprintf('Stop reason    : %s\n', train_stop_reason);

%% ======================= 15) Greedy rollout =======================
T = 2000;

x_true = double(env.init_true_down());
alpha_bias_roll = 0.0;

y = make_measurement(x_true, meas, alpha_bias_roll);
theta_meas_prev = y(1);
alpha_meas_prev = y(2);

theta_unwrap = theta_meas_prev;
theta_unwrap_prev = theta_unwrap;

thetaDot_est = 0.0;
alphaDot_est = 0.0;

u_cmd_prev = 0.0;

dyn_ctx = init_dyn_rn_ctx(env, x_true, 0.0);

traj_meas = zeros(2,T);
traj_pwm  = zeros(1,T);
traj_done = zeros(1,T);

for t = 1:T
    alpha_ctrl = wrapToPi(alpha_meas_prev);
    s = make_state_sincos( ...
        theta_unwrap, ...
        clip(thetaDot_est, env.thetaDot_lim_base), ...
        alpha_ctrl, ...
        clip(alphaDot_est, env.alphaDot_lim_base));

    [Q,~] = q_forward(net, s);
    [~,aIdx] = max(Q); aIdx = aIdx(1);
    pwm_cmd = double(ACTIONS(aIdx));

    [x_true_next, ~, dyn_ctx] = step_with_delay_dynamics_rn(x_true, u_cmd_prev, pwm_cmd, C, env, dyn_ctx);

    y = make_measurement(x_true_next, meas, alpha_bias_roll);
    theta_meas = y(1);
    alpha_meas = y(2);

    dtheta_wrapped = wrapToPi(theta_meas - theta_meas_prev);
    theta_unwrap   = theta_unwrap + dtheta_wrapped;
    thetaDot_raw   = (theta_unwrap - theta_unwrap_prev) / env.dt;

    dalpha_wrapped = wrapToPi(alpha_meas - alpha_meas_prev);
    alphaDot_raw   = dalpha_wrapped / env.dt;

    thetaDot_est = (1-lpf)*thetaDot_est + lpf*thetaDot_raw;
    alphaDot_est = (1-lpf)*alphaDot_est + lpf*alphaDot_raw;

    traj_meas(:,t) = y;
    traj_pwm(t)    = pwm_cmd;

    alpha_ctrl = wrapToPi(alpha_meas);
    done = (abs(theta_unwrap) > env.theta_lim_true) || ...
           (abs(thetaDot_est) > env.thetaDot_lim_base) || ...
           (abs(alphaDot_est) > env.alphaDot_lim_base) || ...
           any(~isfinite([theta_unwrap, alpha_ctrl, thetaDot_est, alphaDot_est]));
    traj_done(t) = done;

    x_true = x_true_next;
    theta_meas_prev = theta_meas;
    alpha_meas_prev = alpha_meas;

    theta_unwrap_prev = theta_unwrap;
    u_cmd_prev = pwm_cmd;

    if done
        traj_meas = traj_meas(:,1:t);
        traj_pwm  = traj_pwm(1:t);
        traj_done = traj_done(1:t);
        break;
    end
end

tt = (0:size(traj_pwm,2)-1)*env.dt;
figR = figure('Name','Greedy rollout (DYN-RN hard gate 10deg)','Color','w');

subplot(4,1,1);
plot(tt, traj_meas(1,:), 'LineWidth',1.2); grid on; ylabel('\theta meas');

subplot(4,1,2);
plot(tt, wrapToPi(traj_meas(2,:)), 'LineWidth',1.2); grid on; ylabel('\alpha meas');

subplot(4,1,3);
stairs(tt, traj_pwm, 'LineWidth',1.1);
grid on; ylabel('PWM'); xlabel('time (s)');

subplot(4,1,4);
stairs(tt, traj_done, 'LineWidth',1.1);
grid on; ylabel('done'); xlabel('time (s)');
yticks([0 1]);

exportgraphics(figR, rollPng, 'Resolution', 180);
fprintf('Greedy rollout saved: %s\n', rollPng);

%% ====================== Helpers ======================

function s = make_state_sincos(theta_unwrap, thetaDot, alpha_wrap, alphaDot)
th = theta_unwrap;
al = alpha_wrap;
s = single([sin(th); cos(th); thetaDot; sin(al); cos(al); alphaDot]);
end

%% ====================== Dynamics RN context ======================

function dyn_ctx = init_dyn_rn_ctx(env, x_obs, u_obs)
dyn_ctx = struct();

if ~isfield(env, 'use_obs_rn') || ~env.use_obs_rn || isempty(env.obs_rn.model)
    dyn_ctx.seq_len = 0;
    dyn_ctx.theta_prev = [];
    dyn_ctx.theta_dot_prev = [];
    dyn_ctx.alpha_prev = [];
    dyn_ctx.alpha_dot_prev = [];
    dyn_ctx.action_prev = [];
    return;
end

seq_len = env.obs_rn.model.seq_len;
nprev = max(seq_len - 1, 0);

dyn_ctx.seq_len = seq_len;
dyn_ctx.theta_prev     = repmat(double(x_obs(1)), 1, nprev);
dyn_ctx.theta_dot_prev = repmat(double(x_obs(2)), 1, nprev);
dyn_ctx.alpha_prev     = repmat(double(x_obs(3)), 1, nprev);
dyn_ctx.alpha_dot_prev = repmat(double(x_obs(4)), 1, nprev);
dyn_ctx.action_prev    = repmat(double(u_obs),    1, nprev);
end

function dyn_ctx = dyn_rn_ctx_push_current(dyn_ctx, x_obs, u_obs)
if isempty(dyn_ctx) || ~isfield(dyn_ctx,'seq_len') || dyn_ctx.seq_len <= 1
    return;
end

dyn_ctx.theta_prev     = [dyn_ctx.theta_prev(2:end),     double(x_obs(1))];
dyn_ctx.theta_dot_prev = [dyn_ctx.theta_dot_prev(2:end), double(x_obs(2))];
dyn_ctx.alpha_prev     = [dyn_ctx.alpha_prev(2:end),     double(x_obs(3))];
dyn_ctx.alpha_dot_prev = [dyn_ctx.alpha_dot_prev(2:end), double(x_obs(4))];
dyn_ctx.action_prev    = [dyn_ctx.action_prev(2:end),    double(u_obs)];
end

function x_in = dyn_rn_build_input_seq(rn, dyn_ctx, x_obs, u_obs)
seq_len = rn.seq_len;
assert(seq_len >= 1, 'Invalid dyn-rn seq_len');

if seq_len == 1
    X = [double(x_obs(1)), double(x_obs(2)), double(x_obs(3)), double(x_obs(4)), double(u_obs)];
else
    Xprev = [ ...
        dyn_ctx.theta_prev; ...
        dyn_ctx.theta_dot_prev; ...
        dyn_ctx.alpha_prev; ...
        dyn_ctx.alpha_dot_prev; ...
        dyn_ctx.action_prev ...
    ];

    Xcurr = [ ...
        double(x_obs(1)); ...
        double(x_obs(2)); ...
        double(x_obs(3)); ...
        double(x_obs(4)); ...
        double(u_obs) ...
    ];

    Xall = [Xprev, Xcurr];
    X = Xall.';
end

x_in = X.';
x_in = x_in(:);
end

function y = dyn_rn_predict_residual_seq(rn, dyn_ctx, x_obs, u_obs)
x = dyn_rn_build_input_seq(rn, dyn_ctx, x_obs, u_obs);
assert(numel(x)==rn.input_dim, 'DYN-RN input dim mismatch: got %d, expect %d', numel(x), rn.input_dim);

xn = (x - rn.x_mean) ./ rn.x_std;

z1 = rn.W1 * xn + rn.b1;
h1 = max(0, z1);

z2 = rn.W2 * h1 + rn.b2;
h2 = max(0, z2);

yn = rn.W3 * h2 + rn.b3;

y = yn .* rn.y_std + rn.y_mean;
y = double(y(:));
end

function [x_true_next, u_eff, dyn_ctx] = step_with_delay_dynamics_rn(x_true, u_cmd_prev, pwm_cmd, C, env, dyn_ctx)
if ~env.use_delay
    x_phys_next = furuta_step_exact_core(x_true, pwm_cmd, C, env.dt);
    u_eff = pwm_cmd;
else
    t1 = sample_trunc_norm_s(env.delay_mu_ms, env.delay_sig_ms, env.delay_lo_ms, env.delay_hi_ms);
    t1 = min(max(t1, 0), env.dt);

    x_mid = furuta_step_exact_core(x_true, u_cmd_prev, C, t1);
    x_phys_next = furuta_step_exact_core(x_mid, pwm_cmd, C, env.dt - t1);

    u_eff = (t1/env.dt)*u_cmd_prev + ((env.dt - t1)/env.dt)*pwm_cmd;
end

x_true_next = x_phys_next;

if isfield(env, 'use_obs_rn') && env.use_obs_rn && ~isempty(env.obs_rn.model)
    runtime_enabled = true;
    if isfield(env.obs_rn, 'runtime_enabled')
        runtime_enabled = env.obs_rn.runtime_enabled;
    end

    if runtime_enabled
        alpha_gate_ref = wrapToPi(x_true(3));
        gate_on = abs(alpha_gate_ref) <= deg2rad(env.obs_rn.alpha_gate_deg);

        if gate_on
            res = dyn_rn_predict_residual_seq(env.obs_rn.model, dyn_ctx, x_true, u_eff);

            switch env.obs_rn.mode
                case 'alpha_only'
                    if env.obs_rn.model.output_dim < 1
                        error('DYN-RN alpha_only requires output_dim >= 1');
                    end
                    dalpha = env.obs_rn.gain_alpha * res(1);
                    dalpha = clip(dalpha, env.obs_rn.clip_alpha_corr);

                    x_true_next(3) = wrapToPi(x_phys_next(3) + dalpha);

                case 'alpha_alpha_dot'
                    if env.obs_rn.model.output_dim < 2
                        error('DYN-RN alpha_alpha_dot requires output_dim >= 2');
                    end
                    dalpha     = env.obs_rn.gain_alpha     * res(1);
                    dalpha_dot = env.obs_rn.gain_alpha_dot * res(2);

                    dalpha     = clip(dalpha,     env.obs_rn.clip_alpha_corr);
                    dalpha_dot = clip(dalpha_dot, env.obs_rn.clip_alpha_dot_corr);

                    x_true_next(3) = wrapToPi(x_phys_next(3) + dalpha);
                    x_true_next(4) = x_phys_next(4) + dalpha_dot;

                case 'full_state'
                    if env.obs_rn.model.output_dim < 4
                        error('DYN-RN full_state requires output_dim >= 4');
                    end

                    dtheta     = env.obs_rn.gain_theta     * res(1);
                    dtheta_dot = env.obs_rn.gain_theta_dot * res(2);
                    dalpha     = env.obs_rn.gain_alpha     * res(3);
                    dalpha_dot = env.obs_rn.gain_alpha_dot * res(4);

                    dtheta     = clip(dtheta,     env.obs_rn.clip_theta_corr);
                    dtheta_dot = clip(dtheta_dot, env.obs_rn.clip_theta_dot_corr);
                    dalpha     = clip(dalpha,     env.obs_rn.clip_alpha_corr);
                    dalpha_dot = clip(dalpha_dot, env.obs_rn.clip_alpha_dot_corr);

                    x_true_next(1) = x_phys_next(1) + dtheta;
                    x_true_next(2) = x_phys_next(2) + dtheta_dot;
                    x_true_next(3) = wrapToPi(x_phys_next(3) + dalpha);
                    x_true_next(4) = x_phys_next(4) + dalpha_dot;

                otherwise
                    error('Unknown env.obs_rn.mode = %s', env.obs_rn.mode);
            end
        end
    end
end

% history 里存的是“当前状态 + 当前动作”，用于预测下一步残差
dyn_ctx = dyn_rn_ctx_push_current(dyn_ctx, x_true, u_eff);
end

function y = make_measurement(x_true, meas, alpha_bias)
theta = x_true(1);
alpha = x_true(3);
if isfield(meas,'use_noise') && meas.use_noise
    theta = theta + meas.theta_sigma*randn();
    alpha = alpha + meas.alpha_sigma*randn() + alpha_bias;
end
y = [theta; alpha];
end

function v = clip(v, lim)
v = max(-lim, min(lim, v));
end

function [aIdx, pwm_cmd] = select_action(net_snap, s, eps, actDim, ACTIONS)
if rand < eps
    aIdx = randi(actDim);
else
    [Q,~] = q_forward(net_snap, s);
    [~,aIdx] = max(Q); aIdx = aIdx(1);
end
pwm_cmd = double(ACTIONS(aIdx));
end

function B = replay_init(stateDim, bufSize)
B.S  = zeros(stateDim,bufSize,'single');
B.A  = zeros(1,bufSize,'uint16');
B.R  = zeros(1,bufSize,'single');
B.S2 = zeros(stateDim,bufSize,'single');
B.D  = false(1,bufSize);
B.ptr = 1; B.n = 0; B.buf = bufSize; B.stateDim = stateDim;
end

function B = replay_push(B, s, aIdx, r, s2, done)
k = B.ptr;
B.S(:,k)  = s;
B.A(k)    = uint16(aIdx);
B.R(k)    = single(r);
B.S2(:,k) = s2;
B.D(k)    = done;
B.ptr = k + 1; if B.ptr > B.buf, B.ptr = 1; end
B.n = min(B.n + 1, B.buf);
end

function [net,opt] = dqn_learn_one(net, tgt, B, conf, opt)
idx = randi(B.n, [1, conf.batch]);
Sz  = B.S(:,idx);
Az  = double(B.A(idx));
Rz  = double(B.R(idx));
Sz2 = B.S2(:,idx);
Dz  = B.D(idx);

[Qcur, Cc] = q_forward(net, Sz);
Qa = Qcur(sub2ind(size(Qcur), Az, 1:conf.batch));

[Qn, ~] = q_forward(tgt, Sz2);
Qmax = max(Qn, [], 1);

Ym = Rz + conf.gamma .* Qmax .* (~Dz);

diff = Qa - Ym;

delta = 1.0;
grad = diff;
big = abs(diff) > delta;
grad(big) = delta .* sign(diff(big));

dQ = zeros(size(Qcur), 'like', Qcur);
dQ(sub2ind(size(Qcur), Az, 1:conf.batch)) = grad ./ double(conf.batch);

G = q_back(net, Cc, dQ);
[net, opt] = adam(net, G, opt);
end

function s2 = furuta_step_exact_core(s, pwm, C, dt)
s2 = furuta_step_rk4_core(s, pwm, C, dt);
end

function s2 = furuta_step_rk4_core(s, pwm, C, dt)
k1 = furuta_f(s, pwm, C);
k2 = furuta_f(s + 0.5*dt*k1, pwm, C);
k3 = furuta_f(s + 0.5*dt*k2, pwm, C);
k4 = furuta_f(s + dt*k3, pwm, C);
s2 = s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4);
end

function ds = furuta_f(s, pwm, C)
th=s(1); thd=s(2); al=s(3); ald=s(4);

g=C.g; ct=C.c_theta; ca=C.c_alpha; K1=C.K1; K2=C.K2;
m1=C.m1; m2=C.m2; l1=C.l1; l1c=C.l1cg; l2c=C.l2cg;
I1z=C.I1z; I2x=C.I2x; I2y=C.I2y; I2z=C.I2z;

sA=sin(al); cA=cos(al); sAcA=sA.*cA;
A = m1*l1c^2 + I1z + m2*l1^2 + (m2*l2c^2 + I2z).*sA.^2 + I2y.*cA.^2;
B =  m2*l1*l2c.*cA;
Cc= -m2*l1*l2c.*sA;
D = 2*(I2z + m2*l2c^2 - I2y).*sAcA;

E =  K1*pwm - K2*thd - ct*thd;

F = -(m2*l2c^2 + I2x);
G = -(m2*l1*l2c.*cA);
H =  (m2*l2c^2 - I2y + I2z).*sAcA;
K =  m2*g*l2c.*sA;
L =  ca*ald;

den1 = A*F - G*B;
thdd = ((-F*Cc).*ald.^2 + (-F*D).*ald.*thd + (B*H).*thd.^2 + (B*K + F*E - B*L)) ./ den1;

den2 = G*B - A*F;
aldd = ((-G*Cc).*ald.^2 + (-G*D).*ald.*thd + (A*H).*thd.^2 + (A*K + G*E - A*L)) ./ den2;

ds = [thd; thdd; ald; aldd];
end

function [Q,C] = q_forward(net,S)
Z1=net.W1*S+net.b1; H1=max(0,Z1);
Z2=net.W2*H1+net.b2; H2=max(0,Z2);
Q =net.W3*H2+net.b3;
C=struct('S',S,'Z1',Z1,'H1',H1,'Z2',Z2,'H2',H2);
end

function G = q_back(net,C,dQ)
dW3 = dQ*C.H2';
db3 = sum(dQ,2);

dH2 = net.W3'*dQ;
dZ2 = dH2;
dZ2(C.Z2<=0)=0;

dW2 = dZ2*C.H1';
db2 = sum(dZ2,2);

dH1 = net.W2'*dZ2;
dZ1 = dH1;
dZ1(C.Z1<=0)=0;

dW1 = dZ1*C.S';
db1 = sum(dZ1,2);

G=struct('dW1',dW1,'db1',db1,'dW2',dW2,'db2',db2,'dW3',dW3,'db3',db3);
end

function [net,opt] = adam(net,G,opt)
opt.t=opt.t+1; a=opt.lr; b1=opt.beta1; b2=opt.beta2; e=opt.eps;
[net.W1,opt.mW1,opt.vW1]=adam_u(net.W1,G.dW1,opt.mW1,opt.vW1,a,b1,b2,e,opt.t);
[net.b1,opt.mb1,opt.vb1]=adam_u(net.b1,G.db1,opt.mb1,opt.vb1,a,b1,b2,e,opt.t);
[net.W2,opt.mW2,opt.vW2]=adam_u(net.W2,G.dW2,opt.mW2,opt.vW2,a,b1,b2,e,opt.t);
[net.b2,opt.mb2,opt.vb2]=adam_u(net.b2,G.db2,opt.mb2,opt.vb2,a,b1,b2,e,opt.t);
[net.W3,opt.mW3,opt.vW3]=adam_u(net.W3,G.dW3,opt.mW3,opt.vW3,a,b1,b2,e,opt.t);
[net.b3,opt.mb3,opt.vb3]=adam_u(net.b3,G.db3,opt.mb3,opt.vb3,a,b1,b2,e,opt.t);
end

function [W,m,v] = adam_u(W,g,m,v,a,b1,b2,e,t)
m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*(g.^2);
mhat=m/(1-b1^t); vhat=v/(1-b2^t);
W=W - a*mhat./(sqrt(vhat)+e);
end

function d = getDesktopDir()
if ispc
    d = fullfile(getenv('USERPROFILE'),'Desktop');
else
    d = fullfile(getenv('HOME'),'Desktop');
end
end

function t_s = sample_trunc_norm_s(mu_ms, sig_ms, lo_ms, hi_ms)
mu = mu_ms * 1e-3;
sg = max(1e-9, sig_ms * 1e-3);
lo = lo_ms * 1e-3;
hi = hi_ms * 1e-3;

for k = 1:1000
    x = mu + sg*randn();
    if x >= lo && x <= hi
        t_s = x;
        return;
    end
end
t_s = min(max(mu, lo), hi);
end

function export_model_to_header_single(net, ACTIONS, outHeaderPath)
stateDim = size(net.W1,2);
h1 = size(net.W1,1);
h2 = size(net.W2,1);
actDim = size(net.W3,1);

fid = fopen(outHeaderPath,'w');
if fid<0, error('Cannot open output header file.'); end

fprintf(fid, '#pragma once\n');
fprintf(fid, '// Auto-generated DQN weights (single net)\n');
fprintf(fid, '// DO NOT EDIT MANUALLY\n\n');
fprintf(fid, '#include <stdint.h>\n\n');

fprintf(fid, '#define DQN_STATE_DIM %d\n', stateDim);
fprintf(fid, '#define DQN_H1 %d\n', h1);
fprintf(fid, '#define DQN_H2 %d\n', h2);
fprintf(fid, '#define DQN_ACT_DIM %d\n\n', actDim);

writeFloat2D(fid, 'DQN_W1', net.W1);
writeFloat1D(fid, 'DQN_b1', net.b1);
writeFloat2D(fid, 'DQN_W2', net.W2);
writeFloat1D(fid, 'DQN_b2', net.b2);
writeFloat2D(fid, 'DQN_W3', net.W3);
writeFloat1D(fid, 'DQN_b3', net.b3);

ACTIONS = int16(ACTIONS(:));
fprintf(fid, '\nstatic const int16_t DQN_ACTIONS[DQN_ACT_DIM] = {');
for i=1:numel(ACTIONS)
    if i<numel(ACTIONS), fprintf(fid, '%d, ', ACTIONS(i));
    else,               fprintf(fid, '%d', ACTIONS(i)); end
end
fprintf(fid, '};\n\n');

fclose(fid);
end

function writeFloat2D(fid, name, M)
M = double(M);
[r,c] = size(M);
fprintf(fid, 'static const float %s[%d][%d] = {\n', name, r, c);
for i=1:r
    fprintf(fid, '  {');
    for j=1:c
        if j<c, fprintf(fid, '%.9g, ', M(i,j));
        else,   fprintf(fid, '%.9g', M(i,j)); end
    end
    if i<r, fprintf(fid, '},\n');
    else,   fprintf(fid, '}\n'); end
end
fprintf(fid, '};\n');
end

function writeFloat1D(fid, name, v)
v = double(v(:));
n = numel(v);
fprintf(fid, 'static const float %s[%d] = {', name, n);
for i=1:n
    if i<n, fprintf(fid, '%.9g, ', v(i));
    else,   fprintf(fid, '%.9g', v(i)); end
end
fprintf(fid, '};\n');
end

%% ====================== RN Loader ======================
function rn = load_rn_bundle_from_mat(matPath)
assert(exist(matPath, 'file') == 2, 'RN .mat file not found: %s', matPath);

S = load(matPath);

rn.seq_len = 1;
if isfield(S, 'seq_len')
    rn.seq_len = double(S.seq_len(1));
end

rn.input_dim = size(S.W1, 2);
if isfield(S, 'input_dim')
    rn.input_dim = double(S.input_dim(1));
end

rn.output_dim = size(S.W3, 1);
if isfield(S, 'output_dim')
    rn.output_dim = double(S.output_dim(1));
end

rn.x_mean = double(S.x_mean(:));
rn.x_std  = double(S.x_std(:));
rn.y_mean = double(S.y_mean(:));
rn.y_std  = double(S.y_std(:));

rn.W1 = double(S.W1);
rn.b1 = double(S.b1(:));
rn.W2 = double(S.W2);
rn.b2 = double(S.b2(:));
rn.W3 = double(S.W3);
rn.b3 = double(S.b3(:));

rn.output_names = strings(0,1);
if isfield(S, 'output_names')
    rn.output_names = string(S.output_names(:));
end

if ~(rn.output_dim == 1 || rn.output_dim == 2 || rn.output_dim == 4)
    error('Current RN integration only supports output_dim = 1, 2, or 4. Got output_dim = %d', rn.output_dim);
end

fprintf('[RN] Loaded weights from MAT:\n');
fprintf('     seq_len   = %d\n', rn.seq_len);
fprintf('     input_dim = %d\n', rn.input_dim);
fprintf('     output_dim= %d\n', rn.output_dim);
fprintf('     W1 = [%d x %d]\n', size(rn.W1,1), size(rn.W1,2));
fprintf('     W2 = [%d x %d]\n', size(rn.W2,1), size(rn.W2,2));
fprintf('     W3 = [%d x %d]\n', size(rn.W3,1), size(rn.W3,2));
if ~isempty(rn.output_names)
    fprintf('     output_names = ');
    disp(rn.output_names(:).');
end
end