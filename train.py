import os
import random
import shutil
import time
from pathlib import Path

import fym
import gym
import matplotlib.pyplot as plt
import numpy as np
import ray
import ray.tune as tune
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from ray.tune import ExperimentAnalysis
from scipy.spatial.transform import Rotation
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from torch.utils.tensorboard.writer import SummaryWriter

from src.env import QuadEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.SafeLoader)


def make_env(config):
    def wrapper():
        # make env
        env = QuadEnv(config["env_config"])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if config["fault_occurs"]:
            # set LoE fault (task)
            task = env.plant.get_task(rf=config["rf"], random=False)
        else:
            task = env.plant.get_task(random=False)
        env.plant.set_task(task)
        return env

    return wrapper


class Actor(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.action_scale = torch.FloatTensor(
            (env.action_space.high - env.action_space.low) / 2.0
        )
        self.action_bias = torch.FloatTensor(
            (env.action_space.high + env.action_space.low) / 2.0
        )

        self.log_std_min = config["log_std_min"]
        self.log_std_max = config["log_std_max"]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod()
            + np.prod(env.action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAC(nn.Module):
    def __init__(self, envs, config):
        super().__init__()
        self.actor = Actor(envs, config)

        # setup networks
        self.qf1 = SoftQNetwork(envs)
        self.qf2 = SoftQNetwork(envs)
        self.qf1_target = SoftQNetwork(envs)
        self.qf2_target = SoftQNetwork(envs)

        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # setup optimizers
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=config["q_lr"],
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()),
            lr=config["policy_lr"],
        )

        # with autotune = True
        self.target_entropy = -torch.prod(torch.Tensor(envs.action_space.shape)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = optim.Adam([self.log_alpha], lr=config["q_lr"])

    def get_action(self, obs, deterministic=False):
        assert isinstance(obs, np.ndarray)
        if obs.ndim == 1:
            action, _, mean = self.actor(torch.Tensor(obs).to(device)[None])
            action = action.detach().cpu().numpy()[0]
            mean = mean.detach().cpu().numpy()[0]
        elif obs.ndim == 2:
            action, _, mean = self.actor(torch.Tensor(obs).to(device))
            action = action.detach().cpu().numpy()
            mean = mean.detach().cpu().numpy()
        else:
            raise ValueError

        if deterministic:
            return mean
        else:
            return action


def sac_trainable(config, checkpoint_dir=None):
    """Ray.tune functional trainable method"""

    writer = SummaryWriter("runs")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )

    # seeding
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # make vectorized multiple envs for averaging the results
    envs = DummyVecEnv([make_env(config) for _ in range(config["n_envs"])])

    # setup policy
    policy = SAC(envs, config)
    policy.to(device)

    start = 0

    # load from the past checkpoint
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        state = torch.load(checkpoint)
        start = state["global_step"] + 1
        policy.load_state_dict(state["policy"])

    # gamma from mgamma
    gamma = 1 - config["mgamma"]

    # create a replay buffer for off-policy RLs
    rb = ReplayBuffer(
        buffer_size=int(config["buffer_size"]),
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        device=device,
        n_envs=config["n_envs"],
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(start, int(config["total_timesteps"])):
        # ALGO LOGIC: put action logic here
        if global_step < config["learning_starts"]:
            actions = np.array(
                [envs.action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions = policy.get_action(obs)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > config["learning_starts"]:
            data = rb.sample(int(config["batch_size"]))
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = policy.actor(
                    data.next_observations
                )
                qf1_next_target = policy.qf1_target(
                    data.next_observations, next_state_actions
                )
                qf2_next_target = policy.qf2_target(
                    data.next_observations, next_state_actions
                )
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - policy.alpha * next_state_log_pi
                )
                next_q_value = data.rewards.flatten() + gamma * (
                    min_qf_next_target
                ).view(-1)

            qf1_a_values = policy.qf1(data.observations, data.actions).view(-1)
            qf2_a_values = policy.qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            policy.q_optimizer.zero_grad()
            qf_loss.backward()
            policy.q_optimizer.step()

            if (
                global_step % config["policy_frequency"] == 0
            ):  # TD 3 Delayed update support
                for _ in range(
                    config["policy_frequency"]
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = policy.actor(data.observations)
                    qf1_pi = policy.qf1(data.observations, pi)
                    qf2_pi = policy.qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((policy.alpha * log_pi) - min_qf_pi).mean()

                    policy.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    policy.actor_optimizer.step()

                    # with autotune = True
                    with torch.no_grad():
                        _, log_pi, _ = policy.actor(data.observations)
                    alpha_loss = (
                        -policy.log_alpha * (log_pi + policy.target_entropy)
                    ).mean()

                    policy.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    policy.a_optimizer.step()
                    policy.alpha = policy.log_alpha.exp().item()

            # update the target networks
            if global_step % config["target_network_frequency"] == 0:
                for param, target_param in zip(
                    policy.qf1.parameters(), policy.qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        config["tau"] * param.data
                        + (1 - config["tau"]) * target_param.data
                    )
                for param, target_param in zip(
                    policy.qf2.parameters(), policy.qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        config["tau"] * param.data
                        + (1 - config["tau"]) * target_param.data
                    )

            # write to writer
            if global_step % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", policy.alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                # with autotune = True
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                for info in infos:
                    if "episode" in info.keys():
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )
                        break

            # Save the model into a checkpoint
            if global_step % (config["total_timesteps"] // 20) == 0:
                with tune.checkpoint_dir(step=global_step) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(
                        {"policy": policy.state_dict(), "global_step": global_step},
                        path,
                    )

    envs.close()
    writer.close()


def hyperparam_tune():
    expdir = Path("exp/origin-hover/SAC/tune")
    if expdir.exists():
        shutil.rmtree(expdir)
    expdir.mkdir(exist_ok=True, parents=True)

    # setup tune search space
    config = {
        "seed": tune.randint(0, 100000),
        "mgamma": tune.loguniform(1e-1, 1e-3),  # gamma = 1 - mgamma
        "q_lr": tune.loguniform(1e-4, 1e-2),
        "policy_lr": tune.loguniform(1e-4, 1e-2),
        "env_config": CONFIG["env_config"],
        **CONFIG["tune"]["config"],
    }
    tune_config = {
        "config": config,
        "num_samples": 32,
        "local_dir": str(expdir),
    }

    # -- CASE: NORMAL
    config["fault_occurs"] = False
    tune_config["name"] = "normal"

    ray.init()
    tune.run(sac_trainable, **tune_config)
    ray.shutdown()

    # -- CASE: FAULT
    config["fault_occurs"] = True
    tune_config["name"] = "fault"

    ray.init()
    tune.run(sac_trainable, **tune_config)
    ray.shutdown()


def train_sac():
    # best hyperparamters finding from ``hyperparam_tune()``

    expdir = Path("exp/origin-hover/SAC/train")
    if expdir.exists():
        shutil.rmtree(expdir)
    expdir.mkdir(exist_ok=True, parents=True)

    # -- CASE: NORMAL
    config = {
        "seed": tune.randint(0, 100000),
        "fault_occurs": False,
        "mgamma": 0.0156,
        "policy_lr": 0.000222,
        "q_lr": 0.00146,
        "env_config": CONFIG["env_config"],
        **CONFIG["tune"]["config"],
    }
    tune_config = {
        "config": config | {"total_timesteps": 100000},
        "num_samples": 10,
        "local_dir": str(expdir),
        "name": "normal",
    }

    ray.init()
    tune.run(sac_trainable, **tune_config)
    ray.shutdown()

    # -- CASE: FAULT
    config = {
        "seed": tune.randint(0, 100000),
        "fault_occurs": True,
        "mgamma": 0.0914,
        "policy_lr": 0.00504,
        "q_lr": 0.00131,
        "env_config": CONFIG["env_config"],
        **CONFIG["tune"]["config"],
    }
    tune_config = {
        "config": config | {"total_timesteps": 100000},
        "num_samples": 10,
        "local_dir": str(expdir),
        "name": "fault",
    }

    ray.init()
    tune.run(sac_trainable, **tune_config)
    ray.shutdown()


# -- TEST


def get_trial(fault_occurs=False, nid=None):
    # get last checkpoint of experiments
    exppath = Path("exp/origin-hover/SAC/train")
    if fault_occurs:
        exppath /= "fault"
    else:
        exppath /= "normal"
    analysis = ExperimentAnalysis(str(exppath))
    assert analysis.trials is not None

    # get trials and a trial
    trials = sorted(analysis.trials, key=lambda t: t.trial_id)
    trial = trials[nid or 0]
    return trial


def test_sac():
    # get a checkpoint
    trial = get_trial(fault_occurs=True)
    config = trial.config
    assert trial.logdir is not None
    testpath = Path(trial.logdir) / "test-flight.h5"

    # get the env used for training
    # remedy for new config
    config["env_config"]["plant_config"] = CONFIG["env_config"]["plant_config"]
    config["env_config"]["fkw"]["max_t"] = 20
    config["env_config"]["outer_loop"] = "PID"
    config["env_config"]["perturb_scale"] = CONFIG["env_config"]["perturb_scale"]
    env = make_env(config)()
    # add a fym logger to the env
    env.env.logger = fym.Logger(path=testpath)

    # make a policy
    policy = SAC(env, config)
    policy.to(device)
    # load policy from the checkpoint
    checkpoint = os.path.join(trial.checkpoint.value, "checkpoint")
    state = torch.load(checkpoint)
    policy.load_state_dict(state["policy"])
    policy.eval()

    # start testing
    episode_reward = 0
    done = False
    obs = env.reset(mode="initial")

    while not done:
        env.render()
        action = policy.get_action(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

    env.close()

    # -- Plot
    data = fym.load(testpath)
    if type(data) is not dict:
        raise ValueError("data should be dict")

    fig, axes = plt.subplots(4, 2, figsize=(12, 8), squeeze=False, sharex=True)

    data["plant"]

    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"].squeeze(-1))
    ax.set_ylabel("Position, m")
    ax.legend([r"$x$", r"$y$", r"$z$"])

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["vel"].squeeze(-1))
    ax.set_ylabel("Velocity, m/s")
    ax.legend([r"$v_x$", r"$v_y$", r"$v_z$"])

    ax = axes[2, 0]
    angles = Rotation.from_matrix(data["plant"]["R"]).as_euler("ZYX")[:, ::-1]
    ax.plot(data["t"], np.rad2deg(angles))
    ax.set_ylabel("Angles, deg")
    ax.legend([r"$\phi$", r"$\theta$", r"$\psi$"])

    ax = axes[3, 0]
    ax.plot(data["t"], data["plant"]["omega"].squeeze(-1))
    ax.set_ylabel("Omega, rad/sec")
    ax.legend([r"$p$", r"$q$", r"$r$"])

    ax.set_xlabel("Time, sec")

    ax = axes[0, 1]
    ax.plot(data["t"], data["rotorfs_cmd"].squeeze(-1)[:, 0], "r--")
    ax.plot(data["t"], data["plant_info"]["rotorfs"].squeeze(-1)[:, 0], "k-")
    ax.set_ylabel("Rotor 1 thrust, N")

    ax = axes[1, 1]
    ax.plot(data["t"], data["rotorfs_cmd"].squeeze(-1)[:, 1], "r--")
    ax.plot(data["t"], data["plant_info"]["rotorfs"].squeeze(-1)[:, 1], "k-")
    ax.set_ylabel("Rotor 2 thrust, N")

    ax = axes[2, 1]
    ax.plot(data["t"], data["rotorfs_cmd"].squeeze(-1)[:, 2], "r--")
    ax.plot(data["t"], data["plant_info"]["rotorfs"].squeeze(-1)[:, 2], "k-")
    ax.set_ylabel("Rotor 3 thrust, N")

    ax = axes[3, 1]
    ax.plot(data["t"], data["rotorfs_cmd"].squeeze(-1)[:, 3], "r--")
    ax.plot(data["t"], data["plant_info"]["rotorfs"].squeeze(-1)[:, 3], "k-")
    ax.set_ylabel("Rotor 4 thrust, N")

    ax.set_xlabel("Time, sec")

    fig.tight_layout()
    # fig.savefig(testdir / "hist.pdf", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    # hyperparam_tune()
    train_sac()
    # test_sac()
