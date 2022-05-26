import os
import random
import time

import gym
import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from loguru import logger
from ray import tune
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from torch.utils.tensorboard.writer import SummaryWriter

from src.env import QuadEnv

with open("config.yaml", "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.SafeLoader)


def make_env(config):
    def wrapper():
        # make env
        env = QuadEnv(config["env_config"])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if config["fault_occurs"]:
            # set LoE fault (task)
            task = env.get_task(rf=config["rf"], random=False)
        else:
            task = env.get_task(random=False)
        env.set_task(task)
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
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make vectorized multiple envs for averaging the results
    envs = DummyVecEnv([make_env(config) for _ in range(config["n_envs"])])

    actor = Actor(envs, config)

    # setup networks
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # setup optimizers
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=config["q_lr"]
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=config["policy_lr"])

    # with autotune = True
    target_entropy = -torch.prod(
        torch.Tensor(envs.action_space.shape).to(device)
    ).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=config["q_lr"])

    # load from the past checkpoint
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        state = torch.load(checkpoint)
        actor.load_state_dict(state["actor"])
        qf1.load_state_dict(state["qf1"])
        qf2.load_state_dict(state["qf2"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        q_optimizer.load_state_dict(state["q_optimizer"])
        a_optimizer.load_state_dict(state["a_optimizer"])

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
    for global_step in range(int(config["total_timesteps"])):
        # ALGO LOGIC: put action logic here
        if global_step < config["learning_starts"]:
            actions = np.array(
                [envs.action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

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
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if (
                global_step % config["policy_frequency"] == 0
            ):  # TD 3 Delayed update support
                for _ in range(
                    config["policy_frequency"]
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # with autotune = True
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.observations)
                    alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % config["target_network_frequency"] == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        config["tau"] * param.data
                        + (1 - config["tau"]) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
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
                writer.add_scalar("losses/alpha", alpha, global_step)
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
                        {
                            "actor": actor.state_dict(),
                            "qf1": qf1.state_dict(),
                            "qf2": qf2.state_dict(),
                            "actor_optimizer": actor_optimizer.state_dict(),
                            "q_optimizer": q_optimizer.state_dict(),
                            "a_optimizer": a_optimizer.state_dict(),
                            "global_step": global_step,
                        },
                        path,
                    )

    envs.close()
    writer.close()


def hyperparam_tune():
    # setup tune search space
    config = {
        "seed": tune.randint(0, 100000),
        "mgamma": tune.loguniform(1e-1, 1e-3),  # gamma = 1 - mgamma
        "q_lr": tune.loguniform(1e-4, 1e-2),
        "policy_lr": tune.loguniform(1e-4, 1e-2),
        "fault_occurs": tune.grid_search([True, False]),
        "env_config": CONFIG["env_config"],
        **CONFIG["tune"]["config"],
    }
    tune_config = {
        "config": config,
        "num_samples": 32,
        "local_dir": "./exp/origin-hover/SAC",
        "name": "hyperparam-tune",
    }

    # tune
    ray.init()
    tune.run(sac_trainable, **tune_config)
    ray.shutdown()


def train_sac():
    # best hyperparamters finding from ``hyperparam_tune()``
    config = {
        "seed": tune.randint(0, 100000),
        "fault_occurs": False,
        "mgamma": 0.00518,
        "policy_lr": 0.000613,
        "q_lr": 0.00471,
        "env_config": CONFIG["env_config"],
        **CONFIG["tune"]["config"],
    }
    tune_config = {
        "config": config,
        "num_samples": 10,
        "local_dir": "./exp/origin-hover/SAC/train",
        "name": "normal",
    }

    ray.init()
    tune.run(sac_trainable, **tune_config)
    ray.shutdown()

    config = {
        "seed": tune.randint(0, 100000),
        "fault_occurs": True,
        "mgamma": 0.032,
        "policy_lr": 0.00392,
        "q_lr": 0.00957,
        "env_config": CONFIG["env_config"],
        **CONFIG["tune"]["config"],
    }
    tune_config = {
        "config": config,
        "num_samples": 10,
        "local_dir": "./exp/origin-hover/SAC/train",
        "name": "fault",
    }

    ray.init()
    tune.run(sac_trainable, **tune_config)
    ray.shutdown()


if __name__ == "__main__":
    # hyperparam_tune()
    train_sac()
