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
from torch.utils.tensorboard.writer import SummaryWriter

from src.env import QuadEnv

with open("config.yaml", "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.SafeLoader)


class Actor(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.action_scalqe = torch.FloatTensor(
            (env.single_action_space.high - env.single_action_space.low) / 2.0
        )
        self.action_bias = torch.FloatTensor(
            (env.single_action_space.high + env.single_action_space.low) / 2.0
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
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
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


def make_env(config):
    def wrapper():
        # make env
        env = QuadEnv(config["env_config"])
        # set LoE fault (task)
        task = env.get_task(rf=config["rf"], random=False)
        env.set_task(task)
        return env

    return wrapper


def train_sac(config):
    run_name = f"{config['exp_name']}__{config['seed']}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make vectorized multiple envs for averaging the results
    envs = gym.vector.SyncVectorEnv(
        [make_env(config) for _ in range(config["num_envs_for_each_trial"])]
    )

    actor = Actor(envs, config["actor_config"])

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
        torch.Tensor(envs.single_action_space.shape).to(device)
    ).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=config["q_lr"])

    rb = ReplayBuffer(
        config["buffer_size"],
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(config["total_timesteps"]):
        # ALGO LOGIC: put action logic here
        if global_step < config["learning_starts"]:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                break

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
            data = rb.sample(config["batch_size"])
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
                ) * config["gamma"] * (min_qf_next_target).view(-1)

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

    envs.close()
    writer.close()


def main():
    # seeding
    seed = CONFIG["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # setup tune search space
    config = {
        "exp_name": "SAC",
        "seed": 0,
        "gamma": tune.choice([0.999, 0.995, 0.99, 0.98, 0.97]),
        "q_lr": tune.loguniform(1e-4, 1e-2),
        "policy_lr": tune.loguniform(1e-4, 1e-2),
        "fault_occurs": tune.grid_search([True, False]),
        "rf": np.array([1, 1, 1, 0]),  # rotor fault
        "num_envs_for_each_trial": 5,
        "buffer_size": 1e6,
        "total_timesteps": 1e6,
        "learning_starts": 5e3,  # timestep to start learning
        "policy_frequency": 2,  # the frequency of training policy (delayed)
        "target_network_frequency": 1,  # the frequency of updates for the target nerworks
        "autotune": True,  # automatic tuning of the entropy coefficient
        "alpha": 0.2,  # entropy regularization coefficient
        "env_config": CONFIG["env_config"],
        "log_std_max": 2,
        "log_std_min": -5,
        "tau": 0.005,  # target smoothing coefficient
    }
    tune_config = {
        "config": config,
        "num_samples": 10,
        "local_dir": "./ray-results",
    }

    # tune
    logger.info("Start tune.run")

    ray.init()
    analysis = tune.run(train_sac, **tune_config)
    ray.shutdown()

    logger.info("Finish tune.run")


if __name__ == "__main__":
    main()
