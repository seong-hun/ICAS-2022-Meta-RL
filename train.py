import argparse
import random
import time
from datetime import datetime
from itertools import count
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
from loguru import logger
from ray.tune.suggest.variant_generator import generate_variants
from scipy.spatial.transform import Rotation
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from torch.utils.tensorboard.writer import SummaryWriter

from src.env import QuadEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# SAC {{{
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
            lr=config["hyperparameters"]["q_lr"],
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()),
            lr=config["hyperparameters"]["policy_lr"],
        )

        # with autotune = True
        self.target_entropy = -torch.prod(torch.Tensor(envs.action_space.shape)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = optim.Adam(
            [self.log_alpha], lr=config["hyperparameters"]["q_lr"]
        )

        self.config = config
        self.learn_count = 0

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

    def learn(self, data):
        config = self.config

        gamma = 1 - config["hyperparameters"]["mgamma"]

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor(
                data.next_observations
            )
            qf1_next_target = self.qf1_target(
                data.next_observations, next_state_actions
            )
            qf2_next_target = self.qf2_target(
                data.next_observations, next_state_actions
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            if config["env_config"]["reward"]["boundsout"] is None:
                next_q_value = data.rewards.flatten() + gamma * (
                    min_qf_next_target
                ).view(-1)
            else:
                next_q_value = data.rewards.flatten() + gamma * (
                    1 - data.dones.flatten()
                ) * (min_qf_next_target).view(-1)

        qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
        qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if (
            self.learn_count % config["policy_frequency"] == 0
        ):  # TD 3 Delayed update support
            for _ in range(
                config["policy_frequency"]
            ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi, _ = self.actor(data.observations)
                qf1_pi = self.qf1(data.observations, pi)
                qf2_pi = self.qf2(data.observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                self.actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                self.actor_loss.backward()
                self.actor_optimizer.step()

                # with autotune = True
                with torch.no_grad():
                    _, log_pi, _ = self.actor(data.observations)
                self.alpha_loss = (
                    -self.log_alpha * (log_pi + self.target_entropy)
                ).mean()

                self.a_optimizer.zero_grad()
                self.alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha = self.log_alpha.exp().item()

        # update the target networks
        if self.learn_count % config["target_network_frequency"] == 0:
            for param, target_param in zip(
                self.qf1.parameters(), self.qf1_target.parameters()
            ):
                target_param.data.copy_(
                    config["tau"] * param.data + (1 - config["tau"]) * target_param.data
                )
            for param, target_param in zip(
                self.qf2.parameters(), self.qf2_target.parameters()
            ):
                target_param.data.copy_(
                    config["tau"] * param.data + (1 - config["tau"]) * target_param.data
                )

        self.learn_count += 1

        # write to writer
        logs = {
            "losses/qf1_values": qf1_a_values.mean().item(),
            "losses/qf2_values": qf2_a_values.mean().item(),
            "losses/qf1_loss": qf1_loss.item(),
            "losses/qf2_loss": qf2_loss.item(),
            "losses/qf_loss": qf_loss.item() / 2.0,
            "losses/actor_loss": self.actor_loss.item(),
            "losses/alpha": self.alpha,
            "losses/alpha_loss": self.alpha_loss.item(),
        }

        return logs


# }}}


class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(*self.X.shape)
        self.X = self.X + dx
        return self.X


class LinearPolicy:
    def __init__(self, K, obs0, action0, noisy=False):
        self.K = K
        self.obs0 = obs0
        self.action0 = action0
        self.noise = OUNoise(4, sigma=0.05) if noisy else 0

    def reset(self):
        if isinstance(self.noise, OUNoise):
            self.noise.reset()

    def get_action(self, obs):
        if isinstance(self.noise, OUNoise):
            noise = self.noise.sample()
        else:
            noise = self.noise

        if obs.ndim == 1:
            action = (
                self.action0 - (self.K @ (obs - self.obs0)[:, None]).ravel() + noise
            )
        elif obs.ndim == 2:
            action = np.vstack(
                [
                    self.action0 - (self.K @ (o - self.obs0)[:, None]).ravel() + noise
                    for o in obs
                ]
            )
        else:
            raise ValueError

        return action


def make_env(config):
    # setup the task and update config
    env_config = config["env_config"]
    task = config["exp"]["task"]
    task["rf"][task["fi"]] = 1 - config["exp"]["LoE"]

    # make env
    env = QuadEnv(env_config)
    task = env.plant.get_task(**task)
    env.plant.set_task(task)

    NHS = env.plant.find_NHS()
    env.set_NHS(NHS)
    return env


def make_env_fn(config):
    def wrapper():
        env = make_env(config)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return wrapper


def setup(user_config):
    """Get the user-defined config"""

    def update(base, new):
        assert isinstance(base, dict), f"{base} is not a dict"
        assert isinstance(new, dict), f"{new} is not a dict"
        for k, v in new.items():
            assert k in base, f"{k} not in {base}"
            if isinstance(v, dict):
                if "grid_search" in v:
                    base[k] = v
                else:
                    update(base[k], v)
            else:
                base[k] = v

    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    update(config, user_config)
    return config


def rollout(env, policy, num_trans, render=False):
    flogger = fym.Logger(max_len=num_trans, mode="deque")

    for n_traj in count(1):
        o = env.reset()
        policy.reset()

        while True:
            if render:
                env.render()

            a = policy.get_action(o)
            next_o, r, d, _ = env.step(a)

            if not d:
                flogger.record(o=o, a=a, next_o=next_o, r=r, d=d)

            o = next_o

            n_trans = len(flogger)

            if n_trans % (num_trans // 10) == 0:
                logger.debug(f"Gathered {n_trans}/{num_trans} transitions")

            if d or n_trans >= num_trans:
                break

        if n_trans >= num_trans:
            break

    info = {"n_traj": n_traj}
    return flogger.buffer, info


def trainable(config: dict, idx: int = 0):
    """Functional trainable method"""

    # seeding
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # make a taskdir
    equilibrium = config["exp"]["hover"]
    LoE = config["exp"]["LoE"]
    taskdir = Path(config["expdir"], f"{equilibrium}-hover/LoE-{int(LoE*100):02d}")
    taskdir.mkdir(parents=True, exist_ok=True)

    # get dataset
    datasetpath = taskdir / "dataset.pth"

    if not datasetpath.exists():
        logger.info(f"There is no {datasetpath}")
        logger.info("Create a dataset")

        env = make_env(config)

        # Set a behavior policy
        params = env.get_mueller_params()
        K = params["udist"] @ params["K"]

        policy = LinearPolicy(
            K=K,
            obs0=env.NHS["obs"],
            action0=env.NHS["action"],
            noisy=True,
        )

        # create a replay buffer for off-policy RLs
        data, rollout_info = rollout(env, policy, num_trans=config["num_trans"])

        sample = {
            "data": data,
            "config": config,
        }
        torch.save(sample, datasetpath)
        logger.info(f"Data was saved in {datasetpath} | Traj: {rollout_info['n_traj']}")

    dataset = torch.load(datasetpath)

    breakpoint()

    # make a trial dir
    trialdir = taskdir / config["exp"]["policy"] / f"trial-{idx:0{len(str(idx))+1}d}"
    trialdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Trial: {trialdir}")

    # save the current config
    with open(trialdir / "config.yaml", "w") as f:
        yaml.dump(config, f, Dumper=yaml.SafeDumper)

    writer = SummaryWriter(trialdir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )

    # make vectorized multiple envs for averaging the results
    envs = DummyVecEnv([make_env_fn(config) for _ in range(config["n_envs"])])

    # setup policy
    policy = SAC(envs, config)
    policy.to(device)

    rb = ReplayBuffer(
        buffer_size=int(config["buffer_size"]),
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        device=device,
        n_envs=config["n_envs"],
        handle_timeout_termination=True,
    )

    # start the game
    start_time = time.time()
    start = 0
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

            # train the policy (policy.train is reserved)
            logs = policy.learn(data)

            # logging scalars to tensorboard
            if global_step % 100 == 0:
                for k, v in logs.items():
                    writer.add_scalar(k, v, global_step)

                logger.info(f"SPS: {int(global_step / (time.time() - start_time))}")
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

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
                ndigit = len(str(int(config["total_timesteps"]))) + 1
                path = trial_dir / f"checkpoint-{global_step:0{ndigit}d}"
                torch.save(
                    {"policy": policy.state_dict(), "global_step": global_step},
                    path,
                )

    envs.close()
    writer.close()


def train_exp1(no_ray=False):
    expdir = f"exp/exp1_{datetime.now().isoformat(timespec='seconds')}"

    exp_configs = [
        {
            "exp": {
                "hover": "origin",
                "LoE": tune.grid_search([0.0, 0.5, 1.0]),
                "policy": tune.grid_search(["SAC", "LQL"]),
            },
            "seed": tune.randint(0, 100000),
            "total_timesteps": 500000,
            "expdir": expdir,
        },
        {
            "exp": {
                "hover": "near",
                "LoE": 1.0,
                "policy": tune.grid_search(["SAC", "LQL"]),
            },
            "seed": tune.randint(0, 100000),
            "total_timesteps": 500000,
            "expdir": expdir,
        },
    ]

    if no_ray:
        futures = [
            trainable(setup(config), i)
            for exp_config in exp_configs
            for i in range(5)
            for _, config in generate_variants(exp_config)
        ]
    else:

        @ray.remote
        def remote_trainable(config, i):
            return trainable(config, i)

        ray.init()
        futures = [
            remote_trainable.remote(setup(config), i)
            for exp_config in exp_configs
            for i in range(5)
            for _, config in generate_variants(exp_config)
        ]
        ray.get(futures)
        ray.shutdown()


# TEST {{{
def test_sac():
    # get a checkpoint
    trial = get_trial(
        local_dir="exp/origin-hover/outer-fixed-with-terminal/SAC/normal",
        trial_id=1,
    )
    config = trial.config
    assert trial.logdir is not None
    testpath = Path(trial.logdir) / "test-flight.h5"

    # get the env used for training
    # remedy for new config
    config = setup(
        {
            "seed": 0,
            "fault_occurs": config["fault_occurs"],
            "env_config": {
                "fkw": {
                    "max_t": 20,
                },
                "reset": {"mode": "neighbor"},
                "outer_loop": "PID",
            },
        }
    )

    # seeding
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # make env
    env = make_env(config)()
    # add a fym logger to the env
    env.env.logger = fym.Logger(path=testpath)

    # make a policy
    policy = SAC(env, config)
    policy.to(device)
    # get last checkpoint
    checkpoint = sorted(
        [
            ckpt / "checkpoint"
            for ckpt in Path(trial.logdir).rglob("checkpoint*")
            if ckpt.is_dir()
        ]
    )[-1]
    logger.info(
        f'Loaded: {Path(checkpoint).relative_to(Path("exp/origin-hover").resolve())}'
    )
    # checkpoint = os.path.join(trial.checkpoint.value, "checkpoint")
    state = torch.load(checkpoint)
    # load policy from the checkpoint
    policy.load_state_dict(state["policy"])
    policy.eval()

    # start testing
    episode_reward = 0
    done = False
    obs = env.reset()

    while not done:
        env.render(mode="tqdm")
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


# }}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--no-ray", action="store_true")
    args = parser.parse_args()

    train_exp1(**vars(args))

    # hyperparam_tune()

    # test_sac()
