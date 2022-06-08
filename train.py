import argparse
import random
import socket
import time
from datetime import datetime
from pathlib import Path

import fym
import gym
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import ray
import ray.tune as tune
import torch
import yaml
from loguru import logger
from ray.tune.suggest.variant_generator import generate_variants
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from torch.utils.tensorboard.writer import SummaryWriter

from src.lql import LQL
from src.sac import SAC
from src.utils import make_env, merge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.SafeLoader)


def make_env_fn(config):
    def wrapper():
        env = make_env(config)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return wrapper


def evaluate(policy, config):
    policy.eval()

    test_config = {
        "env_config": {
            "fkw": {"max_t": 20},
            "outer_loop": "PID",
            "observation_scale": {
                "bound": 1.0,
                "init": 0.1,
            },
        },
    }
    config = merge(config, test_config)
    env = make_env(config)

    episode_reward = 0
    step = 0
    done = False
    obs = env.reset()

    while not done:
        action = policy.get_action(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        step += 1

    env.close()

    policy.train()

    logs = {
        "charts/episodic_return": episode_reward,
        "charts/episodic_length": step,
        "charts/averge_return": episode_reward / step,
    }
    return logs


def trainable(config: dict, idx: int = 0):
    """Functional trainable method"""

    # seeding
    seed = config["train"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # make a taskdir
    equilibrium = config["exp"]["hover"]
    LoE = config["exp"]["LoE"]
    taskdir = Path(config["exp"]["dir"], f"{equilibrium}-hover/LoE-{int(LoE*100):02d}")

    # make a trial dir
    trialdir = taskdir / config["exp"]["policy"] / f"trial-{idx:0{len(str(idx))+1}d}"
    logger.info(f"Trial: {trialdir}")

    # Lazy saving
    configdir = None
    writer = None
    flogger = None

    # make vectorized multiple envs for averaging the results
    envs = DummyVecEnv([make_env_fn(config) for _ in range(config["train"]["n_envs"])])

    # setup policy
    if config["exp"]["policy"] == "SAC":
        policy = SAC(envs, config)
    elif config["exp"]["policy"] == "LQL":
        policy = LQL(envs, config)
    else:
        raise ValueError

    policy.to(device)
    policy.train()

    rb = ReplayBuffer(
        buffer_size=int(config["train"]["buffer_size"]),
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        device=device,
        n_envs=config["train"]["n_envs"],
        handle_timeout_termination=True,
    )

    # start the game
    start_time = time.time()
    start = 0
    obs = envs.reset()

    for global_step in range(start, int(config["train"]["total_timesteps"])):
        # ALGO LOGIC: put action logic here
        if global_step < config["train"]["learning_starts"]:
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
        if global_step > config["train"]["learning_starts"]:
            data = rb.sample(int(config["train"]["batch_size"]))

            # train the policy (policy.train is reserved)
            train_logs = policy.learn(data)

            # Evaluation
            if global_step % 1000 == 0:
                # lazy define writer
                if not trialdir.exists():
                    trialdir.mkdir(parents=True)

                if configdir is None:
                    configdir = trialdir / "config.yaml"
                    with open(configdir, "w") as f:
                        yaml.dump(config, f, Dumper=yaml.SafeDumper)

                if writer is None:
                    # save the current config
                    writer = SummaryWriter(trialdir)
                    writer.add_text(
                        "hyperparameters",
                        "|param|value|\n|-|-|\n%s"
                        % (
                            "\n".join(
                                [f"|{key}|{value}|" for key, value in config.items()]
                            )
                        ),
                    )

                # lazy define flogger
                if flogger is None:
                    flogger = fym.Logger(trialdir / "train-history.h5", max_len=1)

                # Train logs
                for k, v in train_logs.items():
                    writer.add_scalar(k, v, global_step)

                SPS = int(global_step / (time.time() - start_time))
                logger.info(f"[{global_step}] SPS: {SPS}")
                writer.add_scalar("charts/SPS", SPS, global_step)

                # Evaluation logs
                eval_logs = evaluate(policy, config)
                for k, v in eval_logs.items():
                    writer.add_scalar(k, v, global_step)

                flogger.record(global_step=global_step, **eval_logs | train_logs)

            # Save the model into a checkpoint
            if global_step % (config["train"]["total_timesteps"] // 20) == 0:
                policy.eval()

                ndigit = len(str(int(config["train"]["total_timesteps"]))) + 1
                path = trialdir / f"checkpoint-{global_step:0{ndigit}d}"
                torch.save(
                    {"policy": policy.state_dict(), "global_step": global_step},
                    path,
                )

                policy.train()

    envs.close()
    if writer is not None:
        writer.close()
    if flogger is not None:
        flogger.close()


def train_exp1(expdir=None, with_ray=False):
    expdir = (
        expdir
        or f"exp/{socket.gethostname()}_{datetime.now().isoformat(timespec='seconds')}"
    )

    exp_configs = [
        {
            "exp": {
                "hover": "origin",
                "LoE": tune.grid_search([0.0, 0.5, 1.0]),
                "policy": "LQL",
            },
            "env_config": {
                "observation_scale": {
                    "bound": 0.15,
                    "init": 0.1,
                },
            },
            "train": {
                "seed": tune.randint(0, 100000),
            },
        },
        {
            "exp": {"hover": "near", "LoE": 1.0, "policy": "LQL"},
            "env_config": {
                "observation_scale": {
                    "bound": 0.15,
                    "init": 0.1,
                },
            },
            "train": {
                "seed": tune.randint(0, 100000),
            },
        },
        # {
        #     "exp": {
        #         "hover": "origin",
        #         "LoE": tune.grid_search([0.0, 0.5, 1.0]),
        #         "policy": "SAC",
        #     },
        #     "env_config": {
        #         "observation_scale": {"bound": 1.0},
        #     },
        #     "seed": tune.randint(0, 100000),
        # },
        # {
        #     "exp": {
        #         "hover": "origin",
        #         "LoE": tune.grid_search([0.0, 0.5, 1.0]),
        #         "policy": "SAC",
        #     },
        #     "env_config": {
        #         "observation_scale": {
        #             "bound": 1.0,
        #             "init": 0.5,
        #         },
        #     },
        #     "seed": tune.randint(0, 100000),
        # },
    ]

    if exp_configs == []:
        return

    train_config = {
        "exp": {"dir": expdir},
    }

    if not with_ray:
        futures = [
            trainable(merge(CONFIG, config, train_config), i)
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
            remote_trainable.remote(merge(CONFIG, config, train_config), i)
            for exp_config in exp_configs
            for i in range(5)
            for _, config in generate_variants(exp_config)
        ]
        ray.get(futures)
        ray.shutdown()


def plot_trials(ax, trialdir):
    if not trialdir.exists():
        logger.warning(f"NOT EXISTS: {trialdir}")
        return

    cdict = {
        "LQL": plt.get_cmap("Set2")(0),
        "SAC": plt.get_cmap("Set2")(1),
    }
    policy = trialdir.name

    trials = sorted(trialdir.glob("trial-*"))
    dataset = {trial.name: fym.load(trial / "train-history.h5") for trial in trials}
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    k: (
                        np.ravel(v["charts"]["episodic_return"])
                        / np.ravel(v["charts"]["episodic_length"])
                        # np.ravel(v["charts"]["episodic_length"])
                    )
                },
                index=v["global_step"],
            )
            for k, v in dataset.items()
        ],
        axis=1,
    )
    df = df.loc[:, df.std() < 10]

    x = df.index
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    ax.plot(x, mean, color=cdict[policy], label=policy)
    ax.fill_between(x, mean - std, mean + std, color=cdict[policy], alpha=0.2)


# good example: exp/exp1_2022-05-30T20:50:43"
def plot_exp1(expdir=None):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Times New Roman", size=10)
    plt.rc("axes", labelsize=10, grid=True, linewidth=0.5)
    plt.rc("grid", linestyle="--", alpha=0.8, linewidth=0.5)
    plt.rc("legend", fontsize=8)
    plt.rc("lines", markersize=3, linewidth=1)

    expdir = expdir or sorted([p for p in Path("exp").iterdir() if p.is_dir()])[-1]

    # -- Figure 1. Origin-hover

    origindir = expdir / "origin-hover"

    fig, axes = plt.subplots(1, 4, figsize=(9, 2.7), sharey=True)

    get_LoE = lambda x: int(str(x.name).split("-")[-1])

    LoEs = sorted(origindir.iterdir(), key=get_LoE)

    for ax, LoEdir in zip(axes[:3], LoEs):
        for policy in ["LQL", "SAC"]:
            plot_trials(ax, LoEdir / policy)

        ticker = matplotlib.ticker.EngFormatter(unit="")
        ax.xaxis.set_major_formatter(ticker)
        ax.set_xlabel("Number of steps")
        ax.set_title(f"{get_LoE(LoEdir)} \% LoE")
        ax.set_ylim(-25, 1)

    axes[2].legend(loc="lower right")
    axes[0].set_ylabel("Average return")

    origindir = expdir / "near-hover"
    LoEdir = origindir / "LoE-100"

    ax = axes[3]

    for policy in ["LQL", "SAC"]:
        plot_trials(axes[3], LoEdir / policy)

    ticker = matplotlib.ticker.EngFormatter(unit="")
    ax.xaxis.set_major_formatter(ticker)
    ax.set_xlabel("Number of steps")
    ax.legend(loc="lower right")
    ax.set_ylabel("Average return")

    fig.tight_layout()

    plt.show()


#     config = trial.config
#     assert trial.logdir is not None
#     testpath = Path(trial.logdir) / "test-flight.h5"

#     # get the env used for training
#     # remedy for new config
#     config = setup(
#         {
#             "seed": 0,
#             "fault_occurs": config["fault_occurs"],
#             "env_config": {
#                 "fkw": {
#                     "max_t": 20,
#                 },
#                 "reset": {"mode": "neighbor"},
#                 "outer_loop": "PID",
#             },
#         }
#     )

#     # seeding
#     seed = config["seed"]
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     # make env
#     env = make_env(config)()
#     # add a fym logger to the env
#     env.env.logger = fym.Logger(path=testpath)

#     # make a policy
#     policy = SAC(env, config)
#     policy.to(device)
#     # get last checkpoint
#     checkpoint = sorted(
#         [
#             ckpt / "checkpoint"
#             for ckpt in Path(trial.logdir).rglob("checkpoint*")
#             if ckpt.is_dir()
#         ]
#     )[-1]
#     logger.info(
#         f'Loaded: {Path(checkpoint).relative_to(Path("exp/origin-hover").resolve())}'
#     )
#     # checkpoint = os.path.join(trial.checkpoint.value, "checkpoint")
#     state = torch.load(checkpoint)
#     # load policy from the checkpoint
#     policy.load_state_dict(state["policy"])
#     policy.eval()

#     # start testing
#     episode_reward = 0
#     done = False
#     obs = env.reset()

#     while not done:
#         env.render(mode="tqdm")
#         action = policy.get_action(obs, deterministic=True)
#         obs, reward, done, _ = env.step(action)
#         episode_reward += reward

#     env.close()

#     # -- Plot
#     data = fym.load(testpath)
#     if type(data) is not dict:
#         raise ValueError("data should be dict")

#     fig, axes = plt.subplots(4, 2, figsize=(12, 8), squeeze=False, sharex=True)

#     data["plant"]

#     ax = axes[0, 0]
#     ax.plot(data["t"], data["plant"]["pos"].squeeze(-1))
#     ax.set_ylabel("Position, m")
#     ax.legend([r"$x$", r"$y$", r"$z$"])

#     ax = axes[1, 0]
#     ax.plot(data["t"], data["plant"]["vel"].squeeze(-1))
#     ax.set_ylabel("Velocity, m/s")
#     ax.legend([r"$v_x$", r"$v_y$", r"$v_z$"])

#     ax = axes[2, 0]
#     angles = Rotation.from_matrix(data["plant"]["R"]).as_euler("ZYX")[:, ::-1]
#     ax.plot(data["t"], np.rad2deg(angles))
#     ax.set_ylabel("Angles, deg")
#     ax.legend([r"$\phi$", r"$\theta$", r"$\psi$"])

#     ax = axes[3, 0]
#     ax.plot(data["t"], data["plant"]["omega"].squeeze(-1))
#     ax.set_ylabel("Omega, rad/sec")
#     ax.legend([r"$p$", r"$q$", r"$r$"])

#     ax.set_xlabel("Time, sec")

#     ax = axes[0, 1]
#     ax.plot(data["t"], data["rotorfs_cmd"].squeeze(-1)[:, 0], "r--")
#     ax.plot(data["t"], data["plant_info"]["rotorfs"].squeeze(-1)[:, 0], "k-")
#     ax.set_ylabel("Rotor 1 thrust, N")

#     ax = axes[1, 1]
#     ax.plot(data["t"], data["rotorfs_cmd"].squeeze(-1)[:, 1], "r--")
#     ax.plot(data["t"], data["plant_info"]["rotorfs"].squeeze(-1)[:, 1], "k-")
#     ax.set_ylabel("Rotor 2 thrust, N")

#     ax = axes[2, 1]
#     ax.plot(data["t"], data["rotorfs_cmd"].squeeze(-1)[:, 2], "r--")
#     ax.plot(data["t"], data["plant_info"]["rotorfs"].squeeze(-1)[:, 2], "k-")
#     ax.set_ylabel("Rotor 3 thrust, N")

#     ax = axes[3, 1]
#     ax.plot(data["t"], data["rotorfs_cmd"].squeeze(-1)[:, 3], "r--")
#     ax.plot(data["t"], data["plant_info"]["rotorfs"].squeeze(-1)[:, 3], "k-")
#     ax.set_ylabel("Rotor 4 thrust, N")

#     ax.set_xlabel("Time, sec")

#     fig.tight_layout()
#     # fig.savefig(testdir / "hist.pdf", bbox_inches="tight")

#     plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--with-ray", action="store_true")
    parser.add_argument("-d", "--expdir", type=Path)
    parser.add_argument("-p", "--plot", action="store_true")
    args = parser.parse_args()

    if args.plot:
        plot_exp1(expdir=args.expdir)
    else:
        train_exp1(expdir=args.expdir, with_ray=args.with_ray)
