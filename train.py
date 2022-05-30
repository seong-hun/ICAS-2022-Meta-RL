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
import yaml
from loguru import logger
from ray.tune.suggest.variant_generator import generate_variants
from scipy.spatial.transform import Rotation
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from torch.utils.tensorboard.writer import SummaryWriter

from src.env import QuadEnv
from src.lql import LQL
from src.sac import SAC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    # make a trial dir
    trialdir = taskdir / config["exp"]["policy"] / f"trial-{idx:0{len(str(idx))+1}d}"
    logger.info(f"Trial: {trialdir}")

    # save the current config
    writer = SummaryWriter(trialdir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )

    # make vectorized multiple envs for averaging the results
    envs = DummyVecEnv([make_env_fn(config) for _ in range(config["n_envs"])])

    # setup policy
    if config["exp"]["policy"] == "SAC":
        policy = SAC(envs, config)
    elif config["exp"]["policy"] == "LQL":
        policy = LQL(envs, config)
    else:
        raise ValueError

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

            # lazy create the trial dir
            if not trialdir.exists():
                trialdir.mkdir(parents=True, exist_ok=True)
                with open(trialdir / "config.yaml", "w") as f:
                    yaml.dump(config, f, Dumper=yaml.SafeDumper)

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
                path = trialdir / f"checkpoint-{global_step:0{ndigit}d}"
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
