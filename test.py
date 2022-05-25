import argparse
from pathlib import Path

import fym
import matplotlib.pyplot as plt
import numpy as np
import ray.tune.trial
import yaml
from loguru import logger
from ray.rllib.agents import ppo
from ray.tune import ExperimentAnalysis
from scipy.spatial.transform import Rotation

from src import env
from src.env import QuadEnv

with open("config.yaml", "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.SafeLoader)


def get_trial(nid=None):
    # get last checkpoint of experiments
    exppath = Path("ray-results/PPO")
    analysis = ExperimentAnalysis(
        str(exppath), default_metric="episode_len_mean", default_mode="max"
    )
    if analysis.trials is None:
        raise ValueError("There are no trials")

    if nid is not None:
        trial = analysis.trials[nid]
    else:
        trial = analysis.get_best_trial()
    return trial


def test(trial):
    if trial is None:
        raise ValueError("Cannot find a trial")

    config = trial.config

    logger.info(config)

    checkpoint = trial.checkpoint.value

    logger.info(f"Using checkpoint: {checkpoint}")

    # set agent weight to the checkpoint
    agent = ppo.PPOTrainer(config=config)
    agent.restore(checkpoint)

    # make env
    testpath = Path(checkpoint).parent / "test-flight.h5"
    env_config = config["env_config"]
    env_config["fkw"]["max_t"] = 20
    env = QuadEnv(env_config)
    env.logger = fym.Logger(path=testpath)

    logger.info("Start test flight")

    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        env.render()
        action = agent.compute_single_action(obs, explore=False)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

    env.close()

    logger.info("End test flight")


def plot_test(trial):
    if not trial:
        raise ValueError("Trial does not exist.")

    testdir = Path(trial.checkpoint.value).parent
    testpath = testdir / "test-flight.h5"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nid", type=int, default=None)
    args = parser.parse_args()

    env.register()

    trial = get_trial(nid=args.nid)

    test(trial)
    plot_test(trial)


if __name__ == "__main__":
    main()
