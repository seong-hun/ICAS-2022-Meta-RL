import argparse
from pathlib import Path
from pprint import pprint

import fym
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from loguru import logger
from ray.tune import ExperimentAnalysis
from scipy.spatial.transform import Rotation

from src.env import QuadEnv
from src.sac import SAC
from src.utils import make_env, merge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def test(trialdir):
    trialdir = Path(trialdir)
    if not trialdir.exists():
        raise ValueError("Cannot find a trial")

    with open(trialdir / "config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    with open("config.yaml", "r") as f:
        CONFIG = yaml.load(f, Loader=yaml.SafeLoader)

    test_config = {
        "env_config": {
            "fkw": {"max_t": 20, "dt": 0.001},
            "outer_loop": "PID",
            "observation_scale": {
                "bound": 1.0,
                "init": 0.1,
            },
        },
        "exp": {
            "LoE": config["exp"]["LoE"],
            "fi": config["exp"]["task"]["fi"],
        },
    }
    config = merge(CONFIG, test_config)
    pprint(config)
    env = make_env(config)

    policy = SAC(env, config)
    policy.to(device)
    last_checkpoint = sorted(trialdir.glob("checkpoint-*"))[-1]
    model = torch.load(last_checkpoint)
    policy.load_state_dict(model["policy"])
    policy.eval()

    logger.info("Start test flight")

    env.logger = fym.Logger(trialdir / "test-flight.h5")
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action = policy.get_action(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

    env.close()

    logger.info("End test flight")


def plot(trialdir):
    trialdir = Path(trialdir)

    if not trialdir:
        raise ValueError("Trial does not exist.")

    data = fym.load(trialdir / "test-flight.h5")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trialdir", type=str)
    parser.add_argument("-p", "--only-plot", type=str)
    args = parser.parse_args()

    if not args.only_plot:
        test(trialdir=args.trialdir)

    plot(trialdir=args.trialdir)
