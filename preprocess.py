import shutil
from pathlib import Path

import fym
import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
import yaml
from loguru import logger
from sklearn.manifold import TSNE

from src.env import Env, RLEnvWrapper
from src.policy import LinearPolicy, MuellerAgent, UntrimmedPolicyWrapper
from src.utils import rollout

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


@ray.remote
def sample_transitions(i, config):
    # For random state and task initialization
    np.random.seed(i)

    env = Env(max_t=config["prep_max_t"])
    env = RLEnvWrapper(env)

    # Randomly generates a sample task given an index of the faulty rotor.
    # The task is a dictionary of keys ["LoE", "LoM", "LoJ"]
    task = env.get_random_task()  # Make a random task
    env.set_task(task)  # Set the task for the env

    # Set a behavior policy
    params = MuellerAgent.get_mueller_cntr_params(env)

    K = params["udist"] @ params["K"]
    agent = LinearPolicy(K=K, random=True)
    agent = UntrimmedPolicyWrapper(
        agent,
        omega0=env.task["NHS"]["omega"],
        rotorfs0=env.task["NHS"]["rotorfs"],
    )

    env.plant.R.initial_state = env.task["NHS"]["R"]
    env.plant.omega.initial_state = env.task["NHS"]["omega"]

    flogger = fym.Logger(max_len=config["num_transitions_per_task"], mode="deque")

    num_traj, terminal_time = rollout(
        env,
        agent,
        flogger=flogger,
        num_transitions=config["num_transitions_per_task"],
        record_dt=config["record_dt"],
        reset_random=True,
    )

    logger.debug(
        f"fault_index: {env.task['fault_index'][0]} | "
        f"freevar: {task['freevar']:3.2f} | "
        f"Traj: {num_traj} | "
        f"Average flight time: {terminal_time / num_traj:5.2f} sec"
    )

    sample = {
        "task": env.task,
        "data": flogger.buffer,
    }
    return sample


def main():
    prepdir = Path(config["preprocess_path"])
    if prepdir.exists():
        shutil.rmtree(prepdir)
    prepdir.mkdir(parents=True, exist_ok=False)  # make a directory

    ray.init(num_cpus=12)
    config_id = ray.put(config)

    # Create futures
    futures = [
        sample_transitions.remote(i, config_id) for i in range(config["num_tasks"])
    ]

    samples = ray.get(futures)  # Ray get (actual calculation)

    for task_idx, sample in enumerate(samples):
        datapath = prepdir / f"task_{task_idx:02d}.pth"
        torch.save(sample, datapath)
        logger.info(f"Data was saved in {datapath}")

    ray.shutdown()


def tsne():
    prepdir = Path(config["preprocess_path"])

    data = []
    data_index = []

    for i, taskpath in enumerate(prepdir.iterdir()):
        taskdata = torch.load(taskpath)

        xdata = taskdata["data"]["o"].squeeze()
        index = [i] * len(xdata)

        data.append(xdata)
        data_index.append(index)

    data = np.vstack(data)
    data_index = np.hstack(data_index)

    X = TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(data)

    plt.figure()
    cmap = plt.get_cmap("tab20")
    for task_idx in range(i + 1):
        task_msk = data_index == task_idx
        plt.scatter(X[task_msk, 0], X[task_msk, 1], color=cmap(task_idx / i))

    plt.show()


if __name__ == "__main__":
    main()
    # tsne()
