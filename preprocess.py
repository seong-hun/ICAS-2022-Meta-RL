import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
import yaml
from loguru import logger
from sklearn.manifold import TSNE

from src.env import Env, RLEnvWrapper
from src.policy import RLAgent, get_mueller_params
from src.utils import rollout

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


@ray.remote
def ray_sample(i, config):
    return sample(i, config)


def sample(i, config):
    # For random state and task initialization
    seed = i
    np.random.seed(seed)

    env = Env(**config["tasks"]["fkw"])
    env = RLEnvWrapper(env)

    # Randomly generates a sample task given an index of the faulty rotor.
    # The task is a dictionary of keys ["LoE", "LoM", "LoJ"]
    task = env.get_task()  # Make a random task
    env.set_task(task)  # Set the task for the env
    NHS = env.find_NHS(task)
    env.set_initial(NHS)

    # Set a behavior policy
    params = get_mueller_params(env, NHS)
    K = params["udist"] @ params["K"]
    K = np.hstack((np.ones((K.shape[0], 1)), K))

    agent = RLAgent(
        K=K,
        omega0=NHS["omega"],
        rotorfs0=NHS["rotorfs"],
        noisy=True,
    )

    data, rollout_info = rollout(
        env,
        agent,
        num_trans=config["tasks"]["num_trans"],
        num_steps=config["tasks"]["num_steps"],
        reset_random=True,
    )

    datapath = Path(config["tasks"]["dir"]) / f"task_{i:05d}.pth"
    sample = {
        "data": data,
        "info": {
            "task": task,
            "NHS": NHS,
        },
    }
    torch.save(sample, datapath)
    logger.info(f"Data was saved in {datapath} | Traj: {rollout_info['n_traj']}")


def make_tasks(num_tasks=None, append=False, no_ray=False):
    if num_tasks is None:
        num_tasks = config["tasks"]["num_tasks"]

    taskdir = Path("data/tasks")
    init = 0

    if not append and taskdir.exists():
        shutil.rmtree(taskdir)
    else:
        init = len(list(taskdir.rglob("*.pth")))

    taskdir.mkdir(parents=True, exist_ok=True)  # make a directory

    # config setup
    config = setup({
        "exp": {
            "hover": "near"
    })

    if no_ray:
        futures = [sample(i, config) for i in range(init, init + num_tasks)]
    else:
        ray.init()
        futures = [ray_sample.remote(i, config) for i in range(init, init + num_tasks)]
        ray.get(futures)  # Ray get (actual calculation)
        ray.shutdown()


def tsne():
    taskdir = Path(config["preprocess_path"])

    data = []
    data_index = []

    for i, taskpath in enumerate(taskdir.iterdir()):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--num-tasks", type=int)
    parser.add_argument("-a", "--append", action="store_true")
    parser.add_argument("-n", "--no-ray", action="store_true")
    args = parser.parse_args()

    make_tasks(**vars(args))


if __name__ == "__main__":
    main()
    # tsne()
