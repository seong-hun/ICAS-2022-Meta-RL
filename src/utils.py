import traceback
from asyncio import Event
from datetime import datetime
from typing import Tuple

from loguru import logger
import numpy as np
import pandas as pd
import ray
import torch
from ray.actor import ActorHandle
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = "cpu"


def arr2str(array, fmt="{:5.2f}"):
    return ', '.join([fmt.format(x) for x in np.ravel(array)])


def from_numpy(ndarray):
    return torch.from_numpy(np.asarray(ndarray)).double().to(device)


def to_numpy(tensor):
    return tensor.to(device).detach().numpy()


def inertia(X, tol=1e-10):
    eigs = np.linalg.eigvals(X).real
    p = (eigs > tol).sum(axis=-1)
    z = (np.abs(eigs) <= tol).sum(axis=-1)
    n = (eigs < -tol).sum(axis=-1)
    return np.concatenate([p, z, n], axis=-1)


def rollout(env, agent, flogger, num_transitions=None,
            num_trajectories=None, record_dt=None, max_step=1e5, render=False,
            reset_random=True, only_near=False):

    if num_transitions is not None:
        assert num_trajectories is None
        num_trajectories = np.inf
    else:
        num_trajectories = num_trajectories or 1

    if record_dt is None:
        record_period = 1
    else:
        record_period = int(record_dt / env.clock.dt) or 1

    terminal_time = 0
    num_traj = 0
    num_transit = 0

    while True:

        o = env.reset(random=reset_random)
        step = 0

        while True:
            if render:
                env.render()

            a, agent_info = agent.get_action(o)
            next_o, r, d, env_info = env.step(a)
            step += 1

            if step % record_period == 0:
                if not only_near or env_info["near_eq"]:
                    flogger.record(o=o, a=a, r=r, next_o=next_o, d=d,
                                   agent_info=agent_info, env_info=env_info)
                    num_transit += 1

            if num_transitions is not None and num_transit >= num_transitions:
                break

            if d or step >= max_step:
                break

            o = next_o

        terminal_time += env_info["t"]
        num_traj += 1

        if num_transitions is not None and num_transit >= num_transitions:
            break

        if num_traj >= num_trajectories:
            break

        logger.debug("Step {} | transits {}".format(step, num_transit))

    return num_traj, terminal_time


class RunningAvg:
    def __init__(self):
        self._len = 1
        self._avg_dict = {}

    def __getitem__(self, key):
        return self._avg_dict[key]

    def get_avg(self):
        return self._avg_dict

    def items(self):
        return self._avg_dict.items()

    def reset(self):
        self._len = 1
        self._avg_dict = {}

    def add(self, in_dict, base_dict=None):
        n = self._len

        if base_dict is None:
            base_dict = self._avg_dict
            self._len += 1

        for k, v in in_dict.items():
            if isinstance(v, dict):
                if k not in base_dict:
                    base_dict[k] = {}
                self.add(v, base_dict=base_dict[k])

            else:
                if k not in self._avg_dict:
                    base_dict[k] = 0

                base_dict[k] = (n-1) / n * base_dict[k] + 1 / n * v

    def to_writer(self, writer, step, base_key="", base_dict=None):
        if base_dict is None:
            base_dict = self._avg_dict

        if base_key:
            base_key = base_key + "/"

        for k, v in base_dict.items():
            if isinstance(v, dict):
                self.to_writer(writer, step, base_key=k, base_dict=v)
            else:
                writer.add_scalar(base_key + k, v, step)


@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter


class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return


class Writer:
    def __init__(self, run_start=None):
        if run_start is None:
            run_start = datetime.now().strftime("%y-%m-%d_%H:%M:%S")

        self.writer = SummaryWriter(f"log/train/{run_start}")


# Extraction function
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data
