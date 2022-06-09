from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from src.utils import from_numpy

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class TaskDataset(Dataset):
    def __init__(self, preprocess_path, seed=None):
        preprocess_path = Path(preprocess_path)
        self.taskfiles = [d for d in preprocess_path.iterdir() if d.is_file()]
        self.taskfiles.sort()

        self._length = len(self.taskfiles)
        self.rng = np.random.default_rng(seed=seed)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        taskfile = self.taskfiles[idx]
        taskdata = torch.load(taskfile)

        data = taskdata["data"]
        task = taskdata["task"]
        NHS = task["NHS"]

        dlen = len(data["o"])
        tr_idx = self.rng.choice(dlen, config["transit_size"], replace=False)
        ct_idx = self.rng.choice(dlen, config["context_size"], replace=False)

        x, u, next_x = [from_numpy(data[key][tr_idx])
                        for key in ["o", "a", "next_o"]]
        xdot = (next_x - x) / config["dt"]

        context = torch.cat(
            [from_numpy(data[key][ct_idx]) for key in ["o", "a", "next_o"]],
            dim=-1,
        )

        NHS = torch.cat(
            [from_numpy(NHS[k].ravel()) for k in ["omega", "rotorfs"]],
            dim=-1,
        )

        return context, (x, u, xdot), NHS, idx
