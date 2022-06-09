import numpy as np
import scipy.optimize
from scipy.spatial.transform import Rotation
import pandas as pd
from pathlib import Path
from functools import reduce
import warnings

import ray

import fym

from src.env import Env


def find_solutions(env):
    ray.init(num_cpus=12)

    LoE_id = ray.put(env.plant.LoE)

    sol = []
    for freevar in np.linspace(0, 1, 100):
        sol.append(ray_find_near_hover.remote(LoE_id, freevar))

    df = pd.DataFrame([row for row in ray.get(sol) if row])

    path = Path("data/mueller")
    path.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path / "mueller_hover_points.pkl")

    ray.shutdown()


@ray.remote
def ray_find_near_hover(LoE, freevar):
    env = Env()
    env.plant.LoE = LoE
    sol = find_near_hover(env, freevar)
    if sol["result"].success:
        return {
            "fault_index": sol["fault_index"],
            "p": sol["omega"][0, 0],
            "q": sol["omega"][1, 0],
            "r": sol["omega"][2, 0],
            "u1": sol["rotorfs"][0, 0],
            "u2": sol["rotorfs"][1, 0],
            "u3": sol["rotorfs"][2, 0],
            "u4": sol["rotorfs"][3, 0],
            "phi": sol["angles"][0],
            "theta": sol["angles"][1],
            "psi": sol["angles"][2],
            "freevar": sol["freevar"],
            "error": sol["result"].fun,
        }
    else:
        return {}


