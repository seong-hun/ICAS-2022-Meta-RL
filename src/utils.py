from copy import deepcopy
from functools import reduce

import numpy as np

from src.env import QuadEnv


def make_env(config):
    # setup the task and update config
    env_config = config["env_config"]

    # make a task
    rf = np.ones(4)
    rf[config["exp"]["fi"]] = 1 - config["exp"]["LoE"]
    assert np.all((0 <= rf) & (rf <= 1))

    # make env
    env = QuadEnv(env_config)
    task = env.plant.get_task(rf=rf)
    env.plant.set_task(task)

    if config["exp"]["hover"] == "near":
        NHS = env.plant.find_NHS()
        env.set_NHS(NHS)
    return env


def merge(*configs):
    assert len(configs) > 1

    def _merge(base, new):
        assert isinstance(base, dict), f"{base} is not a dict"
        assert isinstance(new, dict), f"{new} is not a dict"
        out = deepcopy(base)
        for k, v in new.items():
            # assert k in out, f"{k} not in {base}"
            if isinstance(v, dict):
                if "grid_search" in v:
                    out[k] = v
                else:
                    out[k] = _merge(out[k], v)
            else:
                out[k] = v

        return out

    return reduce(_merge, configs)
