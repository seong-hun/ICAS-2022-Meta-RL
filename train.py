from itertools import product

import numpy as np
import ray
import yaml
from loguru import logger
from ray import tune

from src import env

with open("config.yaml", "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.SafeLoader)


def main():
    ray.init()

    logger.info("Strat tune.run")

    tune_config = CONFIG["tune"] | {"config": CONFIG["config"]}

    env_config = CONFIG["env_config"]

    # make tune search space
    candidate = [1e-4, 1e-3, 1e-2]
    flat_Qs = [
        np.hstack([np.ones(3) * state_factor for state_factor in state_factors])
        for state_factors in product(*[candidate] * 4)
    ]

    env_config |= {"flat_Q": tune.choice(flat_Qs)}

    tune_config["config"] |= {"env_config": env_config}
    tune.run(**tune_config)

    logger.info("Finish tune.run")

    ray.shutdown()


if __name__ == "__main__":
    main()
