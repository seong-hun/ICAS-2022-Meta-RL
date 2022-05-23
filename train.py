from itertools import product

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
    env_config |= {"flat_Q_factor": tune.choice(list(product(*[candidate] * 4)))}

    # add env_config to config
    tune_config["config"] |= {"env_config": env_config}

    # register env
    env.register()

    # run
    tune.run(**tune_config)

    logger.info("Finish tune.run")

    ray.shutdown()


if __name__ == "__main__":
    main()
