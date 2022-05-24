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

    # run
    config = CONFIG["config"]
    tune_config = CONFIG["tune_config"]
    tune.run(config=config, **tune_config)

    logger.info("Finish tune.run")

    ray.shutdown()


if __name__ == "__main__":
    # register env
    env.register()
    main()
