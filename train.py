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
    tune_config["config"] |= {"env_config": CONFIG["env_config"]}
    tune.run(**tune_config)

    logger.info("Finish tune.run")

    ray.shutdown()


if __name__ == "__main__":
    main()
