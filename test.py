import yaml
from ray.rllib.agents import ppo
from ray.tune import ExperimentAnalysis

from src import env

with open("config.yaml", "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.SafeLoader)


def test():
    analysis = ExperimentAnalysis("./ray-results/PPO")
    last_checkpoint = analysis.get_last_checkpoint()
    config = CONFIG["config"] | {"env_config": CONFIG["env_config"]}
    agent = ppo.PPOTrainer(config=config)
    agent.restore(last_checkpoint)


def main():
    test()


if __name__ == "__main__":
    main()
