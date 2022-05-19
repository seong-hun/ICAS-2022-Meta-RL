from pathlib import Path

import yaml
from loguru import logger
from ray.rllib.agents import ppo
from ray.tune import ExperimentAnalysis

from src.env import QuadEnv

with open("config.yaml", "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.SafeLoader)


def test():
    # get last checkpoint of experiments
    exppath = Path("ray-results/PPO")
    analysis = ExperimentAnalysis(str(exppath))

    logger.info(f"Exp path: {str(exppath)}")

    last_checkpoint = analysis.get_last_checkpoint()

    # set agent weight to the checkpoint
    config = CONFIG["config"] | {"env_config": CONFIG["env_config"]}
    agent = ppo.PPOTrainer(config=config)
    agent.restore(last_checkpoint)

    logger.debug("Agent is restored")

    trainpath = Path(str(last_checkpoint)).relative_to(exppath.resolve())
    logger.info(f"Loaded: {str(trainpath)}")

    # make env
    env = QuadEnv(CONFIG["env_config"])

    logger.info("Start test flight")

    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        env.render()
        action = agent.compute_single_action(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

    logger.info("End test flight")


def main():
    test()


if __name__ == "__main__":
    main()
