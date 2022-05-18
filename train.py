import fym
import gym
import numpy as np
import ray
import yaml
from gym import spaces
from loguru import logger
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from scipy.spatial.transform import Rotation

from src.env import Multicopter

with open("config.yaml", "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.SafeLoader)


class Env(fym.BaseEnv, gym.Env):
    def __init__(self, env_config):
        super().__init__(**env_config["fkw"])
        self.plant = Multicopter()

        # observation: pos (3), vel (3), angles (3), omega (3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(12,))
        # action: rotorfs (4)
        self.action_space = spaces.Box(
            low=0,
            high=self.plant.rotorf_max,
            shape=(4,),
            dtype=np.float32,
        )
        # state space for checking unwanted state
        self.state_space = spaces.Box(
            low=np.hstack(
                [
                    [-10, -10, -20],  # pos
                    [-20, -20, -20],  # vel
                    np.deg2rad([-80, -80, -360]),  # angles
                    [-50, -50, -50],  # omega
                ]
            ),
            high=np.hstack(
                [
                    [10, 10, 0],  # pos
                    [20, 20, 20],  # vel
                    np.deg2rad([80, 80, 360]),  # angles
                    [50, 50, 50],  # omega
                ]
            ),
        )

        # the desired obs for the LQR cost
        self.obs_des = np.zeros(self.observation_space.shape or (12,))
        # set the hovering height = 2m
        self.obs_des[2] = -2

        self.reward_scale = self.clock.dt
        self.flat_Q = env_config["flat_Q"]
        self.flat_R = env_config["flat_R"]
        self.perturb = env_config["perturb"]

        self.rng = np.random.default_rng()

    def step(self, action):
        # get reward using current observation and action
        reward = self.get_reward(action)
        _, done = self.update(action=action)
        next_obs = self.observation()
        done = done or not self.state_space.contains(next_obs)
        return next_obs, reward, done, {}

    def set_dot(self, t, action):
        # make a 2d vector from an 1d array
        rotorfs_cmd = np.float64(action[:, None])
        return self.plant.set_dot(t, rotorfs_cmd)

    def observation(self):
        pos, vel, R, omega = self.plant.observe_list()
        angles = Rotation.from_matrix(R).as_euler("ZYX")[::-1]
        obs = np.hstack((pos.ravel(), vel.ravel(), angles, omega.ravel()))
        return np.float32(obs)

    def get_reward(self, action):
        obs = self.observation()
        # LQR reward
        reward = 1
        reward += np.exp(
            -np.sum((obs - self.obs_des) ** 2 * self.flat_Q)
            - np.sum(action**2 * self.flat_R)
        )
        reward *= self.reward_scale
        return np.float32(reward)

    def reset(self):
        super().reset()

        # randomly perturbate the state
        obs = np.float64(self.state_space.sample())
        self.plant.pos.state = obs[:3][:, None]
        self.plant.vel.state = obs[3:6][:, None]
        self.plant.R.state = Rotation.from_euler("ZYX", obs[6:9][::-1]).as_matrix()
        self.plant.omega.state = obs[9:12][:, None]

        # check the state space contains the random state
        assert self.state_space.contains(self.observation())

        return self.observation()


def experiment(config):
    # remove tune config from config for PPOTrainer
    tune_config = config.pop("tune")
    iterations = tune_config["train_iterations"]
    # make trainer
    logger.debug("Making PPOTrainer")
    train_agent = ppo.PPOTrainer(config=config)
    checkpoint = None
    train_results = {}

    # training
    logger.debug("Start training")
    for i in range(iterations):
        logger.debug(f"Iter {i} start")
        train_results = train_agent.train()
        logger.debug(f"Iter {i} finished")
        if i % tune_config["save_period"] == 0 or i == iterations - 1:
            checkpoint = train_agent.save(tune.get_trial_dir())
        tune.report(**train_results)

    train_agent.stop()

    # evaluation
    eval_agent = ppo.PPOTrainer(config=config)
    eval_agent.restore(checkpoint)
    env = eval_agent.workers.local_worker().env

    eval_results = {
        "eval_reward": 0,
        "eval_eps_length": 0,
    }

    obs = env.reset()
    while True:
        action = eval_agent.compute_single_action(obs)
        _, reward, done, _ = env.step(action)
        eval_results["eval_reward"] += reward
        eval_results["eval_eps_length"] += 1000
        if done:
            break

    results = train_results | eval_results
    tune.report(**results)


def train():
    ray.init(log_to_driver=False)

    config = {
        "env": "quadrotor",
        "env_config": CONFIG["Env"],
        "framework": "torch",
        "num_gpus": 0,
        # "num_workers": 12,
        # "num_envs_per_worker": 50,
        # "lr": 1e-4,
        # "gamma": 0.999,
        "tune": CONFIG["tune"],
    }
    resources = ppo.PPOTrainer.default_resource_request(config)

    logger.info("Strat tune.run")
    tune.run(
        experiment,
        config=config,
        resources_per_trial=resources,
        local_dir="./ray-results",
    )
    logger.info("Finish tune.run")

    ray.shutdown()


def main():
    register_env("quadrotor", lambda env_config: Env(env_config))

    train()


if __name__ == "__main__":
    main()
