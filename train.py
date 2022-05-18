import fym
import gym
import numpy as np
import ray
import yaml
from gym import spaces
from ray import tune
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
        reward = -np.sum((obs - self.obs_des) ** 2 * self.flat_Q) - np.sum(
            action**2 * self.flat_R
        )
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


def train():
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    config = {
        "env": "quadrotor",
        "env_config": CONFIG["Env"],
        "framework": "torch",
        "num_gpus": 0,
        "num_workers": 12,
        "num_envs_per_worker": 50,
        "lr": 1e-4,
        "gamma": 0.999,
    }
    tune.run(
        "PPO",
        config=config,
        stop={
            "training_iteration": 1000,
        },
        local_dir="./ray-results",
        checkpoint_freq=30,
        checkpoint_at_end=True,
    )

    ray.shutdown()


def main():
    register_env("quadrotor", lambda env_config: Env(env_config))

    train()


if __name__ == "__main__":
    main()
