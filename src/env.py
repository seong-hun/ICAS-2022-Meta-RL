import fym
import gym
import numpy as np
from fym.utils.rot import hat
from gym import spaces
from scipy.spatial.transform import Rotation


def cross(x, y):
    return np.cross(x, y, axis=0)


class Multicopter(fym.BaseEnv):
    """A Multicopter enviroment with a varying LoE feature.

    Configuration:

          ^ x
          |
         (0)
    - (3) + (1) -> y
         (2)
          |

    Reference:
        - S. Mallavalli and A. Fekih, “A fault tolerant tracking control for a
          quadrotor UAV subject to simultaneous actuator faults and exogenous
          disturbances,” International Journal of Control, vol. 93, no. 3, pp.
          655–668, Mar. 2020, doi: 10.1080/00207179.2018.1484173.
    """

    g = 9.81
    m = 1.00  # [kg] mass
    r = 0.24  # [m] torque arm
    J = np.diag([8.1, 8.1, 14.2]) * 1e-3
    Jinv = np.linalg.inv(J)
    b = 5.42e-5  # [N s^2 / rad^2] thrust coeff.
    d = 1.1e-6  # [N m s^2 / rad^2] reaction torque coeff.
    Kf = np.diag([5.567, 5.567, 6.354]) * 1e-4  # [N.s/m] drag coeff.
    Kt = np.diag([5.567, 5.567, 6.354]) * 1e-4  # [N.s/rad] aerodynamic drag
    rotorf_min = 0  # [N]
    # rotorf_max = b * 523.0**2  # [N]
    rotorf_max = 20  # [N]
    """ Auxiliary constants """
    e3 = np.vstack((0, 0, 1))
    nrotors = 4
    B = np.array(
        [
            [1, 1, 1, 1],
            [0, -r, 0, r],
            [r, 0, -r, 0],
            [-d / b, d / b, -d / b, d / b],
        ]
    )
    Lambda = np.eye(4)

    def __init__(self):
        super().__init__()
        self.pos = fym.BaseSystem(np.vstack((0, 0, -10)))
        self.vel = fym.BaseSystem(np.zeros((3, 1)))
        self.R = fym.BaseSystem(np.eye(3))
        self.omega = fym.BaseSystem(np.zeros((3, 1)))

    def deriv(self, pos, vel, R, omega, rotorfs):
        u = self.B @ rotorfs
        fT, M = u[:1], u[1:]

        dpos = vel
        dvel = (
            1
            / self.m
            * (+self.m * self.g * self.e3 + R @ (-fT * self.e3) - self.Kf @ vel)
        )
        dR = R @ hat(omega)
        domega = self.Jinv @ (
            M - cross(omega, self.J @ omega) - np.linalg.norm(omega) * self.Kt @ omega
        )
        return dpos, dvel, dR, domega

    def set_dot(self, t, rotorfs_cmd):
        states = self.observe_list()
        rotorfs = self.set_valid(t, rotorfs_cmd)
        dots = self.deriv(*states, rotorfs)
        self.pos.dot, self.vel.dot, self.R.dot, self.omega.dot = dots
        return dict(rotorfs=rotorfs)

    def set_valid(self, t, rotorfs_cmd):
        """Saturate the rotor angular velocities and set faults."""
        rotorfs = np.clip(rotorfs_cmd, self.rotorf_min, self.rotorf_max)
        return self.Lambda @ rotorfs


class QuadEnv(fym.BaseEnv, gym.Env):
    """Environment for inner-loop RL training."""

    def __init__(self, env_config):
        super().__init__(**env_config["fkw"])
        self.plant = Multicopter()

        self.plant.pos.initial_state = np.array(env_config["pos_init"])[:, None]

        # observation: pos (3), vel (3), angles (3), omega (3)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(12,), dtype=np.float32
        )
        # action: rotorfs (4)
        self.action_space = spaces.Box(
            low=0,
            high=self.plant.rotorf_max,
            shape=(4,),
            dtype=np.float32,
        )
        # state space for checking unwanted state
        self.state_space = spaces.Box(
            low=np.float32(
                np.hstack(
                    [
                        env_config["state_space"]["pos"]["low"],
                        env_config["state_space"]["vel"]["low"],
                        np.deg2rad(env_config["state_space"]["angles"]["low"]),
                        env_config["state_space"]["omega"]["low"],
                    ]
                )
            ),
            high=np.float32(
                np.hstack(
                    [
                        env_config["state_space"]["pos"]["high"],
                        env_config["state_space"]["vel"]["high"],
                        np.deg2rad(env_config["state_space"]["angles"]["high"]),
                        env_config["state_space"]["omega"]["high"],
                    ]
                )
            ),
        )

        # the desired obs for the LQR cost
        self.pos_des = np.array(env_config["pos_des"])[:, None]

        # reward weights
        self.weights = env_config["reward_weights"]
        self.reward_scale = self.clock.dt

        # for task setup
        self._tasks = env_config["tasks"]

        self.rng = np.random.default_rng()

    def step(self, action):
        # get reward using current observation and action
        reward = self.get_reward(action)
        _, done = self.update(action=action)
        next_obs = self.observation()
        bounds_out = not self.state_space.contains(next_obs)
        if bounds_out:
            reward += -1
        done = done or bounds_out
        return next_obs, reward, done, {}

    def set_dot(self, t, action):
        # make a 2d vector from an 1d array
        rotorfs_cmd = np.float64(action[:, None])
        plant_info = self.plant.set_dot(t, rotorfs_cmd)
        return {
            "t": t,
            "rotorfs_cmd": rotorfs_cmd,
            "plant_info": plant_info,
            **self.observe_dict(),
        }

    def observation(self):
        _, vel, R, omega = self.plant.observe_list()

        vz = vel.ravel()[2]

        # fixed acc command directing -e3 for inner-loop RL
        n_i = -self.plant.e3
        n_b = R.T @ n_i
        eta = n_b.ravel()[:2]

        obs = np.hstack((vz, eta, omega))

        return np.float32(obs)

    def get_reward(self, action):
        pos, vel, R, _ = self.plant.observe_list()
        angles = Rotation.from_matrix(R).as_euler("ZYX")[::-1]

        # position error
        pe = np.linalg.norm(pos - self.pos_des)

        # hovering reward
        reward = 0
        reward += -pe * self.weights["pos"]
        reward += -np.linalg.norm(vel) * self.weights["vel"]
        reward += -np.linalg.norm(angles) * self.weights["angles"]
        reward += -np.linalg.norm(action) * self.weights["action"]

        # scaling
        reward *= self.reward_scale
        return np.float32(reward)

    def reset(self, with_obs=None):
        super().reset()

        if with_obs is not None:
            obs = with_obs
        else:
            # randomly perturbate the state
            # obs = np.float64(self.state_space.sample())
            obs = self.observation()

        # set states from obs
        self.plant.pos.state = obs[:3][:, None]
        self.plant.vel.state = obs[3:6][:, None]
        self.plant.R.state = Rotation.from_euler("ZYX", obs[6:9][::-1]).as_matrix()
        self.plant.omega.state = obs[9:12][:, None]

        # check the state space contains the random state
        assert self.state_space.contains(self.observation())

        return self.observation()

    def get_task(self, random=True, **kwargs):
        # default task
        task = {
            "rf": np.ones(4),  # rotor faults (0: complete failure, 1: healthy)
            "fi": (),  # fault index
            "mf": 1,  # loss of mass
            "Jf": np.ones(3),  # loss of J
            "fv": 0.5,  # free variable
        }

        # update with random task
        if random:
            rng = self.rng

            rand_rf = np.ones(4)
            rand_fi = tuple(
                sorted(
                    rng.choice(
                        range(4),
                        rng.integers(1, self._tasks["max_frotors"] + 1),
                        replace=False,
                    )
                )
            )
            for i in rand_fi:
                rand_rf[i] = rng.uniform(*self._tasks["rfrange"])

            task |= {
                "rf": rand_rf,
                "fi": rand_fi,
                "mf": rng.uniform(*self._tasks["mfrange"]),
                "Jf": rng.uniform(*self._tasks["Jfrange"], size=3),
                "fv": rng.uniform(*self._tasks["fvrange"]),
            }

        # update with given task
        task |= kwargs

        # update fault index to match rotor faults (rf)
        task["fi"] = tuple(
            i for i, v in enumerate(task["rf"]) if v <= self._tasks["rfrange"][1]
        )

        return task

    def set_task(self, task):
        self.plant.Lambda = np.diag(task["rf"])
        self.plant.m *= task["mf"]
        self.plant.J = np.diag(task["Jf"]) @ self.plant.J
        self.task = task
