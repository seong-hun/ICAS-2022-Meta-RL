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

    def __init__(self, plant_config):
        super().__init__()
        self.pos = fym.BaseSystem(np.array(plant_config["init"]["pos"])[:, None])
        self.vel = fym.BaseSystem(np.zeros((3, 1)))
        self.R = fym.BaseSystem(np.eye(3))
        self.omega = fym.BaseSystem(np.zeros((3, 1)))

        self.task_config = plant_config["task_config"]

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
            rand_rf = np.ones(4)
            rand_fi = tuple(
                sorted(
                    np.random.choice(
                        range(4),
                        np.random.randint(1, self.task_config["max_frotors"] + 1),
                        replace=False,
                    )
                )
            )
            for i in rand_fi:
                rand_rf[i] = np.random.uniform(*self.task_config["rfrange"])

            task |= {
                "rf": rand_rf,
                "fi": rand_fi,
                "mf": np.random.uniform(*self.task_config["mfrange"]),
                "Jf": np.random.uniform(*self.task_config["Jfrange"], size=3),
                "fv": np.random.uniform(*self.task_config["fvrange"]),
            }

        # update with given task
        task |= kwargs

        # update fault index to match rotor faults (rf)
        task["fi"] = tuple(
            i for i, v in enumerate(task["rf"]) if v <= self.task_config["rfrange"][1]
        )

        return task

    def set_task(self, task):
        self.Lambda = np.diag(task["rf"])
        self.m *= task["mf"]
        self.J = np.diag(task["Jf"]) @ self.J
        self.task = task


class PID:
    def __init__(self, kP, kI, kD, dt):
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.dt = dt

        self._prev_e = None
        self._prev_e_int = 0

    def get(self, e):
        # At first, prev_e = e
        prev_e = self._prev_e or e

        e_deriv = (e - prev_e) / self.dt
        e_int = self._prev_e_int

        u = self.kP * e + self.kI * e_int + self.kD * e_deriv
        return u

    def update(self, e):
        prev_e = self._prev_e or e
        e_int = self._prev_e_int

        if self.kI:
            self._prev_e_int = e_int + (prev_e + e) / 2 * self.dt  # trapz
        if self.kD:
            self._prev_e = e


class FixedOuterLoop:
    def __init__(self):
        self.F_des_i = np.vstack((0, 0, -1))

    def get(self, pos):
        return self.F_des_i

    def update(self, pos):
        pass


class PIDOuterLoop:
    def __init__(self, env):
        self.clock = env.clock

        # PID outer-loop controller
        dt = self.clock.dt
        self.PID_x = PID(kP=0.25, kI=0.006, kD=0.4, dt=dt)
        self.PID_y = PID(kP=0.25, kI=0.006, kD=0.4, dt=dt)
        self.PID_z = PID(kP=1, kI=0.1, kD=1, dt=dt)

        # set reference
        self.get_pos_ref = lambda t: np.vstack((0, 0, -5))

        # Constants
        self.m = env.plant.m
        self.g = env.plant.g
        self.e3 = env.plant.e3

    def get(self, pos):
        t = self.clock.get()
        pos_ref = self.get_pos_ref(t)
        pos_error = (pos_ref - pos).ravel()

        ax_des = self.PID_x.get(pos_error[0])
        ay_des = self.PID_y.get(pos_error[1])
        az_des = self.PID_z.get(pos_error[2])
        a_des = np.vstack((ax_des, ay_des, az_des))

        F_des_i = self.m * (a_des - self.g * self.e3)
        return F_des_i

    def update(self, pos):
        t = self.clock.get()
        pos_ref = self.get_pos_ref(t)
        pos_error = (pos_ref - pos).ravel()
        self.PID_x.update(pos_error[0])
        self.PID_y.update(pos_error[1])
        self.PID_z.update(pos_error[2])


class QuadEnv(fym.BaseEnv):
    def __init__(self, env_config):
        super().__init__(**env_config["fkw"])
        self.plant = Multicopter(plant_config=env_config["plant_config"])

        # set the outer loop
        if env_config["outer_loop"] == "PID":
            self.outer = PIDOuterLoop(self)
        elif env_config["outer_loop"] == "fixed":
            self.outer = FixedOuterLoop()

        # -- FOR RL
        # observation: vz (1), eta (2), omega (3) -- TOTAL: 6
        self.observation_space = spaces.Box(
            low=np.float32(
                np.hstack(
                    [
                        env_config["observation_space"]["vz"]["low"],
                        env_config["observation_space"]["eta"]["low"],
                        env_config["observation_space"]["omega"]["low"],
                    ]
                )
            ),
            high=np.float32(
                np.hstack(
                    [
                        env_config["observation_space"]["vz"]["high"],
                        env_config["observation_space"]["eta"]["high"],
                        env_config["observation_space"]["omega"]["high"],
                    ]
                )
            ),
            dtype=np.float32,
        )
        # action: rotorfs (4)
        self.action_space = spaces.Box(
            low=0,
            high=self.plant.rotorf_max,
            shape=(4,),
            dtype=np.float32,
        )

        # reward (- LQR cost)
        self.LQR_Q = np.diag(env_config["LQR"]["Q"])
        self.LQR_R = np.diag(env_config["LQR"]["R"])
        # scaling reward for discretization
        self.reward_scale = self.clock.dt

        # -- TEST ENV
        self.pscale = env_config["perturb_scale"]

    def reset(self, mode="random"):
        """Reset the plant states.

        Parameters
        ----------
        mode : {"random", "neighbor", "initial"}
        """
        super().reset()

        if mode == "neighbor":
            angles = Rotation.from_matrix(self.plant.R.state).as_euler("ZYX")
            angles = np.deg2rad(
                np.random.normal(loc=angles, scale=self.pscale["angles"])
            )
            self.plant.pos.state = np.random.normal(
                loc=self.plant.pos.state,
                scale=self.pscale["pos"],
            )
            self.plant.vel.state = np.random.normal(
                loc=self.plant.vel.state,
                scale=self.pscale["vel"],
            )
            self.plant.R.state = Rotation.from_euler("ZYX", angles).as_matrix()
            self.plant.omega.state = np.random.normal(
                loc=self.plant.omega.state, scale=self.pscale["omega"]
            )
        elif mode == "random":
            obs = np.float64(self.observation_space.sample())
            pos, vel, R, omega = self.obs2state(obs)
            self.plant.pos.state = pos
            self.plant.vel.state = vel
            self.plant.R.state = R
            self.plant.omega.state = omega
        elif mode == "initial":
            pass
        else:
            raise ValueError

        obs = self.observation()
        assert self.observation_space.contains(obs), breakpoint()

        return obs

    def step(self, action):
        # -- BEFORE UPDATE
        obs = self.observation()
        pos = self.plant.pos.state

        # Update the action in line with the desired total thrust
        F_des_i = self.outer.get(pos)
        F_des_norm = np.linalg.norm(F_des_i)
        action = (F_des_norm - sum(action)) * np.ones(4) / 4 + action

        # -- UPDATE
        # ``info`` contains ``t`` and ``state_dict``
        _, done = self.update(action=action)
        self.outer.update(pos=pos)

        # -- AFTER UPDATE
        # get reward
        next_obs = self.observation()
        reward = self.get_reward(obs, action, next_obs)
        # get done
        bounds_out = not self.observation_space.contains(next_obs)
        done = done or bounds_out
        # get info
        info = {}
        return next_obs, reward, done, info

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

    def observation(self, dtype=None):
        # get states
        pos, vel, R, omega = self.plant.observe_list()

        # make obs
        vz = vel.ravel()[2]
        # get n_i
        F_des_i = self.outer.get(pos)
        F_des_norm = np.linalg.norm(F_des_i)
        n_i = F_des_i / max(F_des_norm, 1e-13)
        n_b = R.T @ n_i
        eta = n_b.ravel()[:2]
        omega = omega.ravel()
        obs = np.hstack((vz, eta, omega))

        # set dtype for RL policies
        dtype = dtype or np.float32
        return dtype(obs)

    def get_reward(self, obs, action, next_obs):
        bounds_out = not self.observation_space.contains(next_obs)
        if bounds_out:
            reward = -5000
        else:
            obs = obs[:, None]
            action = action[:, None]

            # hovering reward
            reward = -(
                +obs.T @ self.LQR_Q @ obs + action.T @ self.LQR_R @ action
            ).ravel()

        # scaling
        reward *= self.reward_scale
        return np.float32(reward)

    def eta2R(self, eta):
        n3 = np.vstack((eta, -np.sqrt(1 - (eta**2).sum())))
        e3 = np.vstack((0, 0, 1))

        RT = np.zeros((3, 3))
        e3xn3 = cross(e3, n3)
        if np.all(np.isclose(e3xn3, 0)):
            e3xn3 = np.vstack((0, 1, 0))
        RT[:, 0:1] = -cross(e3xn3, n3)
        RT[:, 1:2] = e3xn3
        RT[:, 2:3] = -n3

        return RT.T

    def obs2state(self, obs):
        """Convert observation to pos, vel, R, omega."""
        vz = obs[0]
        eta = obs[1:3][:, None]
        omega = obs[3:][:, None]

        pos = self.plant.pos.initial_state
        vel = np.vstack((0, 0, vz))
        R = self.eta2R(eta)
        return pos, vel, R, omega
