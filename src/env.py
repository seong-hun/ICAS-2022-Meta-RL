import logging
import warnings

import fym
import numpy as np
import scipy.optimize
import yaml
from fym.utils.rot import hat
from loguru import logger as loguru_logger
from numpy.linalg import matrix_rank
from scipy.spatial.transform import Rotation

from src.utils import arr2str

logger = logging.getLogger(__name__)

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def cross(x, y):
    return np.cross(x, y, axis=0)


def perturb(system, size):
    if np.shape(size) == ():
        _perturb = size * np.random.randn(*system.state.shape)
    else:
        _perturb = size
    system.state = system.state + _perturb


def ctrb_matrix(A, B, normalized=False):
    assert (n := A.shape[0]) == A.shape[1]
    f = A.max() if normalized else 1
    C = np.hstack([np.linalg.matrix_power(A, i) / f**i @ B for i in range(n)])
    return C


def ctrb(A, B):
    C = ctrb_matrix(A, B, normalized=True)
    return matrix_rank(C) == A.shape[0]


class SimpleLinearEnv:
    def __init__(self):
        self.x = np.zeros(3)
        self.dt = config["dt"]
        self.t = 0

        self.A = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])
        self.B = np.array([[0, 1], [0, 0], [1, 0]])
        loguru_logger.debug(f"Controllability: {ctrb(self.A, self.B)}")

    def render(self):
        pass

    def reset(self, random=True):
        self.x = np.random.randn(3)
        self.t = 0
        return self.x.copy()

    def step(self, u):
        t = self.t
        next_x = (
            self.x + self.dt * (+self.A @ self.x[:, None] + self.B @ u[:, None]).ravel()
        )
        reward = [-(self.x**2).sum() - (u**2).sum()]
        done = self.t > 1
        info = {
            "t": t,
        }

        self.x = next_x
        self.t += self.dt
        return next_x, reward, done, info


class Multicopter(fym.BaseEnv):
    """S. Mallavalli and A. Fekih, “A fault tolerant tracking control for a
    quadrotor UAV subject to simultaneous actuator faults and exogenous
    disturbances,” International Journal of Control, vol. 93, no. 3, pp.
    655–668, Mar. 2020, doi: 10.1080/00207179.2018.1484173."""

    """ Physical constants """
    g = 9.81

    """ Physical properties """
    m = 1.00  # [kg] mass
    l = 0.24  # [m] torque arm
    _J = np.diag([8.1, 8.1, 14.2]) * 1e-3
    _Jr = np.diag([0, 0, 1.04]) * 1e-6
    b = 5.42e-5  # [N s^2 / rad^2] thrust coeff.
    d = 1.1e-6  # [N m s^2 / rad^2] reaction torque coeff.
    Kf = np.diag([5.567, 5.567, 6.354]) * 1e-4  # [N.s/m] drag coeff.
    Kt = np.diag([5.567, 5.567, 6.354]) * 1e-4  # [N.s/rad] aerodynamic drag
    rotorf_min = 0.0  # [N]
    rotorf_max = b * 523.0**2  # [N]

    """ Auxiliary constants """
    e3 = np.vstack((0, 0, 1))
    nrotors = 4

    """ Configuration
          ^ x
          |
         (0)
    - (3) + (1) -> y
         (2)
          |
    """
    # (F, M) = B @ rotorfs
    # rotorfs = b * rotorws**2; rotorws: rotor speed
    B = np.array(
        [[1, 1, 1, 1], [0, -l, 0, l], [l, 0, -l, 0], [-d / b, d / b, -d / b, d / b]]
    )

    def __init__(self):
        super().__init__()
        self.pos = fym.BaseSystem(np.vstack((0, 0, -2)))
        self.vel = fym.BaseSystem(np.zeros((3, 1)))
        self.R = fym.BaseSystem(np.eye(3))
        self.omega = fym.BaseSystem(np.zeros((3, 1)))
        self.J = self._J

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
            +M - cross(omega, self.J @ omega) - np.linalg.norm(omega) * self.Kt @ omega
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
        return self.get_Lambda(t) @ rotorfs

    def get_Lambda(self, t):
        Lambda = np.eye(self.nrotors)
        if t >= 0:  # Currently, the fault occurs at the start
            Lambda = self.LoE
        return Lambda

    def get_fault_index(self, t=0):
        return tuple(np.flatnonzero(np.diag(self.get_Lambda(t)) == 0))

    @property
    def J(self):
        return self._J

    @J.setter
    def J(self, J):
        self._J = J
        self._Jinv = np.linalg.inv(J)

    @property
    def Jinv(self):
        return self._Jinv


class Env(fym.BaseEnv):
    def __init__(self, max_t=10):
        super().__init__(max_t=max_t, dt=config["dt"])
        self.plant = Multicopter()

        # Define observation and action spaces for dimension references
        self.observation_dim = 6  # omega (3) and n_des (3)
        self.action_dim = 4  # each rotor forces

    def reset(self, random=True):
        super().reset()
        if random:
            perturb(self.plant.pos, 0.3)
            perturb(self.plant.vel, 0.5)
            angles = Rotation.from_matrix(self.plant.R.state).as_euler("ZYX")
            angles += np.deg2rad(10 * np.random.randn(3))
            self.plant.R.state = Rotation.from_euler("ZYX", angles).as_matrix()
            perturb(self.plant.omega, 4)
        return self.observation()

    def observation(self):
        return {"t": self.clock.get(), **self.plant.observe_dict()}

    def step(self, rotorfs_cmd):
        info, done = self.update(rotorfs_cmd=rotorfs_cmd)

        next_state = self.observation()
        reward = 0
        return next_state, reward, done or self.terminal(), info

    def terminal(self):
        pos, vel, R, omega = self.plant.observe_list()

        # Attitude bound condition
        _, theta, phi = np.rad2deg(Rotation.from_matrix(R).as_euler("ZYX"))
        if abs(phi) > 70 or abs(theta) > 70:
            loguru_logger.debug(
                "DONE: angles out | "
                f"phi: {phi:5.2f} [deg], theta: {theta:5.2f} [deg]"
            )
            return True

        if np.any(abs(omega) > 80):
            loguru_logger.debug(f"DONE: omega out | omega: {arr2str(omega)} [rad/s]")
            return True

        if pos[2] > 0:  # Fall to the ground
            loguru_logger.debug(f"DONE: altitude out | h: {-pos[2]} [m]")
            return True

        return False

    def get_reward(self, state, action, next_state):
        return 0

    def set_dot(self, t, rotorfs_cmd):
        plant_info = self.plant.set_dot(t, rotorfs_cmd)
        return dict(
            t=t,
            **self.observe_dict(),
            rotorfs_cmd=rotorfs_cmd,
            plant_info=plant_info,
            obs=self.observation(),
        )

    def set_task(self, task):
        self.plant.LoE = np.diag(task["LoE"])
        self.plant.m *= task["LoM"]
        self.plant.J = np.diag(task["LoJ"]) @ self.plant.J
        self.task = dict(
            task,
            fault_index=(np.argmin(task["LoE"]),),
            NHS=self.find_NHS(freevar=task["freevar"]),
        )

    def get_random_task(self):
        LoE = np.ones(4)
        LoE[np.random.randint(4)] = 0.2 * np.random.rand()
        return {
            "LoE": LoE,  # Loss of effectiveness
            "LoM": np.random.uniform(0.95, 1),  # Loss of mass
            "LoJ": np.random.uniform(0.95, 1, size=3),  # Loss of J
            "freevar": config["freevar_max"] * np.random.rand(),
        }

    def find_NHS(self, freevar=0.4):
        def fun(x):
            """Nonlinear function to be zero.

            Parameters
            ----------
            x : array_like, (6 + nrotors,)
                The optimization variable. Composed of (angles, omega,
                rotorfs).  angles: phi, theta

            """
            m, g = plant.m, plant.g
            e3 = zB = plant.e3

            phi, theta, omega, rotorfs = x[0], x[1], x[2:5], x[5:]
            R = Rotation.from_euler("ZYX", [0, theta, phi]).as_matrix()
            omega = omega[:, None]
            rotorfs = Full @ rotorfs[:, None]
            rotorfs = plant.set_valid(0, rotorfs)

            pos = vel = np.zeros((3, 1))
            *_, domega = plant.deriv(pos, vel, R, omega, rotorfs)

            fT = (plant.B @ rotorfs)[0]

            omega_norm = np.linalg.norm(omega)
            zBomega = zB.T @ omega

            # Construct constraints
            consts = []

            # Constraint 1
            consts.append(fT - m * g * omega_norm / np.abs(zBomega))

            # Constraint 2
            consts.append(np.sign(zBomega) / omega_norm * R @ omega - e3)

            # Constraint 3
            consts.append(domega)

            consts_raveled = np.hstack(list(map(np.ravel, consts)))
            error = consts_raveled**2 * [
                1,  # for const 1
                100,
                100,
                100,  # for const 2
                1,
                1,
                1,  # for const3
            ]
            # if error < 80:
            #     breakpoint()
            return np.sum(error)

        plant = self.plant
        fi = np.argmin(np.diag(plant.LoE))  # single fault_index

        Full = np.delete(np.eye(plant.nrotors), fi, axis=1)
        n_normal = plant.nrotors - 1
        rotorfs = np.ones(n_normal) * plant.m * plant.g / n_normal

        x0 = np.hstack(
            (
                np.deg2rad([10, 10]),  # angles (phi, theta, psi)
                np.array([1, 1, 20 * (-1) ** fi]),  # omega
                rotorfs,
            )
        )

        bounds = (
            2 * [np.deg2rad([-80, 80])]  # phi, theta
            + 3 * [(-50, 50)]  # omega
            + n_normal * [(plant.rotorf_min, plant.rotorf_max)]  # rotors
        )

        # u = x[5:8]
        ui = 5
        left_i = fi % 3
        oppose_i = (left_i + 1) % 3
        right_i = (left_i + 2) % 3

        constraints = [
            {
                "type": "eq",
                "fun": lambda x: x[ui + oppose_i] / x[ui + left_i] - freevar,
            },
            {"type": "eq", "fun": lambda x: x[ui + left_i] - x[ui + right_i]},
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = scipy.optimize.minimize(
                fun=fun,
                x0=x0,
                method="SLSQP",
                tol=1e-10,
                bounds=bounds,
                constraints=constraints,
                options={},
            )
        x = result.x

        NHS = {
            # NHS
            "omega": x[2:5][:, None],
            "rotorfs": Full @ x[5:][:, None],
            # Aux
            "fault_index": (fi,),
            "freevar": freevar,
            "angles": [x[0], x[1], 0],  # phi, theta, psi
            "R": Rotation.from_euler("ZYX", [0, x[1], x[0]]).as_matrix(),
            "opt_result": result,
        }
        return NHS


class EnvWrapper:
    def __init__(self, env):
        self.env = env
        self.plant = env.plant
        self.clock = env.clock

        self.Q = np.diag(config["Q"])
        self.R = np.diag(config["R"])

    def set_task(self, task):
        self.env.set_task(task)
        self.task = self.env.task

    def get_random_task(self):
        return self.env.get_random_task()

    def observation(self):
        return self.obs_wrapper(self.env.observation())

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.obs_wrapper(obs)


class RLEnvWrapper(EnvWrapper):
    def __init__(self, env, except_near=False):
        super().__init__(env)
        self.except_near = except_near

    def obs_wrapper(self, obs):
        R = obs["R"]
        omega = obs["omega"]

        F_des_i = -self.plant.m * self.plant.g * self.plant.e3
        F_des_b = R.T @ F_des_i
        F_des_norm = np.linalg.norm(F_des_b)
        n_des_b = F_des_b / max(F_des_norm, 1e-13)

        eta = n_des_b[:2]
        alpha = omega
        x = np.vstack(
            (
                eta,
                alpha,
            )
        )
        obs = x.ravel()
        return obs

    def action_wrapper(self, action):
        rotorfs = action[:, None]
        return rotorfs

    def step(self, action):
        rotorfs = self.action_wrapper(action)
        next_obs, reward, done, info = self.env.step(rotorfs)
        next_o = self.obs_wrapper(next_obs)
        if self.except_near:
            o = self.obs_wrapper(info["obs"])
            near = np.sum((next_o - o) ** 2) < 1e-6
            done = done or near
        return next_o, reward, done, info


class TrimmedEnvWrapper(EnvWrapper):
    def __init__(self, env, omega0, rotorfs0):
        super().__init__()
        n0_b = -omega0 / np.linalg.norm(omega0) * np.sign(omega0[-1])
        eta0 = n0_b[:2]
        self.x0 = np.vstack((eta0, omega0)).ravel()
        self.rotorfs0 = rotorfs0.ravel()

    def obs_wrapper(self, obs):
        return obs - self.x0

    def action_wrapper(self, action):
        rotorfs = self.rotorfs0 + action
        return rotorfs

    def step(self, action):
        obs = self.obs_wrapper(self.env.observation())
        rotorfs = self.action_wrapper(action)
        next_obs, _, done, info = self.env.step(rotorfs)
        reward = self.get_reward(obs, action)
        return self.obs_wrapper(next_obs), reward, done, info

    def get_reward(self, dx, du):
        reward = -(
            +dx[None, :] @ self.Q @ dx[:, None] + du[None, :] @ self.R @ du[:, None]
        )
        reward = np.exp(reward)
        return reward.ravel()
