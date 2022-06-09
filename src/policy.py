import logging
from functools import reduce

import fym
import numpy as np
from fym.utils.rot import hat

from src.utils import arr2str

logger = logging.getLogger(__name__)


class RLAgent:
    def __init__(self, K, omega0, rotorfs0):
        self.policy = LinearPolicy(K=K)

        n0_b = - omega0 / np.linalg.norm(omega0) * np.sign(omega0[-1])
        eta0 = n0_b[:2]
        self.x0 = np.vstack((eta0, omega0))

        self.nbdnc = (np.hstack((0, 0, -1)) @ n0_b).ravel()
        self.omega0 = omega0
        self.rotorfs0 = rotorfs0

    def get_action(self, F_des_i, R, omega):
        F_des_b = R.T @ F_des_i

        F_des_norm = np.linalg.norm(F_des_b)
        n_des_b = F_des_b / max(F_des_norm, 1e-13)

        eta = n_des_b[:2]
        alpha = omega
        x = np.vstack([eta, alpha])

        dx = x - self.x0
        obs = dx.ravel()

        action, _ = self.policy.get_action(obs)
        du = action[:, None]

        fT_des = F_des_norm / self.nbdnc
        rotorfs0 = fT_des / sum(self.rotorfs0) * self.rotorfs0
        rotorfs = rotorfs0 + du

        agent_info = {
            "F_des_b": F_des_b.ravel(),
            "rotorfs0": self.rotorfs0.ravel(),
            "omega0": self.omega0.ravel(),
        }

        return rotorfs, agent_info


class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(*self.X.shape)
        self.X = self.X + dx
        return self.X


class PID:
    def __init__(self, kP, kI, kD, dt):
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.dt = dt

        self._prev_e = None
        self._prev_e_int = 0

    def get(self, e):
        prev_e = self._prev_e or e
        e_deriv = (e - prev_e) / self.dt
        e_int = self._prev_e_int

        u = self.kP * e + self.kI * e_int + self.kD * e_deriv

        if self.kI:
            self._prev_e_int = e_int + (prev_e + e) / 2 * self.dt  # trapz
        if self.kD:
            self._prev_e = e

        return u


class ConstantOuterLoop:
    def __init__(self, env):
        self.mg = env.plant.m * env.plant.g * env.plant.e3

    def get(self, pos, pos_ref):
        return -self.mg


class PIDOuterLoop:
    def __init__(self, env):
        # PID outer-loop controller
        self.PID_x = PID(kP=0.25, kI=0.006, kD=0.4, dt=env.clock.dt)
        self.PID_y = PID(kP=0.25, kI=0.006, kD=0.4, dt=env.clock.dt)
        self.PID_z = PID(kP=1, kI=0.1, kD=1, dt=env.clock.dt)

        # Constants
        self.m = env.plant.m
        self.g = env.plant.g
        self.e3 = env.plant.e3

    def get(self, pos, pos_ref):
        pos = pos.ravel()
        pos_ref = pos_ref.ravel()

        ax_des = self.PID_x.get(pos_ref[0] - pos[0])
        ay_des = self.PID_y.get(pos_ref[1] - pos[1])
        az_des = self.PID_z.get(pos_ref[2] - pos[2])
        a_des = np.vstack((ax_des, ay_des, az_des))

        F_des_i = self.m * (a_des - self.g * self.e3)
        return F_des_i


class OuterControlWrapper:
    def __init__(self, agent, outer, reference=None):
        self.agent = agent
        self.outer = outer
        self.reference = reference or self._reference

    def _reference(self, t):
        return np.vstack((0, 0, -2))

    def get_action(self, obs):
        t = obs["t"]
        R = obs["R"]
        omega = obs["omega"]

        pos = obs["pos"]
        pos_ref = self.reference(t)
        F_des_i = self.outer.get(pos, pos_ref)

        agent_info = {
            "pos_ref": pos_ref.ravel()
        }

        action, _agent_info = self.agent.get_action(F_des_i, R, omega)

        agent_info |= _agent_info

        return action, agent_info


class MuellerAgent:
    def __init__(self, env, NHS, random=False):
        self.env = env
        self.random = random
        self.noise = OUNoise((4, 1), sigma=0.05)

        # For different fault_index
        fault_index = env.fault_index

        # Controller parameter for a fault_index
        cntr_params = self.get_mueller_cntr_params(
            plant=env.plant,
            fault_index=fault_index,
            homega=NHS["omega"],
            hrotorfs=NHS["rotorfs"],
        )

        self.policy = LinearPolicy(K=cntr_params["K"])

        self.x0 = cntr_params["x0"]
        self.Ccb = cntr_params["Ccb"]
        self.rotorfs0 = reduce(
            lambda p, q: np.insert(p, q, 0, axis=0),
            fault_index,
            cntr_params["partial_hover_rotorfs"],
        )
        self.omega0 = NHS["omega"]
        self.nbdnc = cntr_params["nbdnc"]
        self.udist = cntr_params["udist"]

        self.NHS = NHS

        logger.debug(
            "Hover sol | "
            + "omega: {} [rad/s] | ".format(arr2str(NHS["omega"]))
            + "rotorfs: {} [N]".format(arr2str(NHS["rotorfs"]))
        )

    @staticmethod
    def get_mueller_cntr_params(env):
        def deriv(x, u):
            eta12, alp = x[:2], x[2:]
            eta = np.vstack((eta12, -1))

            pos = vel = np.zeros((3, 1))
            R = np.eye(3)
            omega = Ccb.T @ alp
            rotors = hrotorfs + udist @ u
            rotors[(fault_index,)] = 0
            *_, omega_dot = plant.deriv(pos, vel, R, omega, rotors)

            eta_dot = - np.cross(alp, eta, axis=0)
            alp_dot = Ccb @ omega_dot
            return np.vstack((eta_dot[:2], alp_dot))

        plant = env.plant
        fault_index = env.task["fault_index"]
        homega = env.task["NHS"]["omega"]
        hrotorfs = env.task["NHS"]["rotorfs"]
        # LQR control gains
        Q = np.diag([20, 20, 0, 0, 0])
        R = np.diag([1, 1])

        # ``nb`` is a unit vector in body-fixed coordinates parallel to
        # ``homega`` but pointing -z direction of inertial coords.
        nb = - homega / np.linalg.norm(homega) * np.sign(homega[-1])
        nc = np.vstack((0, 0, -1))

        # Ccb @ nb = nc
        nbcnc = np.cross(nb, nc, axis=0)
        h = hat(nbcnc)
        Ccb = np.eye(3) + h + h @ h * (1 - nb.T @ nc) / (nbcnc.T @ nbcnc)

        nbdnc = nc.T @ nb
        x0 = np.vstack((0, 0, Ccb @ homega))

        # Number of healthy rotors
        N = plant.nrotors - len(fault_index)

        udist = reduce(
            lambda p, q: np.insert(p, q, np.zeros(N - 1), axis=0),
            fault_index,
            np.vstack((np.eye(N - 1), -np.ones(N - 1))),
        )

        # Rotor forces without faulty rotors
        partial_hover_rotorfs = np.delete(hrotorfs, fault_index, axis=0)

        # Linearized
        u0 = np.zeros((N - 1, 1))
        A = fym.jacob_analytic(deriv, 0)(x0, u0)[:, :, 0]
        B = fym.jacob_analytic(deriv, 1)(x0, u0)[:, :, 0]
        K, _ = fym.clqr(A, B, Q, R)

        return {
            "udist": udist,
            "x0": x0,
            "K": K,
            "Ccb": Ccb,
            "nbdnc": nbdnc.ravel(),
            "partial_hover_rotorfs": partial_hover_rotorfs,
        }

    def get_action(self, F_des_i, R, omega):
        F_des_b = R.T @ F_des_i

        F_des_norm = np.linalg.norm(F_des_b)
        n_des_b = F_des_b / max(F_des_norm, 1e-13)

        eta = (self.Ccb @ n_des_b)[:2]  # in C-coords.
        alpha = self.Ccb @ omega  # in C-coords.
        x = np.vstack((eta, alpha))
        dx = x - self.x0

        obs = dx.ravel()
        action, _ = self.policy.get_action(obs)
        du = action[:, None]

        fT_des = F_des_norm / self.nbdnc
        rotorfs0 = fT_des * self.rotorfs0 / sum(self.rotorfs0)
        rotorfs = rotorfs0 + self.udist @ du

        if self.random:
            rotorfs += self.noise.sample()

        agent_info = {
            "F_des_b": F_des_b.ravel(),
            "rotorfs0": self.rotorfs0.ravel(),
            "omega0": self.omega0.ravel(),
        }
        return rotorfs, agent_info


class UntrimmedPolicyWrapper:
    def __init__(self, agent, omega0, rotorfs0):
        self.agent = agent

        n0_b = - omega0 / np.linalg.norm(omega0) * np.sign(omega0[-1])
        eta0 = n0_b[:2]
        self.x0 = np.vstack((eta0, omega0)).ravel()
        self.rotorfs0 = rotorfs0.ravel()

    def get_action(self, obs):
        dx = obs - self.x0
        du, agent_info = self.agent.get_action(dx)
        u = self.rotorfs0 + du
        return u, agent_info


class LinearPolicy:
    def __init__(self, K, random=False):
        self.K = K
        self.random = random
        self.noise = OUNoise(len(K), sigma=0.15)

    def get_action(self, obs):
        dx = obs[:, None]
        du = - (self.K @ dx).ravel()

        if self.random:
            du += self.noise.sample()

        return du.ravel(), {}
