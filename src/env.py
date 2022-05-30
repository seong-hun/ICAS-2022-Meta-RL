import functools
import warnings

import fym
import gym
import numpy as np
import scipy.optimize
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
        self.task = None

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

    def find_NHS(self):
        def get_errors(x):
            """Nonlinear function to be zero.

            Parameters
            ----------
            x : array_like, (6 + nrotors,)
                The optimization variable. Composed of (angles, omega,
                rotorfs).  angles: phi, theta

            """
            m, g = self.m, self.g
            e3 = zB = self.e3

            phi, theta, omega, rotorfs = x[0], x[1], x[2:5], x[5:]
            R = Rotation.from_euler("ZYX", [0, theta, phi]).as_matrix()
            omega = omega[:, None]
            rotorfs = Full @ rotorfs[:, None]
            rotorfs = self.set_valid(0, rotorfs)

            pos = vel = np.zeros((3, 1))
            *_, domega = self.deriv(pos, vel, R, omega, rotorfs)

            fT = (self.B @ rotorfs)[0]

            omega_norm = np.linalg.norm(omega)
            zBomega = zB.T @ omega

            # Construct constraints
            e1 = fT - m * g * omega_norm / np.abs(zBomega)
            e2 = np.sign(zBomega) / omega_norm * R @ omega - e3
            e3 = domega

            errors = [np.square(e).sum() for e in [e1 * 10, e2 * 20, e3]]
            return errors

        assert self.task is not None, "Task must be set"

        fi = tuple(self.task["fi"])
        fv = self.task["fv"]

        if fi == ():
            NHS = {
                # NHS
                "omega": np.zeros((3, 1)),
                "rotorfs": self.m * self.g * np.ones((4, 1)) / 4,
                # equilibrium
                "obs": np.zeros(6),
                "action": np.zeros(4),
                # Aux
                "fi": fi,
                "fv": fv,
                "angles": [0, 0, 0],
                "R": np.eye(3),
                "opt_result": None,
                "errors": None,
            }
            return NHS

        Full = np.delete(np.eye(self.nrotors), fi, axis=1)
        n_normal = self.nrotors - len(fi)  # number of healthy rotors

        if fi == (0,):
            omega0 = [5, 0, 15]
            rotorfs0 = [4, 1, 4]
        elif fi == (1,):
            omega0 = [0, -5, -15]
            rotorfs0 = [4, 4, 1]
        elif fi == (2,):
            omega0 = [-5, 0, 15]
            rotorfs0 = [1, 4, 4]
        elif fi == (3,):
            omega0 = [0, 5, -15]
            rotorfs0 = [4, 1, 4]
        elif fi == (0, 1):
            omega0 = [-25, -10, -20]
            rotorfs0 = [12, 4]
        elif fi == (1, 2):
            omega0 = [-12, 6, 20]
            rotorfs0 = [6, 2]
        elif fi == (2, 3):
            omega0 = [20, 10, -20]
            rotorfs0 = [12, 4]
        elif fi == (0, 3):
            omega0 = [4, -25, 20]
            rotorfs0 = [12, 4]
        elif fi == (0, 2):
            omega0 = [0, -15, 15]
            rotorfs0 = [10, 4]
        elif fi == (1, 3):
            omega0 = [15, 1, -18]
            rotorfs0 = [10, 3]
        else:
            raise ValueError

        x0 = np.hstack(
            (
                np.deg2rad([1, 1]),  # angles (phi, theta, psi)
                omega0,
                rotorfs0,
            )
        )

        bounds = (
            2 * [np.deg2rad([-80, 80]).tolist()]
            + 3 * [(-70, 70)]  # omega
            + n_normal * [(self.rotorf_min, self.rotorf_max)]  # rotors
        )

        # u = x[5:8]
        ui = 5

        if len(fi) == 1:
            left_i = fi[0] % 3  # the rotor left to the fault rotor
            oppose_i = (left_i + 1) % 3
            right_i = (left_i + 2) % 3

            constraints = [
                {
                    "type": "eq",
                    "fun": lambda x: x[ui + oppose_i] / x[ui + left_i] - fv,
                },
                {"type": "eq", "fun": lambda x: x[ui + left_i] - x[ui + right_i]},
            ]
        elif len(fi) == 2:
            constraints = [
                {"type": "eq", "fun": lambda x: x[ui + 1] / x[ui] - fv},
            ]
        else:
            constraints = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = scipy.optimize.minimize(
                fun=lambda x: np.sum(get_errors(x)),
                x0=x0,
                method="SLSQP",
                tol=1e-10,
                bounds=bounds,
                constraints=constraints,
                options={},
            )

        x = result.x

        omega = x[2:5][:, None]
        rotorfs = Full @ x[5:][:, None]

        omega0 = omega.ravel()
        eta0 = (-omega0 / np.linalg.norm(omega0) * np.sign(omega0[-1]))[:2]
        obs = np.hstack((0, eta0, omega0))
        action = rotorfs.ravel()

        NHS = {
            # NHS
            "omega": omega,
            "rotorfs": rotorfs,
            # equilibrium
            "obs": obs,
            "action": action,
            # Aux
            "fi": fi,
            "fv": fv,
            "angles": [x[0], x[1], 0],  # phi, theta, psi
            "R": Rotation.from_euler("ZYX", [0, x[1], x[0]]).as_matrix(),
            "opt_result": result,
            "errors": get_errors(x),
        }
        return NHS


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
    def __init__(self, env):
        m = env.plant.m
        g = env.plant.g
        self.F_des_i = m * g * np.vstack((0, 0, -1))

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


class QuadEnv(fym.BaseEnv, gym.Env):
    def __init__(self, env_config):
        super().__init__(**env_config["fkw"])
        self.plant = Multicopter(plant_config=env_config["plant_config"])

        # set the outer loop
        if env_config["outer_loop"] == "PID":
            self.outer = PIDOuterLoop(self)
        elif env_config["outer_loop"] == "fixed":
            self.outer = FixedOuterLoop(self)

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
        self.LQR_Q = np.diag(env_config["reward"]["Q"])
        self.LQR_R = np.diag(env_config["reward"]["R"])
        # scaling reward for discretization
        self.reward_scale = self.clock.dt
        self.boundsout_reward = env_config["reward"]["boundsout"]
        self.obs0 = 0
        self.action0 = 0

        # -- TEST ENV
        self.reset_mode = env_config["reset"]["mode"]
        self.pscale = env_config["reset"]["perturb_scale"]

        # NHS
        self.NHS = {}

    def reset(self):
        """Reset the plant states.

        Parameters
        ----------
        mode : {"random", "neighbor", "initial"}
        """
        super().reset()

        if self.reset_mode == "neighbor":
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
        elif self.reset_mode == "random":
            obs = np.float64(self.observation_space.sample())
            pos, vel, R, omega = self.obs2state(obs)
            self.plant.pos.state = pos
            self.plant.vel.state = vel
            self.plant.R.state = R
            self.plant.omega.state = omega
        elif self.reset_mode == "initial":
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
        reward = self.get_reward(obs, action)
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

    def get_reward(self, obs, action):
        obs = (obs - self.obs0)[:, None]
        action = (action - self.action0)[:, None]

        # hovering reward
        reward = -(obs.T @ self.LQR_Q @ obs + action.T @ self.LQR_R @ action).ravel()

        # if self.boundsout_reward is None, then reward = - LQR cost
        # elif self.boundsout_reward set to some value,
        # then reward = self.boundsout_reward - LQR cost
        if self.boundsout_reward is not None:
            reward += self.boundsout_reward

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

    def set_NHS(self, NHS):
        assert self.NHS == {}
        self.NHS = NHS

        self.obs0 = NHS["obs"]
        self.action0 = NHS["action"]

        self.plant.R.initial_state = NHS["R"]
        self.plant.omega.initial_state = NHS["omega"]

    def get_mueller_params(self):
        def deriv(x, u):
            vz, eta, omega = x[0], x[1:3], x[3:]

            pos = np.zeros((3, 1))
            vel = np.vstack((0, 0, vz))
            R = self.eta2R(eta)
            rotors = hrotorfs + udist @ u
            rotors[(fi,)] = 0

            _, vel_dot, _, omega_dot = plant.deriv(pos, vel, R, omega, rotors)

            n3 = np.vstack((eta, -np.sqrt(1 - (eta**2).sum())))
            n_dot = -np.cross(omega, n3, axis=0)

            return np.vstack((vel_dot[2], n_dot[:2], omega_dot))

        assert self.plant.task is not None
        assert self.NHS is not None

        plant = self.plant
        fi = self.plant.task["fi"]
        hrotorfs = self.NHS["rotorfs"]

        # Number of healthy rotors
        N = plant.nrotors - len(fi)

        udist = functools.reduce(
            lambda p, q: np.insert(p, q, np.zeros(N - 1), axis=0),
            fi,
            np.vstack((np.eye(N - 1), -np.ones(N - 1))),
        )

        # LQR control gains
        Q = np.diag([5, 20, 20, 0, 0, 0])
        R = np.diag(np.ones(N - 1))

        # # Rotor forces without faulty rotors
        # partial_hover_rotorfs = np.delete(hrotorfs, fi, axis=0)

        # Linearized
        x0 = self.NHS["obs"][:, None]
        u0 = np.zeros((N - 1, 1))
        A = fym.jacob_analytic(deriv, 0)(x0, u0)[:, :, 0]
        B = fym.jacob_analytic(deriv, 1)(x0, u0)[:, :, 0]
        K, _ = fym.clqr(A, B, Q, R)

        return {
            "udist": udist,
            # "x0": x0,
            "K": K,
            # "Ccb": Ccb,
            # "nbdnc": nbdnc.ravel(),
            # "partial_hover_rotorfs": partial_hover_rotorfs,
        }
