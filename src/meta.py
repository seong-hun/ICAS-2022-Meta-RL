import numpy as np
import torch
import yaml
from loguru import logger
from torch import nn
from tqdm import tqdm

from src.utils import arr2str, from_numpy, inertia

torch.set_default_dtype(torch.float64)

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def batch_kron(x, y, symmetric=False):
    kron = x[..., None] * y[..., None, :]  # Kron prod
    return kron.reshape(*kron.shape[:-2], -1)


class InnerRL(nn.Module):
    def __init__(self, iter_max, render=False):
        super().__init__()
        self.dt = config["dt"]
        self.Q = from_numpy(np.diag(config["Q"]))
        self.R = from_numpy(np.diag(config["R"]))
        self.iter_max = iter_max
        self.s = config["s"]

        self.l1_loss = nn.L1Loss()

        self.norm = config["norm"]
        for k, v in self.norm.items():
            self.norm[k] = from_numpy(v)

        self.tqdm_bar = None
        if render:
            self.tqdm_bar = tqdm(total=iter_max)

    def get_x(self, omega0):
        omega0norm = torch.linalg.norm(omega0, axis=-1, keepdims=True)
        n0 = -omega0 / omega0norm * torch.sign(omega0[:, -1:])

        eta0 = n0[:, :2]
        alpha0 = omega0

        x0 = torch.cat([eta0, alpha0], dim=-1)
        return x0

    def linearize(self, transits, omega0, rotorfs0):
        """Linearize transits with z."""
        x0 = self.get_x(omega0)
        u0 = rotorfs0

        x = transits[0]  # B, K, 5
        u = transits[1]  # B, K, 4
        next_x = transits[3]  # B, K, 5

        x0 = x0[:, None, :]  # B, 1, 5
        u0 = u0[:, None, :]  # B, 1, 4

        dx = x - x0  # B, K, 5
        du = u - u0

        rewards = -(
            +dx.unsqueeze(-2) @ self.Q @ dx[..., None]
            + du.unsqueeze(-2) @ self.R @ du[..., None]
        ).squeeze(-1)
        rewards = torch.exp(rewards) - 1

        xdot = (next_x - x) / self.dt
        return dx, du, rewards, xdot

    def train(self, transits, x0, u0, K_init=None, verbose=False):
        """Train inner RL"""
        x, u, xdot = transits

        dx = x - x0.unsqueeze(-2)
        du = u - u0.unsqueeze(-2)

        LQR_cost = (
            +dx.unsqueeze(-2) @ self.Q @ dx[..., None]
            + du.unsqueeze(-2) @ self.R @ du[..., None]
        ).squeeze(-1)
        LQR_cost = 1 - torch.exp(-LQR_cost)
        Y = -LQR_cost  # B, N, 1

        Phi1 = batch_kron(xdot, dx) + batch_kron(dx, xdot)  # B, N, n*n
        kron_xdot_du = batch_kron(xdot, du)  # B, N, nm

        B = dx.shape[0]
        n = dx.shape[-1]
        m = du.shape[-1]
        w1s = n * n
        w2s = n * m

        W1 = torch.zeros(B, 1, n, n).double()
        W3 = torch.zeros(B, 1, m, m).double()

        if K_init is None:
            K = torch.randn(B, 1, m, n).double()
        else:
            K = K_init

        for i in range(self.iter_max):
            if self.tqdm_bar is not None:
                self.tqdm_bar.update(1)

            udot = -self.s * (du + (K @ dx[..., None]).squeeze(-1)) - (
                K @ xdot[..., None]
            ).squeeze(
                -1
            )  # B, N, m

            Phi2 = 2 * (kron_xdot_du + batch_kron(dx, udot))  # B, N, nm
            Phi3 = batch_kron(udot, du) + batch_kron(du, udot)  # B, N, mm

            Phi = torch.cat((Phi1, Phi2, Phi3), axis=-1)  # B, N, nn + nm + mm

            with torch.no_grad():
                # B, nn + nm + mm, 1
                w = torch.linalg.pinv(Phi) @ Y

                w1 = w[:, None, :w1s, :]  # B, 1, nn, 1
                w2 = w[:, None, w1s : w1s + w2s, :]  # B, 1, nm, 1
                w3 = w[:, None, w1s + w2s :, :]  # B, 1, mm, 1

                W1 = w1.reshape(B, 1, n, n).transpose(2, 3)  # B, 1, m, n
                W2 = w2.reshape(B, 1, n, m).transpose(2, 3)  # B, 1, m, n
                W3 = w3.reshape(B, 1, m, m).transpose(2, 3)  # B, 1, m, n

                # Policy improvement
                K = torch.linalg.inv(W3) @ W2  # B, 1, m, n

                nan_index = torch.isnan(K)
                if torch.sum(nan_index) > 0:
                    K[nan_index] = torch.randn_like(K[nan_index]).double()

            if verbose and i % 100 == 0:
                logger.debug(f"Iter {i} | W1 inertia: {inertia(W1)}")
                logger.debug(f"Iter {i} |  K: {arr2str(K, fmt='{:7.4f}')}")
                logger.debug(f"Iter {i} | error: {torch.linalg.norm(Phi @ w - Y)}")

        result = {
            "K": K,  # B, 1, m, n
            "dx": dx,  # B, N, n
            "du": du,  # B, N, m
            "Phi": Phi,  # B, N, nn + nm + mm
            "Y": Y,  # B, N, 1
            "w": w,  # B, nn + nn + mm, 1
            "W1": W1,
            "W2": W2,
            "W3": W3,
        }
        return result

    def loss(self, result):
        loss = self.l1_loss(result["Phi"] @ result["w"], result["Y"])
        return loss
