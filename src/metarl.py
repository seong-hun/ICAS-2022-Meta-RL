import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetaRL(nn.Module):
    def __init__(self):
        super().__init__()

        policy_config = config["policy_config"]["SAC"]

        if policy_config["dtype"] == "float32":
            self.dtype = torch.float32
        elif policy_config["dtype"] == "float64":
            self.dtype = torch.float64
        torch.set_default_dtype(self.dtype)

        self.x_dim = config["meta_train"]["x_dim"]
        self.e_dim = config["meta_train"]["e_dim"]
        self.u_dim = config["meta_train"]["u_dim"]
        self.c_dim = 2 * self.x_dim + self.u_dim
        self.z_dim = config["meta_train"]["z_dim"]
        self.Q = from_numpy(np.diag(config["meta_train"]["Q"]))
        self.R = from_numpy(np.diag(config["meta_train"]["R"]))
        self.s = config["meta_train"]["s"]

        self.embedder = networks.Embedder(
            self.c_dim, self.e_dim + self.u_dim + self.z_dim
        )
        self.policy = networks.Policy(self.x_dim, self.u_dim, self.z_dim)
        self.critic = networks.Critic(self.x_dim, self.u_dim, self.z_dim)

        self.optimizer_Q_E = Adam(
            list(self.embedder.parameters()) + list(self.critic.parameters())
        )
        self.optimizer_pi = Adam(self.policy.parameters())

    def load_models(self, path):
        ckpt = torch.load(path)
        self.load_state_dict(ckpt["model"])
        logger.debug(f"Model loaded: {path}")

    def save_models(self, path, **kwargs):
        self.eval()

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.state_dict(),
                "optim_Q_E": self.optimizer_Q_E.state_dict(),
                "optim_pi": self.optimizer_pi.state_dict(),
                **kwargs,
            },
            path,
        )
        logger.debug(f"Model saved: {path}")

        self.train()

    def infer(self, context):
        B, C = context.shape[:2]
        c = context.view(B * C, -1)  # B*C, c_dim
        e = self.embedder(c).reshape(B, C, -1)  # B, C, _z_dim
        e = e.mean(dim=1)  # B, _z_dim

        vz0 = torch.zeros((B, 1))
        omega0 = e[..., : self.e_dim]
        omeganorm = torch.clamp(
            torch.linalg.norm(omega0, dim=-1, keepdims=True), min=1e-13
        )
        eta0 = -omega0 / omeganorm * torch.sign(omega0[:, 2:])
        x0 = torch.hstack((vz0, eta0[:, :2], omega0))
        u0 = e[..., self.e_dim : self.e_dim + self.u_dim]
        z = e[..., self.e_dim + self.u_dim :]
        return x0, u0, z

    def learn(self, data):
        pass

    def get_action(self, obs):
        pass
