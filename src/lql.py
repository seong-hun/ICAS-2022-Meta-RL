import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.K = nn.Linear(
            np.prod(env.observation_space.shape), np.prod(env.action_space.shape)
        )

        self.action_scale = torch.FloatTensor(
            (env.action_space.high - env.action_space.low) / 2.0
        )
        self.action_bias = torch.FloatTensor(
            (env.action_space.high + env.action_space.low) / 2.0
        )

        self.log_std_min = config["log_std_min"]
        self.log_std_max = config["log_std_max"]

    def forward(self, x):
        mean = -self.K(x)
        normal = torch.distributions.Normal(mean, 5)
        x_t = normal.sample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        return action, None, mean


class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.H = nn.Linear(
            np.prod(env.observation_space.shape) + np.prod(env.action_space.shape),
            np.prod(env.observation_space.shape) + np.prod(env.action_space.shape),
        )

    def forward(self, x, u):
        gradQ = self.H(torch.cat([x, u], dim=-1))
        return gradQ


class LQL(nn.Module):
    def __init__(self, envs, config):
        super().__init__()
        self.actor = Actor(envs, config)
        self.critic = Critic(envs)

        self.q_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config["hyperparameters"]["q_lr"],
        )
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config["hyperparameters"]["policy_lr"],
        )

        self.config = config

        self.dt = config["env_config"]["fkw"]["dt"]
        self.s = config["s"]
        self.Q = torch.Tensor(np.diag(config["env_config"]["reward"]["Q"])).to(device)
        self.R = torch.Tensor(np.diag(config["env_config"]["reward"]["R"])).to(device)
        self.x_dim = np.prod(envs.observation_space.shape)

        self.obs0 = envs.get_attr("obs0")[0]
        self.action0 = envs.get_attr("action0")[0]

    def get_action(self, obs, deterministic=False):
        assert isinstance(obs, np.ndarray)
        if obs.ndim == 1:
            action, _, mean = self.actor(torch.Tensor(obs - self.obs0).to(device)[None])
            action = action.detach().cpu().numpy()[0]
            mean = mean.detach().cpu().numpy()[0]
        elif obs.ndim == 2:
            action, _, mean = self.actor(torch.Tensor(obs - self.obs0).to(device))
            action = action.detach().cpu().numpy()
            mean = mean.detach().cpu().numpy()
        else:
            raise ValueError

        if deterministic:
            return mean + self.action0
        else:
            return action + self.action0

    def learn(self, data):
        dx = data.observations - torch.Tensor(self.obs0).to(device)
        du = data.actions - torch.Tensor(self.action0).to(device)
        xdot = (data.next_observations - data.observations) / self.dt

        # -- UPDATE CRITIC
        self.actor.requires_grad_(False)
        *_, udot_xdot = self.actor(xdot)
        *_, udot_dx = self.actor(dx)
        udot = udot_xdot - self.s * (du - udot_dx)

        self.critic.requires_grad_(True)
        gradQ = self.critic(dx, du)

        xudot = torch.cat([xdot, udot], dim=-1).unsqueeze(-2)
        gradQ_xudot = (xudot @ gradQ[..., None]).squeeze()

        LQR_cost = (
            +dx.unsqueeze(-2) @ self.Q @ dx[..., None]
            + du.unsqueeze(-2) @ self.R @ du[..., None]
        ).squeeze()
        # LQR_cost = - torch.exp(-LQR_cost)

        q_loss = F.l1_loss(2 * gradQ_xudot, -LQR_cost)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # -- UPDATE POLICY
        dx = dx.detach()

        self.actor.requires_grad_(True)
        *_, du = self.actor(dx)

        self.critic.requires_grad_(False)
        gradQ = self.critic(dx, du)
        graduQ = gradQ[..., self.x_dim :]

        actor_loss = graduQ.square().sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # write to writer
        logs = {
            "losses/qf_loss": q_loss.item(),
            "losses/actor_loss": actor_loss.item(),
        }

        return logs
