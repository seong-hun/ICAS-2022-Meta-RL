import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = nn.Linear(in_features, out_features, bias=False)

    def forward(self, input):
        out = self.model(input)
        return out


class Actor(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.K = nn.Linear(
            np.prod(env.observation_space.shape), np.prod(env.action_space.shape)
        )
        self.fc_logstd = nn.Sequential(
            nn.Linear(np.prod(env.observation_space.shape), 32),
            nn.Linear(32, np.prod(env.action_space.shape)),
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
        mean = self.K(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean


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
        self.learn_count = 0

        self.dt = config["env_config"]["fkw"]["dt"]
        self.s = config["s"]
        self.Q = torch.Tensor(np.diag(config["env_config"]["reward"]["Q"])).to(device)
        self.R = torch.Tensor(np.diag(config["env_config"]["reward"]["R"])).to(device)
        self.x_dim = np.prod(envs.observation_space.shape)

    def get_action(self, obs, deterministic=False):
        assert isinstance(obs, np.ndarray)
        if obs.ndim == 1:
            action, _, mean = self.actor(torch.Tensor(obs).to(device)[None])
            action = action.detach().cpu().numpy()[0]
            mean = mean.detach().cpu().numpy()[0]
        elif obs.ndim == 2:
            action, _, mean = self.actor(torch.Tensor(obs).to(device))
            action = action.detach().cpu().numpy()
            mean = mean.detach().cpu().numpy()
        else:
            raise ValueError

        if deterministic:
            return mean
        else:
            return action

    def learn(self, data):
        dx = data.observations
        du = data.actions
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

        self.learn_count += 1

        # write to writer
        logs = {
            "losses/q_loss": q_loss.item(),
            "losses/actor_loss": actor_loss.item(),
        }

        return logs
