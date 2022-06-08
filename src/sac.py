import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, env, policy_config):
        super().__init__()
        self.fc1 = nn.Linear(np.prod(env.observation_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))

        # action rescaling
        self.action_scale = torch.FloatTensor(
            (env.action_space.high - env.action_space.low) / 2.0
        )
        self.action_bias = torch.FloatTensor(
            (env.action_space.high + env.action_space.low) / 2.0
        )

        self.log_std_min = policy_config["log_std_min"]
        self.log_std_max = policy_config["log_std_max"]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
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
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod()
            + np.prod(env.action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAC(nn.Module):
    def __init__(self, envs, config):
        super().__init__()

        policy_config = config["policy_config"]["SAC"]

        if policy_config["dtype"] == "float32":
            self.dtype = torch.float32
        elif policy_config["dtype"] == "float64":
            self.dtype = torch.float64
        torch.set_default_dtype(self.dtype)

        self.actor = Actor(envs, policy_config)

        # setup networks
        self.qf1 = SoftQNetwork(envs)
        self.qf2 = SoftQNetwork(envs)
        self.qf1_target = SoftQNetwork(envs)
        self.qf2_target = SoftQNetwork(envs)

        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # setup optimizers
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=policy_config["q_lr"],
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()),
            lr=policy_config["policy_lr"],
        )

        # with autotune = True
        self.target_entropy = -torch.prod(torch.Tensor(envs.action_space.shape)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = optim.Adam([self.log_alpha], lr=policy_config["q_lr"])

        self.boundsout = config["env_config"]["reward"]["boundsout"]

        self.policy_config = policy_config
        self.learn_count = 0

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
        policy_config = self.policy_config

        observations = data.observations.to(self.dtype)
        actions = data.actions.to(self.dtype)
        next_observations = data.next_observations.to(self.dtype)

        gamma = 1 - policy_config["mgamma"]

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor(next_observations)
            qf1_next_target = self.qf1_target(next_observations, next_state_actions)
            qf2_next_target = self.qf2_target(next_observations, next_state_actions)
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            if self.boundsout is None:
                next_q_value = data.rewards.flatten() + gamma * (
                    min_qf_next_target
                ).view(-1)
            else:
                next_q_value = data.rewards.flatten() + gamma * (
                    1 - data.dones.flatten()
                ) * (min_qf_next_target).view(-1)

        qf1_a_values = self.qf1(observations, actions).view(-1)
        qf2_a_values = self.qf2(observations, actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if (
            self.learn_count % policy_config["policy_frequency"] == 0
        ):  # TD 3 Delayed update support
            for _ in range(
                policy_config["policy_frequency"]
            ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi, _ = self.actor(observations)
                qf1_pi = self.qf1(observations, pi)
                qf2_pi = self.qf2(observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                self.actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                self.actor_loss.backward()
                self.actor_optimizer.step()

                # with autotune = True
                with torch.no_grad():
                    _, log_pi, _ = self.actor(observations)
                self.alpha_loss = (
                    -self.log_alpha * (log_pi + self.target_entropy)
                ).mean()

                self.a_optimizer.zero_grad()
                self.alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha = self.log_alpha.exp().item()

        # update the target networks
        if self.learn_count % policy_config["target_network_frequency"] == 0:
            for param, target_param in zip(
                self.qf1.parameters(), self.qf1_target.parameters()
            ):
                target_param.data.copy_(
                    policy_config["tau"] * param.data
                    + (1 - policy_config["tau"]) * target_param.data
                )
            for param, target_param in zip(
                self.qf2.parameters(), self.qf2_target.parameters()
            ):
                target_param.data.copy_(
                    policy_config["tau"] * param.data
                    + (1 - policy_config["tau"]) * target_param.data
                )

        self.learn_count += 1

        # write to writer
        logs = {
            "losses/qf1_values": qf1_a_values.mean().item(),
            "losses/qf2_values": qf2_a_values.mean().item(),
            "losses/qf1_loss": qf1_loss.item(),
            "losses/qf2_loss": qf2_loss.item(),
            "losses/qf_loss": qf_loss.item() / 2.0,
            "losses/actor_loss": self.actor_loss.item(),
            "losses/alpha": self.alpha,
            "losses/alpha_loss": self.alpha_loss.item(),
        }

        return logs
