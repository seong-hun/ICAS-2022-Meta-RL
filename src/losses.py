import torch.nn as nn
import yaml

from src.meta import InnerRL

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class LossEG(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_RL = InnerRL(config["num_inner_iterations"])
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def loss_cnt(self, x_t, x_hat):
        loss = self.l1_loss(x_t.reshape(-1), x_hat.reshape(-1))
        return loss * config["cnt_weight"]

    def loss_lyap(self, x_hat, y_t):
        transits = y_t[..., :5], y_t[..., 5:9], 0, y_t[..., 9:]
        omega0_hat, rotorfs0_hat = x_hat[..., :3], x_hat[..., 3:]
        transits = self.inner_RL.linearize(transits, omega0_hat, rotorfs0_hat)
        result = self.inner_RL.train(transits)
        loss = self.inner_RL.loss(result)
        return loss * config["lyap_weight"]

    def loss_mch(self, e_hat, W_i):
        loss = self.l1_loss(W_i.reshape(-1), e_hat.reshape(-1))
        return loss * config["mch_weight"]

    def loss_adv(self, r_hat):
        return -r_hat.mean()

    def forward(self, x_t, x_hat, r_hat, e_hat, W_i, y_t):
        lyap = self.loss_lyap(x_hat, y_t)
        mch = self.loss_mch(e_hat, W_i)
        adv = self.loss_adv(r_hat)
        cnt = self.loss_cnt(x_t, x_hat)
        loss = (lyap + mch + adv + cnt).reshape(1)
        info = {
            "lyap": lyap.item(),
            "mch": mch.item(),
            "adv": adv.item(),
            "cnt": cnt.item(),
        }
        return loss, info


class LossD(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, r, r_hat):
        loss = self.relu(1 + r_hat) + self.relu(1 - r)
        loss = loss.mean()
        return loss.reshape(1)
