import shutil
from datetime import datetime
from pathlib import Path

import fym
import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from loguru import logger
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src import networks
from src.dataset import TaskDataset
from src.utils import RunningAvg, from_numpy

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def save_model(model, path):
    model.eval()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.debug(f"Model saved: {path}")

    model.train()


class OverhaulGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = networks.Embedder()
        self.G = networks.Generator()

    def forward(self, context, transits):
        transit = [t[:, -1, ...] for t in transits]  # 3, B, dim
        c = context.view(-1, *context.shape[2:])  # B*K, ...

        e_vectors = self.E(c).reshape(*context.shape[:2], -1)  # B, K, e_dim
        e_hat = e_vectors.mean(dim=1)  # B, e_dim
        obs = transit[0]
        z_hat = self.G(obs, e_hat)  # B, z_dim
        return z_hat


def trainable(i):
    seed = i
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)

    run_start = datetime.now().strftime("%y-%m-%d_%H:%M:%S")
    expdir = Path(f"exp/{run_start}/trial_{i:02d}")

    writer = SummaryWriter(expdir / "log")
    flogger = fym.Logger(expdir / "train/history.h5", max_len=1)

    # region SETUP NETWORKS ---------------------------------------------------
    x_dim = config["x_dim"]
    u_dim = config["u_dim"]
    c_dim = 2 * x_dim + u_dim
    z_dim = config["z_dim"]
    Q = from_numpy(np.diag(config["Q"]))
    R = from_numpy(np.diag(config["R"]))
    s = config["s"]

    embedder = networks.Embedder(c_dim, x_dim + u_dim + z_dim)
    policy = networks.Policy(x_dim, u_dim, z_dim)
    critic = networks.Critic(x_dim, u_dim, z_dim)

    optimizer_Q_E = Adam(list(embedder.parameters()) + list(critic.parameters()))
    optimizer_pi = Adam(policy.parameters())

    # endregion

    # region SETUP DATASET ----------------------------------------------------
    src = Path(config["preprocess_path"])
    dst = expdir / src.name
    preprocess_path = shutil.copytree(src, dst)
    dataset = TaskDataset(preprocess_path, seed=seed)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    # endregion

    # region SET TENSORBOARD GRAPH --------------------------------------------
    # overhaul_model = OverhaulGraph()
    # _, context, transits, _ = next(iter(dataloader))
    # writer.add_graph(overhaul_model, [context, transits])
    # # endregion

    # region TRAINING LOOP ----------------------------------------------------

    global_step = 0
    running_loss = RunningAvg()

    logger.info("Training start")

    for epoch in range(config["num_epochs"]):
        epoch_start = datetime.now()

        embedder.train()
        policy.train()
        critic.train()

        for batch, (context, transits, NHS, _) in enumerate(dataloader):
            # context: B, C, c_dim(=14)
            # transits: 3 x (B, T, dim)
            # NHS: B, dim(=7)

            # region PROGRESS BATCH -------------------------------------------
            # Calculate average encoing vector for transits
            B, C = context.shape[:2]
            c = context.view(B * C, -1)  # B*C, c_dim
            _z = embedder(c).reshape(B, C, -1)  # B, C, _z_dim
            _z = _z.mean(dim=1)  # B, _z_dim

            x, u, xdot = transits

            # region UPDATE CRITIC/EMBEDDING ----------------------------------
            x0 = _z[..., :x_dim]
            u0 = _z[..., x_dim : x_dim + u_dim]
            z = _z[..., x_dim + u_dim :]
            dx = x - x0.unsqueeze(-2)
            du = u - u0.unsqueeze(-2)

            policy.requires_grad_(False)
            udot = policy(z, xdot) - s * (du - policy(z, dx))

            critic.requires_grad_(True)
            gradQ = critic(z, dx, du)

            xudot = torch.cat([xdot, udot], dim=-1).unsqueeze(-2)
            gradQ_xudot = (xudot @ gradQ[..., None]).squeeze()

            LQR_cost = (
                +dx.unsqueeze(-2) @ Q @ dx[..., None]
                + du.unsqueeze(-2) @ R @ du[..., None]
            ).squeeze()

            loss_critic = F.l1_loss(2 * gradQ_xudot, -LQR_cost)
            loss_critic += (z.abs() - 1).clamp(min=0).sum(dim=-1).mean()

            optimizer_Q_E.zero_grad()
            loss_critic.backward()
            optimizer_Q_E.step()
            # endregion

            # region UPDATE POLICY --------------------------------------------
            dx = dx.detach()
            z = z.detach()

            policy.requires_grad_(True)
            du = policy(z, dx)

            critic.requires_grad_(False)
            gradQ = critic(z, dx, du)
            graduQ = gradQ[..., x_dim:]

            loss_actor = graduQ.square().sum(dim=-1).mean()
            # endregion

            optimizer_pi.zero_grad()
            loss_actor.backward()
            optimizer_pi.step()
            # endregion

            # region SHOW PROGRESS --------------------------------------------
            loss = loss_actor + loss_critic

            running_loss.add(
                {
                    "Loss": {
                        "total": loss.item(),
                        "actor": loss_actor.item(),
                        "critic": loss_critic.item(),
                    },
                }
            )

            if global_step % config["stream_period"] == 0:
                logger.info(
                    f"({global_step}) "
                    f"Epoch {epoch + 1}: [{batch + 1}/{len(dataloader)}] | "
                    f"Loss: {loss.item():5.2e} | "
                )

            if global_step % config["log_period"] == 0:
                running_loss.to_writer(writer, global_step)
                flogger.record(global_step=global_step, **running_loss.get_avg())
                running_loss.reset()

            global_step += 1

        epoch_end = datetime.now()

        # region SAVE ---------------------------------------------------------
        if epoch % config["model_save_period"] == 0:
            savetime = epoch_end.strftime("%y-%m-%d_%H:%M:%S")
            modeldir = expdir / f"train/model/{savetime}"

            save_model(embedder, modeldir / "Embedder.pth")
            save_model(critic, modeldir / "Critic.pth")
            save_model(policy, modeldir / "Policy.pth")

            logger.debug(f"Epoch {epoch + 1} finished in {epoch_end - epoch_start}.")
        # endregion

    # endregion

    writer.close()
    flogger.close()


def train():
    @ray.remote
    def ray_trainable(i):
        return trainable(i)

    ray.init(num_cpus=10)
    futures = [ray_trainable.remote(i) for i in range(10)]
    ray.get(futures)
    ray.shutdown()


if __name__ == "__main__":
    train()
