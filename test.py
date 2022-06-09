import sys
from pathlib import Path

import fym
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from scipy.spatial.transform import Rotation
from sklearn import manifold
from torch.utils.data import DataLoader

from src import meta, networks
from src.dataset import TaskDataset
from src.env import Env
from src.policy import OuterControlWrapper, PIDOuterLoop, RLAgent
from src.utils import arr2str, rollout, to_numpy

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def load_model(model, traindir):
    filepath = traindir / f"{type(model).__name__}.pth"
    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict)
    return model


def plot(testpath=None):
    if testpath is None:
        testpath = Path(sys.argv[1])

    data, info = fym.load(testpath, with_info=True)
    if type(data) is not dict:
        raise ValueError("data should be dict")

    LoE = arr2str(info["task"]["LoE"], "{:5.3f}").replace(", ", "_")
    logger.info(f"LoE: {LoE}")

    data = data["env_info"]

    fig, axes = plt.subplots(4, 2, figsize=(10, 5.8), squeeze=False, sharex=True)

    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"].squeeze(-1))
    ax.axhline(-2, color="r", ls="--")
    ax.set_ylabel("Position, m")
    ax.legend([r"$x$", r"$y$", r"$z$", "$z_d$"])
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["vel"].squeeze(-1))
    ax.set_ylabel("Velocity, m/s")
    ax.legend([r"$v_x$", r"$v_y$", r"$v_z$"])

    ax = axes[2, 0]
    angles = Rotation.from_matrix(data["plant"]["R"]).as_euler("ZYX")[:, ::-1]
    ax.plot(data["t"], np.rad2deg(angles))
    ax.set_ylabel("Angles, deg")
    ax.legend([r"$\phi$", r"$\theta$", r"$\psi$"])

    ax = axes[3, 0]
    ax.plot(data["t"], data["plant"]["omega"].squeeze(-1))
    ax.set_ylabel("Angular velocity, rad/s")
    ax.legend([r"$p$", r"$q$", r"$r$"])

    ax.set_xlabel("Time, sec")

    ax = axes[0, 1]
    ax.plot(data["t"], data["rotorfs_cmd"].squeeze(-1)[:, 0], "r--")
    ax.plot(data["t"], data["plant_info"]["rotorfs"].squeeze(-1)[:, 0], "k-")
    ax.set_ylabel("Rotor 1 thrust, N")

    ax = axes[1, 1]
    ax.plot(data["t"], data["rotorfs_cmd"].squeeze(-1)[:, 1], "r--")
    ax.plot(data["t"], data["plant_info"]["rotorfs"].squeeze(-1)[:, 1], "k-")
    ax.set_ylabel("Rotor 2 thrust, N")

    ax = axes[2, 1]
    ax.plot(data["t"], data["rotorfs_cmd"].squeeze(-1)[:, 2], "r--")
    ax.plot(data["t"], data["plant_info"]["rotorfs"].squeeze(-1)[:, 2], "k-")
    ax.set_ylabel("Rotor 3 thrust, N")

    ax = axes[3, 1]
    ax.plot(data["t"], data["rotorfs_cmd"].squeeze(-1)[:, 3], "r--")
    ax.plot(data["t"], data["plant_info"]["rotorfs"].squeeze(-1)[:, 3], "k-")
    ax.set_ylabel("Rotor 4 thrust, N")
    ax.legend(["Command"])

    ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.align_ylabels(axes)

    plt.show()

    if data["t"][-1] < 15:
        logger.warning("Figure will not be saved")
    else:
        fig.savefig(f"img/response_{LoE}.pdf", bbox_inches="tight")


def test_seen_task():
    trialdir = Path(sys.argv[1])
    traindir = sorted((trialdir / "train/model").iterdir())[-1]
    testdir = Path(str(traindir).replace("train", "test")) / "seen"
    testdir.mkdir(parents=True, exist_ok=True)
    testpath = testdir / "data.h5"

    dataset = TaskDataset("data/preprocess")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    x_dim = config["x_dim"]
    u_dim = config["u_dim"]
    c_dim = 2 * x_dim + u_dim
    z_dim = config["z_dim"]

    embedder = networks.Embedder(c_dim, x_dim + u_dim + z_dim)
    policy = networks.Policy(x_dim, u_dim, z_dim)
    critic = networks.Critic(x_dim, u_dim, z_dim)

    load_model(embedder, traindir)
    load_model(policy, traindir)
    load_model(critic, traindir)

    embedder.eval()
    policy.eval()

    context, transits, NHS, idx = next(iter(dataloader))

    with torch.no_grad():
        B, C = context.shape[:2]
        c = context.view(B * C, -1)  # B*C, c_dim
        _z = embedder(c).reshape(B, C, -1)  # B, C, _z_dim
        _z = _z.mean(dim=1)  # B, _z_dim

        x0 = _z[..., :x_dim]
        u0 = _z[..., x_dim : x_dim + u_dim]
        z = _z[..., x_dim + u_dim :]

    taskfile = dataset.taskfiles[idx]
    taskdata = torch.load(taskfile)
    task = taskdata["task"]
    NHS = task["NHS"]

    env = Env(max_t=config["test"]["max_t"])
    env.set_task(task)

    # region INNER RL ---------------------------------------------------------
    K_init = policy.K(z).reshape(1, 1, u_dim, x_dim)
    inner_RL = meta.InnerRL(config["test"]["num_epochs"])
    result = inner_RL.train(transits, x0, u0, K_init=K_init, verbose=True)
    K = to_numpy(result["K"].squeeze())
    # endregion

    omega0_hat = to_numpy(x0[0, 2:])[:, None]
    rotorfs0_hat = to_numpy(u0[0])[:, None]

    agent = RLAgent(
        K=K,
        omega0=omega0_hat,
        rotorfs0=rotorfs0_hat,
    )

    agent = OuterControlWrapper(
        agent=agent,
        outer=PIDOuterLoop(env),
    )

    # Initialize env near the hover solution
    omega0_t = NHS["omega"]
    rotorfs0_t = NHS["rotorfs"]

    logger.info("Task | " + "LoE: {}".format(arr2str(task["LoE"])))
    logger.info(
        "Trained sol | "
        + "omega: {} [rad/s] | ".format(arr2str(omega0_hat))
        + "rotorfs: {} [N]".format(arr2str(rotorfs0_hat))
    )
    logger.info(
        "  True sol  | "
        + "omega: {} [rad/s] | ".format(arr2str(omega0_t))
        + "rotorfs: {} [N]".format(arr2str(rotorfs0_t))
    )

    env.plant.R.initial_state = NHS["R"]
    env.plant.omega.initial_state = omega0_t

    flogger = fym.Logger(testpath)
    flogger.set_info(
        task=task,
        K=K,
        omega0_hat=omega0_hat,
        rotorfs0_hat=rotorfs0_hat,
        omega0_t=omega0_t,
        rotorfs0_t=rotorfs0_t,
    )

    rollout(
        env,
        agent,
        flogger=flogger,
        reset_random=True,
        render=True,
    )

    flogger.close()

    logger.info(f"Test file was saved in {testpath}.")

    plot(testpath)


def embedder_result():
    trialdir = Path(sys.argv[1])
    traindir = sorted((trialdir / "train/model").iterdir())[-1]

    dataset = TaskDataset("data/preprocess")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    x_dim = config["x_dim"]
    u_dim = config["u_dim"]
    c_dim = 2 * x_dim + u_dim
    z_dim = config["z_dim"]

    embedder = networks.Embedder(c_dim, x_dim + u_dim + z_dim)
    load_model(embedder, traindir)
    embedder.eval()

    X = []
    flags = []
    freevars = []
    omegas = []

    for batch, (context, _, _, idx) in enumerate(dataloader):
        B, C = context.shape[:2]
        c = context.view(B * C, -1)  # B*C, c_dim
        _z = embedder(c).reshape(B, C, -1)  # B, C, _z_dim
        _z = _z.mean(dim=1)  # B, _z_dim

        taskfile = dataset.taskfiles[idx]
        taskdata = torch.load(taskfile)

        task = taskdata["task"]
        fault_index = task["fault_index"]
        freevar = task["freevar"]

        z = _z[..., x_dim + u_dim :]
        X.append(to_numpy(z))
        flags.append([fault_index[0]] * B)
        freevars.append(freevar)
        omegas.append(task["NHS"]["omega"].ravel())

    X = np.vstack(X)
    flags = np.hstack(flags)
    freevars = np.hstack(freevars)
    omegas = np.vstack(omegas)

    logger.info("Start t-SNE fitting")
    tsne = manifold.TSNE(**config["tsne"])
    Y = tsne.fit_transform(X)
    logger.info("Complete t-SNE fitting")

    cmap = plt.get_cmap("Dark2")(range(4))
    # cmap = np.array(sns.color_palette("hls", 4))

    fig = plt.figure(figsize=[4, 4])

    if config["tsne"]["n_components"] == 3:
        ax = fig.add_subplot(projection="3d", aspect="equal")
    else:
        ax = fig.add_subplot(aspect="equal")

    for fidx in range(4):
        idx = flags == fidx
        label = f"Rotor {fidx}"
        color = np.tile(cmap[fidx][:3], (sum(idx), 1))
        alpha = freevars[idx][:, None] + 0.2
        rgba = np.hstack((color, alpha))
        ax.scatter(*Y[idx].T, c=rgba, label=label, lw=0, s=40)

        # xtext, ytext = np.median(Y[idx, :], axis=0)
        # ax.text(xtext + 4, ytext + 6, label, fontsize=12, color=cmap[fidx][:3])
        # txt.set_path_effects(
        #     [patheffects.stroke(linewidth=5, foreground="w"), patheffects.normal()]
        # )
        # txts.append(txt)

    # ax.legend()
    ax.axis("off")
    ax.axis("tight")
    fig.savefig("img/embedding.pdf")

    plt.show()


def plot_hist():
    color = plt.get_cmap("Set1")(1)

    expdir = Path(sys.argv[1])
    trials = sorted(expdir.glob("trial_*"))
    dataset = {trial.name: fym.load(trial / "train/history.h5") for trial in trials}

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.7))

    for ax, key in zip(axes, ["actor", "critic"]):
        df = pd.concat(
            [
                pd.DataFrame(
                    {k: (np.ravel(v["Loss"][key]))},
                    index=v["global_step"],
                )
                for k, v in dataset.items()
            ],
            axis=1,
        )
        # df = df.loc[:, df.std() < 10]

        x = df.index
        mean = df.mean(axis=1)
        std = df.std(axis=1)

        ax.plot(x, mean, color=color)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

        ticker = matplotlib.ticker.EngFormatter(unit="")
        ax.xaxis.set_major_formatter(ticker)
        ax.set_yscale("log")
        ax.set_ylabel(f"{key.title()} loss")
        ax.set_xlabel("Number of iterations")
        # ax.set_ylim(-25, 1)

    fig.tight_layout()

    plt.show()

    fig.savefig("img/history.pdf")


def main():
    # seed = np.random.randint(10000)
    # logger.info(f"Seed: {seed}")
    seed = 3690
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.set_default_dtype(torch.float64)

    # plot_hist()
    # embedder_result()
    test_seen_task()
    # plot()


if __name__ == "__main__":
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Times New Roman", size=10)
    plt.rc("axes", labelsize=8, grid=True, linewidth=0.5)
    plt.rc("grid", linestyle="--", alpha=0.8, linewidth=0.5)
    plt.rc("legend", fontsize=8)
    plt.rc("lines", markersize=3, linewidth=1.2)

    main()
