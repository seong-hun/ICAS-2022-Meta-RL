import math
import sys
from pathlib import Path
from itertools import chain

import fym
import matplotlib.animation
from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import art3d, proj3d
from scipy.spatial.transform import Rotation

logger.remove()
logger.add(sys.stderr, level="DEBUG")

show = plt.show


class Quadrotor:
    def __init__(self, ax, fault_index=None, body_diameter=0.315,
                 rotor_radius=0.15):
        d = body_diameter

        # Body
        body_segs = np.array([
            [[d, 0, 0], [0, 0, 0]],
            [[-d, 0, 0], [0, 0, 0]],
            [[0, d, 0], [0, 0, 0]],
            [[0, -d, 0], [0, 0, 0]]
        ])
        colors = (
            (1, 0, 0, 1),
            (0, 0, 1, 1),
            (0, 0, 1, 1),
            (0, 0, 1, 1),
        )

        self.body = art3d.Line3DCollection(
            body_segs, colors=colors, linewidth=2)

        kwargs = dict(radius=rotor_radius, ec="k", fc="k", alpha=0.3)
        self.rotors = [
            Circle((d, 0), **kwargs),
            Circle((0, d), **kwargs),
            Circle((-d, 0), **kwargs),
            Circle((0, -d), **kwargs),
        ]
        for index in fault_index:
            self.rotors.pop(index)

        ax.add_collection3d(self.body)
        for rotor in self.rotors:
            ax.add_patch(rotor)
            art3d.pathpatch_2d_to_3d(rotor, z=0)

        self.body._base = self.body._segments3d
        for rotor in self.rotors:
            rotor._segment3d = np.array(rotor._segment3d)
            rotor._center = np.array(rotor._center + (0,))
            rotor._base = rotor._segment3d

    def set(self, pos, R=np.eye(3)):
        # Rotate
        self.body._segments3d = np.array([
            R @ point
            for point in self.body._base.reshape(-1, 3)
        ]).reshape(self.body._base.shape)

        for rotor in self.rotors:
            rotor._segment3d = np.array([
                R @ point for point in rotor._base
            ])

        # Translate
        self.body._segments3d = self.body._segments3d + pos

        for rotor in self.rotors:
            rotor._segment3d = rotor._segment3d + pos


class FuncAnimation(matplotlib.animation.FuncAnimation):
    def __init__(self, fig, func=None, init_func=None, blit=True,
                 *args, **kwargs):
        self.fig = fig
        init_func = self.init_wrapper(init_func)
        func = self.func_wrapper(func)
        super().__init__(fig, func=func, init_func=init_func, blit=blit,
                         *args, **kwargs)

    def init_wrapper(self, init_func):
        def wrapper():
            if init_func is None:
                return

            self.iterable_of_artists = init_func() or []
            for ax in self.fig.axes:
                for name in ['collections', 'patches', 'lines',
                             'texts', 'artists', 'images']:
                    artist = getattr(ax, name)
                    for art in artist:
                        if art not in set(self.iterable_of_artists):
                            self.iterable_of_artists.append(art)

            return self.iterable_of_artists
        return wrapper

    def func_wrapper(self, func):
        def wrapper(frame):
            if func is None:
                return
            func(frame)
            return self.iterable_of_artists
        return wrapper

    def save(self, filename, writer="ffmpeg", *args, **kwargs):
        super().save(filename, writer=writer, *args, **kwargs)


def plot_mueller_solutions(path):
    df = pd.read_pickle(path)
    fault_indices = df.fault_index.unique()

    fig, axes = plt.subplots(
        3, len(fault_indices),
        sharex="col", sharey="row", squeeze=False,
        figsize=[5.06, 6.05])

    default_style = dict(color="k", lw=1.2)
    style = {
        "p": dict(ls="-", label="p"),
        "q": dict(ls=":", label="q"),
        "r": dict(ls="-.", label="r"),
        "u1": dict(ls="--", label=r"$f_1$"),
        "u2": dict(ls=":", label=r"$f_2$"),
        "u3": dict(ls=":", label=r"$f_3$"),
        "error": dict(ls="-"),
    }

    def get_plot(ax, dff, key):
        return ax.plot(dff.free_var, dff[key], **default_style | style[key])

    for col, fault_index in enumerate(fault_indices):
        ax = axes[0, col]
        ax.set_title(f"Fault rotors: {str(tuple(np.asarray(fault_index) + 1))}")

        dff = df[df.fault_index == fault_index]
        get_plot(ax, dff, "p")
        get_plot(ax, dff, "q")
        get_plot(ax, dff, "r")
        ax.set_ylim(-30, 50)
        ax.legend(ncol=3)

        ax = axes[1, col]
        ax.fault_index = fault_index
        get_plot(ax, dff, "u1")
        if fault_index == (3, ):
            get_plot(ax, dff, "u2")
            ax.set_xlabel(r"Motor thrust ratio $\| f_2 \| / \| f_1 \|$")
            ax.set_xlim(0, 1)
        if fault_index == (1, 3):
            get_plot(ax, dff, "u3")
            ax.set_xlabel(r"Motor thrust ratio $\| f_3 \| / \| f_1 \|$")
            ax.set_xlim(0, 1)
        elif fault_index == (2, 3):
            get_plot(ax, dff, "u2")
            ax.set_xlabel(r"Motor thrust ratio $\| f_2 \| / \| f_1 \|$")
            ax.set_xlim(0, 0.5)
        ax.legend(ncol=2)

        ax = axes[2, col]
        ax.fault_index = fault_index
        get_plot(ax, dff, "error")
        ax.set_yscale("log")

        # Each axis has the following information
        for ax in axes[:, col]:
            ax.fault_index = fault_index
            ax.xdata = ax.lines[0].get_xdata()
            ax.df = df[df.fault_index == fault_index]

    fig.tight_layout()
    return fig, axes


def resize_callback(event):
    figsize = event.canvas.figure.get_size_inches()
    print(f"figsize: [{figsize[0]}, {figsize[1]}]")


def get_close_xdata(xdata, x):
    index = min(np.searchsorted(xdata, x), len(xdata) - 1)
    return xdata[index], index


class SnappingCursor:
    """
    A cross hair cursor that snaps to the data point of a line, which is
    closest to the *x* position of the cursor.

    For simplicity, this assumes that *x* values of the data are sorted.
    """
    def __init__(self, axes, df):
        self.axes = axes
        self.canvas = axes.flatten()[0].figure.canvas

        for ax in axes.flat:
            ax.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')

        self._last_ax = None
        self._last_index = None

    def set_cross_hair_visible(self, visible, column_axes=None):
        need_redraw = False
        axes = self.axes.flat if column_axes is None else column_axes
        for ax in axes:
            need_redraw |= ax.vertical_line.get_visible() != visible
            ax.vertical_line.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        ax = event.inaxes
        if not ax:
            self._last_index = None
            self._last_ax = None
            if self.set_cross_hair_visible(False):
                self.canvas.draw()
        else:
            index = np.any(self.axes == ax, axis=0)
            column_axes = self.axes[:, index].ravel()

            self.set_cross_hair_visible(True, column_axes=column_axes)

            x, index = get_close_xdata(ax.xdata, event.xdata)

            if ax == self._last_ax and index == self._last_index:
                return  # still on the same axis and data point. Nothing to do.
            self._last_index = index
            self._last_ax = ax

            # update the line positions
            for cax in column_axes:
                cax.vertical_line.set_xdata(x)
            self.canvas.draw()


def get_hover_data(event):
    ax = event.inaxes
    free_var, _ = get_close_xdata(ax.xdata, event.xdata)
    fault_index = ax.fault_index
    df = ax.df[ax.df.free_var == free_var]

    hover_sol = {
        "fault_index": fault_index,
        "R": Rotation.from_euler(
            "ZYX",
            df[["psi", "theta", "phi"]].to_numpy().ravel()
        ).as_matrix(),
        "omega": df[[*"pqr"]].to_numpy().T,
        "rotors": df[["u1", "u2", "u3", "u4"]].to_numpy().T,
    }
    logger.info("Hover solution:")
    logger.info(f"    fault_index : {fault_index}")
    logger.info(f"    free_var    : {free_var:5.2f}")
    logger.info(f"    omega       : {hover_sol['omega'].T}")
    logger.info(f"    rotors      : {hover_sol['rotors'].T}")
    return hover_sol


class Runner:
    def __init__(self, env, cntr):
        self.env = env
        env.cntr = cntr
        path = Path("data/mueller") / "-".join([
            "run",
            env.__class__.__name__,
            cntr.__class__.__name__,
        ])
        self.env.logger = fym.Logger(path.with_suffix(".h5"))

    def on_press(self, event):
        hover_sol = get_hover_data(event)

        path = self.run(hover_sol)

        logger.info(f"Simulation data saved in ({path})")

        make_movie(path)

    def run(self, hover_sol):
        env = self.env
        env.set_hover_sol(hover_sol)
        env.logger.set_info(**hover_sol)

        env.reset()

        while not env.step():
            pass

        env.close()
        return env.logger.path


class Arrow3D(FancyArrowPatch):
    def __init__(self, ax, xs=[0, 0], ys=[0, 0], zs=[0, 0], *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        ax.add_artist(self)
        self.do_3d_projection = self.draw

    def set_positions_3d(self, posA, posB):
        self._verts3d = [vert for vert in np.vstack((posA, posB)).T]

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
        return min(zs) if zs.size else np.nan


def make_movie(path):
    data, info = fym.load(path, with_info=True)

    fault_index = (np.argmin(info["task"]["LoE"]),)

    fig = plt.figure()
    ax = plt.subplot(111, projection="3d")

    text = ax.text(0, 0, -1, "")
    quadrotor = Quadrotor(ax, fault_index=fault_index)
    supports, = plt.plot([], [], [], "k--", lw=0.8)

    # ndesi = Arrow3D(ax, mutation_scale=25, lw=1.2, arrowstyle="simple", color="b")
    # ni = Arrow3D(ax, mutation_scale=15, lw=1.2, arrowstyle="simple", color="k")
    # wi = Arrow3D(ax, mutation_scale=15, lw=1.2, arrowstyle="simple", color="b")

    rotors = [
        Arrow3D(ax, mutation_scale=15, lw=1, arrowstyle="simple", color="r")
        for _ in range(4 - len(fault_index))]

    def init_func():
        ax.set_xlim3d(-2, 2)
        ax.set_ylim3d(-2, 2)
        ax.set_zlim3d(-2, 2)
        ax.invert_zaxis()

    def func(i):
        text.set_text(f"time: {data['t'][i]:5.2f}")

        pos = data["plant"]["pos"][i].ravel()
        R = data["plant"]["R"][i]
        quadrotor.set(pos=pos, R=R)

        x, y, z = pos
        xlim = ax.get_xlim3d()[0]
        supports.set_data_3d(
            [xlim, xlim, xlim, x],
            [0, 0, y, y],
            [0, z, z, z]
        )

        # ndesi.set_positions_3d(pos, pos + data["cntr"]["ndesi"][i].ravel())
        # ni.set_positions_3d(pos, pos + 0.1 * data["cntr"]["ni"][i].ravel())
        # wi.set_positions_3d(pos, pos + 0.1 * data["cntr"]["wi"][i].ravel())

        for j in range(len(rotors)):
            base = pos + R @ quadrotor.rotors[j]._center
            fvec = np.array([0, 0, -data["plant_info"]["rotorfs"][i].ravel()[j]])
            rotors[j].set_positions_3d(base, base + 0.5 * R @ fvec)

    fps = 60
    dt = data["t"][1] - data["t"][0]  # 0.01
    step = math.floor(1 / dt / fps)
    frames = np.arange(0, len(data["t"]), step)
    interval = int(dt * step * 1e3)

    logger.info("Start animation.")
    anim = FuncAnimation(
        fig, func=func, init_func=init_func,
        frames=frames, interval=interval, blit=False)

    plt.show()

#     class Pbar:
#         def __init__(self):
#             self.pbar = None
#             self._i = 0

#         def anim_progress(self, i, total):
#             if self.pbar is None:
#                 self.pbar = tqdm(total=total)

#             self.pbar.update(i - self._i)
#             self._i = i

#         def close(self):
#             self.pbar.close()

#     anim_path = Path(path).with_suffix(".mp4")
#     logger.info("Saving the animation...")

#     pbar = Pbar()
#     anim.save(anim_path, progress_callback=pbar.anim_progress)
#     pbar.close()
#     logger.info(f"Animation saved in ({anim_path})")


def plot_hist(data):
    fig, axes = plt.subplots(4, 3, figsize=(12, 8), squeeze=False, sharex=True)
    env_data = data["env_info"]

    ax = axes[0, 0]
    ax.plot(env_data["t"], env_data["plant"]["pos"].squeeze(-1))
    ax.set_ylabel("Position, m")
    ax.legend([r"$x$", r"$y$", r"$z$"])

    ax = axes[1, 0]
    ax.plot(env_data["t"], env_data["plant"]["vel"].squeeze(-1))
    ax.set_ylabel("Velocity, m/s")
    ax.legend([r"$v_x$", r"$v_y$", r"$v_z$"])

    ax = axes[2, 0]
    angles = Rotation.from_matrix(env_data["plant"]["R"]).as_euler("ZYX")[:, ::-1]
    ax.plot(env_data["t"], np.rad2deg(angles))
    ax.set_ylabel("Angles, deg")
    ax.legend([r"$\phi$", r"$\theta$", r"$\psi$"])

    ax = axes[3, 0]
    ax.plot(env_data["t"], env_data["plant"]["omega"].squeeze(-1))
    ax.set_ylabel("Omega, rad/sec")
    ax.legend([r"$p$", r"$q$", r"$r$"])

    ax.set_xlabel("Time, sec")

    ax = axes[0, 1]
    ax.plot(env_data["t"], env_data["rotorfs_cmd"].squeeze(-1)[:, 0], "r--")
    ax.plot(env_data["t"], env_data["plant_info"]["rotorfs"].squeeze(-1)[:, 0], "k-")
    ax.plot(env_data["t"], data["agent_info"]["rotorfs0"][:, 0], "b-")
    ax.set_ylabel("Rotor 1 thrust, N")

    ax = axes[1, 1]
    ax.plot(env_data["t"], env_data["rotorfs_cmd"].squeeze(-1)[:, 1], "r--")
    ax.plot(env_data["t"], env_data["plant_info"]["rotorfs"].squeeze(-1)[:, 1], "k-")
    ax.plot(env_data["t"], data["agent_info"]["rotorfs0"][:, 1], "b-")
    ax.set_ylabel("Rotor 2 thrust, N")

    ax = axes[2, 1]
    ax.plot(env_data["t"], env_data["rotorfs_cmd"].squeeze(-1)[:, 2], "r--")
    ax.plot(env_data["t"], env_data["plant_info"]["rotorfs"].squeeze(-1)[:, 2], "k-")
    ax.plot(env_data["t"], data["agent_info"]["rotorfs0"][:, 2], "b-")
    ax.set_ylabel("Rotor 3 thrust, N")

    ax = axes[3, 1]
    ax.plot(env_data["t"], env_data["rotorfs_cmd"].squeeze(-1)[:, 3], "r--")
    ax.plot(env_data["t"], env_data["plant_info"]["rotorfs"].squeeze(-1)[:, 3], "k-")
    ax.plot(env_data["t"], data["agent_info"]["rotorfs0"][:, 3], "b-")
    ax.set_ylabel("Rotor 4 thrust, N")

    ax.set_xlabel("Time, sec")

    ax = axes[0, 2]
    ax.plot(env_data["t"], data["agent_info"]["F_des_b"].squeeze()[:, :3])
    ax.set_ylabel(r"$F_{des}^b$, N")
    ax.legend([r"$F_1$", r"$F_2$", r"$F_3$"])

    fig.tight_layout()


def icass2022():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "font.size": 8,
        "axes.labelsize": 7,
        "axes.grid": True,
        "axes.linewidth": 0.3,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.markersize": 3,
        "lines.linewidth": 1,
        "grid.linestyle": "--",
        "grid.alpha": 0.8,
        "grid.linewidth": 0.3,
    })

    adjust1 = dict(left=0.14, right=0.927, bottom=0.18)
    adjust2 = adjust1 | dict(bottom=0.12)

    # region FIGURE 1 ---------------------------------------------------------

    total_loss = fym.load("data/ICAS2022/total_loss.h5")["total_loss"]
    df = pd.DataFrame(total_loss)
    ewm = df.ewm(alpha=0.1, adjust=False).mean()

    fig = plt.figure("Total Loss", figsize=(3.5, 1.8))
    ax = fig.subplots()
    colors = plt.cm.get_cmap("tab20c").colors

    ax.plot(df, c=colors[3])
    ax.plot(ewm, c=colors[0])
    ax.set_ylabel("Total loss")
    ax.set_xlabel("Iteration")
    ax.set_xlim(0, len(df))
    ax.set_ylim(0.5, 3.5)

    fig.tight_layout()
    fig.subplots_adjust(**adjust1)

    fig.savefig("img/fig1.pdf", dpi=600, bbox_inches="tight", pad_inches=0.04)
    # plt.show()

    # endregion

    # region FIGURE 2 ---------------------------------------------------------

    data = fym.load("data/ICAS2022/good_result.h5")

    t = data["o"]["t"]
    omega = data["o"]["omega"].squeeze(-1)
    rotorfs = data["env_info"]["plant_info"]["rotorfs"].squeeze(-1)

    fig = plt.figure("Hover", figsize=(3.5, 2.8))
    axes = fig.subplots(2, 1, sharex=True)
    colors = plt.cm.get_cmap("Set1").colors

    my_cycler = (
        cycler(linestyle=["--", "-", "-.", ":"])
        * cycler(c="k")
    )

    ax = axes[0]
    ax.set_prop_cycle(my_cycler)
    ax.plot(t, omega[:, 0], label=r"$\omega_1$")
    ax.plot(t, omega[:, 1], label=r"$\omega_2$")
    ax.plot(t, omega[:, 2], label=r"$\omega_3$")
    # ax.plot(t, data["agent_info"]["omega0"][:, 0], color="k", linestyle="--")
    # ax.plot(t, data["agent_info"]["omega0"][:, 1], color="k", linestyle="--")
    # ax.plot(t, data["agent_info"]["omega0"][:, 2], color="k", linestyle="--")
    ax.set_ylabel("Angular velocuty, rad/sec")
    # ax.set_ylim(0.5, 3.5)
    ax.legend()

    ax = axes[1]
    ax.set_prop_cycle(my_cycler)
    ax.plot(t, rotorfs[:, 0], label=r"$f_1$")
    ax.plot(t, rotorfs[:, 1], label=r"$f_2$")
    ax.plot(t, rotorfs[:, 2], label=r"$f_3$")
    ax.plot(t, rotorfs[:, 3], label=r"$f_4$")
    # ax.plot(t, data["agent_info"]["rotorfs0"][:, 0], color="k", linestyle="--")
    # ax.plot(t, data["agent_info"]["rotorfs0"][:, 1], color="k", linestyle="--")
    # ax.plot(t, data["agent_info"]["rotorfs0"][:, 2], color="k", linestyle="--")
    # ax.plot(t, data["agent_info"]["rotorfs0"][:, 3], color="k", linestyle="--")
    ax.set_ylabel("Rotor forces, N")
    ax.legend(loc="upper right")

    ax.set_xlabel("Time, sec")
    ax.set_xlim(0, data["o"]["t"].max())

    fig.tight_layout()
    fig.subplots_adjust(**adjust2)

    fig.savefig("img/fig2.pdf", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.show()

    # endregion

    # dataset = {
    #     "4k": {
    #         "path": Path("data/ICAS2022/test/bad"),
    #         "style": {
    #             "c": "k",
    #         }
    #     },
    #     "9k": {
    #         "path": Path("data/ICAS2022/test/good"),
    #         "style": {
    #             "c": "b",
    #         },
    #     },
    # }

    # # df = pd.DataFrame()

    # plt.figure()
    # for k, v in dataset.items():
    #     for path in v["path"].iterdir():
    #         data, info = fym.load(path, with_info=True)
    #         omega0_hat = info["omega0_hat"]
    #         omega = data["o"]["omega"].squeeze(-1)
    #         omega_error = np.linalg.norm(omega - omega0_hat, axis=1)
    #         t = data["o"]["t"]
    #         plt.plot(t, omega_error, label=k, **v["style"])
    #         # df = pd.concat([
    #         #     df,
    #         #     pd.DataFrame({
    #         #         "time": (t * 1000).astype(int),
    #         #         "error": omega_error,
    #         #         "iteration": k,
    #         #     })
    #         # ], ignore_index=True)

    # # sns.lineplot(data=df, x="time", y="error", hue="iteration")
    # plt.show()


def main():
    icass2022()


if __name__ == "__main__":
    main()
