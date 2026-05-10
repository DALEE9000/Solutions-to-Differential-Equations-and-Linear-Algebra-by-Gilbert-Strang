import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import scienceplots
import imageio_ffmpeg
plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

plt.style.use(['science', 'dark_background'])

CMAP = plt.cm.RdYlBu_r  # cool blue (cold) → pale yellow (mid) → dark red (hot)
n_x = 500
K   = 100

def make_mp4(x, t_vals, U_grid, filename, label, xticks=None, xticklabels=None):
    n_t  = len(t_vals)
    vmin = U_grid.min()
    vmax = U_grid.max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    def to_segments(y):
        pts = np.array([x, y]).T.reshape(-1, 1, 2)
        return np.concatenate([pts[:-1], pts[1:]], axis=1)

    def seg_colors(y):
        return (y[:-1] + y[1:]) / 2  # midpoint value per segment

    fig, ax = plt.subplots(figsize=(24, 16))
    lc = LineCollection(to_segments(U_grid[0]), cmap=CMAP, norm=norm, linewidth=5)
    lc.set_array(seg_colors(U_grid[0]))
    ax.add_collection(lc)

    pad = max(0.15 * (vmax - vmin), 0.1)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_xlabel('$x$', fontsize=36)
    ax.set_ylabel('$u(x,t)$', fontsize=36)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=32)
    else:
        ax.set_xticks([x[0], x[-1]])
        ax.set_xticklabels([f'${x[0]:.2g}$', f'${x[-1]:.2g}$'], fontsize=32)
    ax.tick_params(labelsize=32)
    ax.minorticks_off()
    fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
    fig.text(0.13, 0.92, label, fontsize=28, color='white', alpha=0.6)
    time_text = ax.set_title(f'$t = {t_vals[0]:.4f}$', fontsize=36)

    def update(i):
        lc.set_segments(to_segments(U_grid[i]))
        lc.set_array(seg_colors(U_grid[i]))
        time_text.set_text(f'$t = {t_vals[i]:.4f}$')

    anim = animation.FuncAnimation(fig, update, frames=n_t, interval=50)
    anim.save(filename, writer=animation.FFMpegFileWriter(fps=20))
    plt.close(fig)
    print(f"Saved {filename}")


# --- 8.3.2 ---
# U(x,t) = (1/sqrt(4*pi*c*t)) * exp(-x^2 / (4*c*t))
c = 1e-5
x = np.linspace(-0.01, 0.01, n_x)
t_vals = np.linspace(0.0001, 0.01, 100)
xg = x[None, :]
tg = t_vals[:, None]
U = (1 / np.sqrt(4*np.pi*c*tg)) * np.exp(-xg**2 / (4*c*tg))
make_mp4(x, t_vals, U, 'chap8sec8.3ex2.mp4', 'Ex 8.3.2')

# --- 8.3.4 ---
# U(x,t) = 1/2 + sum_{k=1}^{K} exp(-c*(2k-1)^2*pi^2*t) * (2/pi)*((-1)^k/(2k-1)) * cos((4k-2)*pi*x)
c = 1
x = np.linspace(-1, 1, n_x)
t_vals = np.linspace(0, 0.5, 100)
xg = x[None, None, :]
tg = t_vals[None, :, None]
k  = np.arange(1, K + 1)[:, None, None]
U = 0.5 + np.sum(
    np.exp(-c*(2*k-1)**2*np.pi**2*tg) * (2/np.pi) * ((-1)**k / (2*k-1)) * np.cos((4*k-2)*np.pi*xg),
    axis=0
)
make_mp4(x, t_vals, U, 'chap8sec8.3ex4.mp4', 'Ex 8.3.4')

# --- 8.3.5 ---
# U(x,t) = 1/(2*pi) + (1/pi) * sum_{k=1}^{K} exp(-k^2*pi^2*t) * cos(k*x)
x = np.linspace(-7, 7, n_x)
t_vals = np.linspace(0.001, 0.5, 100)
xg = x[None, None, :]
tg = t_vals[None, :, None]
k  = np.arange(1, K + 1)[:, None, None]
U = 1/(2*np.pi) + (1/np.pi) * np.sum(
    np.exp(-k**2*np.pi**2*tg) * np.cos(k*xg),
    axis=0
)
make_mp4(x, t_vals, U, 'chap8sec8.3ex5.mp4', 'Ex 8.3.5',
         xticks=np.array([-2, -1, 0, 1, 2]) * np.pi,
         xticklabels=[r'$-2\pi$', r'$-\pi$', '$0$', r'$\pi$', r'$2\pi$'])

# --- 8.3.6 ---
# U(x,t) = (1/sqrt(4*pi*t)) * (exp(-(x-a)^2/(4t)) - exp(-(x+a)^2/(4t)))
a = 1
x = np.linspace(-3, 3, n_x)
t_vals = np.linspace(0.001, 0.5, 100)
xg = x[None, :]
tg = t_vals[:, None]
U = (1 / np.sqrt(4*np.pi*tg)) * (np.exp(-(xg - a)**2 / (4*tg)) - np.exp(-(xg + a)**2 / (4*tg)))
make_mp4(x, t_vals, U, 'chap8sec8.3ex6.mp4', 'Ex 8.3.6')
