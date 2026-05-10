import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scienceplots
import imageio_ffmpeg
plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

plt.style.use(['science', 'dark_background'])

n_x    = 500
t_vals = np.arange(0, 2.01, 0.01)
n_t    = len(t_vals)
c      = 1
K      = 100

def make_mp4(x, U_grid, filename, L, label):
    fig, ax = plt.subplots(figsize=(24, 16))
    line, = ax.plot(x, U_grid[0], color='#29ABE2', linewidth=5)
    pad = max(0.15 * (U_grid.max() - U_grid.min()), 0.1)
    ax.set_xlim(0, L)
    ax.set_ylim(U_grid.min() - pad, U_grid.max() + pad)
    ax.set_xlabel('$x$', fontsize=36)
    ax.set_ylabel('$u(x,t)$', fontsize=36)
    ax.set_xticks([0, L])
    ax.set_xticklabels(['$0$', '$L$'], fontsize=32)
    ax.tick_params(labelsize=32)
    ax.minorticks_off()
    fig.text(0.13, 0.92, label, fontsize=28, color='white', alpha=0.6)
    time_text = ax.set_title('$t = 0.00$', fontsize=36)

    def update(i):
        line.set_ydata(U_grid[i])
        time_text.set_text(f'$t = {t_vals[i]:.2f}$')

    anim = animation.FuncAnimation(fig, update, frames=n_t, interval=50)
    anim.save(filename, writer=animation.FFMpegFileWriter(fps=20))
    plt.close(fig)
    print(f"Saved {filename}")

# --- 8.4.12 ---
L  = 1
x  = np.linspace(0, L, n_x)
k  = np.arange(1, K + 1)[:, None, None]
xg = x[None, None, :]
tg = t_vals[None, :, None]

U = (2/L) * np.sum(
    (-1)**(k+1) * np.sin((2*k-1)*np.pi*xg/L) * np.cos((2*k-1)*np.pi*c*tg/L),
    axis=0
)
make_mp4(x, U, 'chap8sec8.4ex12.mp4', L, 'Ex 8.4.12')

# --- 8.4.13 ---
U = (8/np.pi**2) * np.sum(
    (-1)**k / (2*k-1)**2 * np.sin((2*k-1)*np.pi*xg/L) * np.cos((2*k-1)*np.pi*c*tg/L),
    axis=0
)
make_mp4(x, U, 'chap8sec8.4ex13.mp4', L, 'Ex 8.4.13')

# --- 8.4.14 ---
U = (2/np.pi) * np.sum(
    (1/k) * (1 - np.cos(k*np.pi/2)) * np.sin(k*np.pi*xg/L) * np.cos(k*np.pi*c*tg/L),
    axis=0
)
make_mp4(x, U, 'chap8sec8.4ex14.mp4', L, 'Ex 8.4.14')

# --- 8.4.16 ---
L  = 2
x  = np.linspace(0, L, n_x)
k  = np.arange(0, K)[:, None, None]
xg = x[None, None, :]

U = np.sum(
    np.sin(np.pi*(2*k+1)/(2*L)) * np.sin(np.pi*(2*k+1)*xg/(2*L)) * np.cos(np.pi*(2*k+1)*c*tg/(2*L)),
    axis=0
)
make_mp4(x, U, 'chap8sec8.4ex16.mp4', L, 'Ex 8.4.16')
