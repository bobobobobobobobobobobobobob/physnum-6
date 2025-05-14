import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === Fichiers ===
prefix = "output"
psi_file = f"{prefix}_psi2.out"
pot_file = f"{prefix}_pot.out"
x_file = f"{prefix}_x.out"
obs_file = f"{prefix}_obs.out"

# === Chargement du maillage et du potentiel ===
V = np.fromfile(pot_file, np.float64)
x = np.fromfile(x_file, np.float64)

obs = np.fromfile(obs_file, np.float64).reshape((-1, 8))

# === Lecture de |ψ(x,t)| depuis le fichier psi2 ===
psi2_raw = np.fromfile(psi_file, np.float64).reshape((obs.shape[0],-1))
abs_values = psi2_raw[:,0::3]
real_values = psi2_raw[:,1::3]
im_values = psi2_raw[:,2::3]

nt, nx = abs_values.shape

# === Animation ===
fig, ax = plt.subplots()
line_psi, = ax.plot(x, abs_values[0], label="|ψ(x,t)|")
#line_pot, = ax.plot(x, V / np.max(V) * np.max(psi_values), '--', label="Potentiel V(x)")

ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("|ψ(x,t)|", fontsize=14)
ax.set_title("Évolution de |ψ(x,t)|")
ax.legend()
ax.grid(True)

def update(frame):
    line_psi.set_ydata(abs_values[frame])
    ax.set_title(f"|ψ(x,t)| — frame {frame+1}/{nt}")
    return line_psi

ani = animation.FuncAnimation(fig, update, frames=nt, interval=12)
plt.tight_layout()
#ani.save("woo.mp4")
plt.show()

