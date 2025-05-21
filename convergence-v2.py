import numpy as np
import matplotlib.pyplot as plt
import subprocess

# === Paramètres fixes ===
repertoire = "./"
executable = "physnum6"
input_filename = "configuration.in"
output_base = "output"

# Paramètres physiques
tfin = 0.08
xL = -1
xR = 1
xa = -0.5
xb = 0.5
om0 = 100
x0 = -0.5
sigma_norm = 0.04
n = 16

# Paramètres numériques
Nsteps = 800
Nintervals = 512
t_trans = 0.035

# === Étude en fonction de V0 ===
V0_values = np.linspace(100, 1000, 20).astype(int)

P_trans = []
E_over_V0 = []

for V0 in V0_values:
    cmd = f"{repertoire}{executable} {input_filename} tfin={tfin} xL={xL} xR={xR} V0={V0} xa={xa} xb={xb} om0={om0} x0={x0} n={n} sigma_norm={sigma_norm} Nsteps={Nsteps} Nintervals={Nintervals} output={output_base}"
    subprocess.run(cmd, shell=True)

    obs = np.fromfile(f"{output_base}_obs.out", dtype=np.float64).reshape(-1, 8)
    times = obs[:, 0]
    idx_trans = np.argmin(np.abs(times - t_trans))

    E = obs[idx_trans, 3]  # colonne énergie
    P_right = obs[idx_trans, 2]  # colonne proba droite (x > 0)

    E_over_V0.append(E / V0)
    P_trans.append(P_right)

# === Tracé
plt.figure()
plt.plot(E_over_V0, P_trans, 'o-')
plt.xlabel("⟨E⟩ / V₀")
plt.ylabel("P_trans (x > 0 à t ≈ 0.035)")
plt.grid(True)
plt.tight_layout()
plt.show()


