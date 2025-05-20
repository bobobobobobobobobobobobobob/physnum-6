import numpy as np
import matplotlib.pyplot as plt
import subprocess

repertoire = "./"
executable = "physnum6"
input_filename = "configuration.in"
output_base = "output"

# Paramètres physiques
tfin = 0.08
xL = -1
xR = 1
V0 = 0
xa = 0
xb = 0
om0 = 100
x0 = -0.5
sigma_norm = 0.04
n = 16

# === CHOIX : paramètre à faire varier ===
param_name = "Nsteps"  # ou "Nintervals"
param_values = np.linspace(300,1000,20).astype(int)

# Paramètres fixes associés
fixed_Nsteps = 800
fixed_Nintervals = 512

x_final = []
dt_list = []

for val in param_values:
    Nsteps = val if param_name == "Nsteps" else fixed_Nsteps
    Nintervals = val if param_name == "Nintervals" else fixed_Nintervals

    cmd = f"{repertoire}{executable} {input_filename} tfin={tfin} xL={xL} xR={xR} V0={V0} xa={xa} xb={xb} om0={om0} x0={x0} n={n} sigma_norm={sigma_norm} Nsteps={Nsteps} Nintervals={Nintervals} output={output_base}"
    subprocess.run(cmd, shell=True)

    obs = np.fromfile(f"{output_base}_obs.out", dtype=np.float64).reshape(-1, 8)
    x_final.append(obs[-1, 4])

    if param_name == "Nsteps":
        dt_list.append(tfin / Nsteps)   #  dt pas de temps
    else:
        dx = (xR - xL) / Nintervals     # element de maillage h(x)
        dt_list.append(dx)

# === Tracé
plt.figure()
plt.plot(np.array(dt_list)**2, x_final, '+-')
plt.xlabel("∆t" if param_name == "Nsteps" else "h (dx)")
plt.ylabel("⟨x⟩(tfin)")
plt.grid(True)
plt.show()


