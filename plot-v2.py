import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess


repertoire = "./"                     # dossier contenant l'exécutable
executable = "physnum6"            # nom de l'exécutable
input_filename = "configuration.in"  # fichier d'entrée
output_base = "output"    # un autre truc qu'il faut apparament

# Définir les paramètres

fs=12
ls=12

tfin = 0.08
xL = -1
xR = 1
V0 = 1350
xa = -0.5
xb = 0.5
om0 = 100
x0 = -0.5
sigma_norm = 0.04
n = 16
#Numérique
Nsteps = 800
Nintervals = 512

# Construire la commande
cmd = f"{repertoire}{executable} {input_filename} tfin={tfin} xL={xL} xR={xR} V0={V0} xa={xa} xb={xb} om0={om0} x0={x0} n={n} sigma_norm={sigma_norm} Nsteps={Nsteps} Nintervals={Nintervals} output={output_base}"

subprocess.run(cmd, shell=True)



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
line_abs, = ax.plot(x, abs_values[0], color='black', label="|ψ(x,t)|")
line_re,  = ax.plot(x, real_values[0], color='blue', label="Re(ψ(x,t))")
line_im,  = ax.plot(x, im_values[0], color='red', label="Im(ψ(x,t))")
#line_pot, = ax.plot(x, V / np.max(V) * np.max(psi_values), '--', label="Potentiel V(x)")

ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("|ψ(x,t)|", fontsize=14)
ax.set_title("Évolution de |ψ(x,t)|")
ax.legend()
ax.grid(True)

def update(frame):
    line_abs.set_ydata(abs_values[frame])
    line_re.set_ydata(real_values[frame])
    line_im.set_ydata(im_values[frame])
    ax.set_title(f"|ψ(x,t)| — frame {frame+1}/{nt}")
    return line_abs,line_re,line_im

ani = animation.FuncAnimation(fig, update, frames=nt, interval=12)
plt.tight_layout()

#quantitées physiques :

x_exp=obs[:,4]
p_exp=obs[:,6]
times=obs[:,0]
Energie=obs[:,3]
p_gauche=obs[:,1]
p_droite = obs[:,2]


# solution théorique :


# tracé des graphiques

#tracé des trajectoires

plt.figure()
X, T = np.meshgrid(x, times) 
plt.pcolormesh(X, T, abs_values, shading='auto', cmap='viridis')  
plt.colorbar(label="|ψ(x,t)|")
plt.xlabel("Position x")
plt.ylabel("Temps t")

plt.figure()
plt.pcolormesh(X, T, real_values, shading='auto', cmap='viridis')  
plt.colorbar(label="Re(ψ(x,t))")
plt.xlabel("Position x")
plt.ylabel("Temps t")



# tracé de <x> 
plt.figure()
plt.scatter(times,x_exp,label="<x>_quantique (t)")

plt.xlabel("temps [s]")
plt.ylabel("<x>")
plt.grid(True)
plt.legend(fontsize=ls)



# tracé de <p> 
plt.figure()
plt.scatter(times,p_exp,label="<p>_quantique (t)")
plt.xlabel("temps [s]")
plt.ylabel("<x>")
plt.grid(True)
plt.legend(fontsize=ls)

#tracé de V

plt.figure()
plt.plot(x, V, label="Potentiel V(x)", color='black')
plt.xlabel("x")
plt.ylabel("V(x)")
plt.grid(True)
plt.legend(fontsize=ls)

# tracé de la propbabilité a gauche et a droite .
rapport_V0_E = V0 / Energie[0]

plt.figure()
plt.plot(times, p_gauche, label=f"P_gauche (V₀/⟨E⟩ ≈ {rapport_V0_E:.2f})")
plt.plot(times, p_droite, label="P_droite")
plt.xlabel("temps [s]")
plt.ylabel("Probabilité")
plt.grid(True)
plt.legend(fontsize=ls)
plt.show()



plt.show()

