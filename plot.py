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

m=1
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
def cc(en):
    return en * np.max(abs_values) *2 / np.max(V)
ax.plot(x, cc(V), "k--", label="V(x)")
ax.plot(x, np.ones_like(x)*cc(np.max(obs[:,3])), 'y--')
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
prob_total = obs[:,1] + obs[:,2]
#probleme
delta_x = np.sqrt(np.abs(obs[:,5] - obs[:,4]**2))
delta_p = np.sqrt(np.abs(obs[:,7] - obs[:,6]**2))
#probleme
incertitude = delta_x * delta_p


# solution théorique :

om0_pr = om0/np.abs(xL)
p0 = p_exp[0]
x_class=x0 * np.cos(om0_pr * times) + p0/(om0*m) * np.sin(om0_pr*times)
p_class=-m*x0*om0_pr*np.sin(om0_pr * times) + p0*np.cos(om0_pr*times)



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
plt.scatter(times,x_exp,label="<x>_quantique (t)", s=3)
plt.scatter(times,x_class,label="<x>_classique (t)", s=3)
plt.xlabel("temps [s]")
plt.ylabel(r"$\left<x\right>$")
plt.grid(True)
plt.legend(fontsize=ls)



# tracé de <p> 
plt.figure()
plt.scatter(times,p_exp,label=r"$\left<p\right>_{quantique} (t)$",s=3)
plt.scatter(times,p_class,label=r"$\left<p\right>_{classique} (t)$",s=3)
plt.xlabel("temps [s]")
plt.ylabel(r"$\left<p\right>$")
plt.grid(True)
plt.legend(fontsize=ls)

#tracé de V

plt.figure()
plt.plot(x, V, label="Potentiel V(x)", color='black')
plt.xlabel("x")
plt.ylabel("V(x)")
plt.grid(True)
plt.legend(fontsize=ls)


#conservation de la probabilité


plt.figure()
plt.plot(times, prob_total, label="Probabilité totale")
plt.axhline(1, color='gray', linestyle='--', label="probabilité =1")
plt.xlabel("temps [s]")
plt.ylabel("∫|ψ|² dx")
plt.grid(True)
plt.legend(fontsize=ls)

#conservation de l'energie

plt.figure()
plt.plot(times, obs[:,3], label="Energie dans le temps")
#plt.axhline(Energie[0], color='gray', linestyle='--', label="Energie initiale")
plt.xlabel("temps [s]")
plt.ylabel("Energie [j]")
plt.grid(True)
plt.legend(fontsize=ls)

# tracé du principe de Heisenberg


hbar = 1.0
plt.figure()
plt.plot(times, incertitude, label="Δx · Δp")
plt.plot(times, delta_x , label="Δx")
plt.plot(times, delta_p, label="Δp")
plt.axhline(hbar/2, color='gray', linestyle='--', label="ℏ/2 = 0.5")
plt.xlabel("Temps [s]")
plt.ylabel("Δx · Δp")
plt.title("Principe d’incertitude de Heisenberg")
plt.legend(fontsize=fs)
plt.grid(True)



plt.show()

