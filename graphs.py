import gsd.hoomd
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# user inputs
filename    = "/home/julianmonincx/Advanced Computational Condensed Matter/traj_eq.gsd"
N_chains    = 1000
N_monomers  = 10
dt          = 1e-3          # MD timestep
dim         = 3             # 2 or 3
kB          = 1.0           # Boltzmann constant in reduced units
# ------------------------------------------------------------------

t_list        = []
avg_e2e_list  = []
radius_list   = []
T_list        = []

with gsd.hoomd.open(name=filename, mode='r') as traj:
    for i, frame in enumerate(tqdm(traj, desc="Reading frames")):
        # --- timestamp
        t_list.append(i * dt)

        # --- coordinates & velocities
        positions  = frame.particles.position
        velocities = frame.particles.velocity
        if velocities is None:
            raise RuntimeError("Velocities missing from trajectory; cannot compute temperature.")

        masses     = getattr(frame.particles, "mass", np.ones(len(positions)))

        # --- ⟨end-to-end⟩
        e2e = []
        for c in range(N_chains):
            start = c * N_monomers
            end   = start + N_monomers - 1
            e2e.append(np.linalg.norm(positions[end] - positions[start]))
        avg_e2e_list.append(np.mean(e2e))

        # --- max radial extent
        com         = positions.mean(axis=0)
        max_radius  = np.max(np.linalg.norm(positions - com, axis=1))
        radius_list.append(max_radius)

        # --- instantaneous temperature
        KE   = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
        dof  = dim * len(positions)          # translational DOF
        T_list.append((2.0 * KE) / (dof * kB))

# ------------------------------------------------------------------
# Plot ⟨E2E⟩ vs time
plt.figure()
plt.plot(t_list, avg_e2e_list)
plt.xlabel("Time (simulation units)")
plt.ylabel("Average end-to-end distance")
plt.title("Polymer ⟨E2E⟩ vs Time")
plt.tight_layout()
plt.savefig("e2e_vs_time.png")
plt.close()

# Plot max radius vs time
plt.figure()
plt.plot(t_list, radius_list)
plt.xlabel("Time (simulation units)")
plt.ylabel("Max radius from CoM")
plt.title("Polymer extent vs Time")
plt.tight_layout()
plt.savefig("polymer_radius_vs_time.png")
plt.close()

# Plot temperature vs time
plt.figure()
plt.plot(t_list, T_list)
plt.xlabel("Time (simulation units)")
plt.ylabel("Instantaneous temperature")
plt.title("Temperature vs Time")
plt.tight_layout()
plt.savefig("temperature_vs_time.png")
plt.close()
