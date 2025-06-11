import gsd.hoomd
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# Inputs
filename = "/home/julianmonincx/Advanced Computational Condensed Matter/traj_eq.gsd"
N_chains = 1000
N_monomers = 10

with gsd.hoomd.open(name=filename, mode='r') as traj:
    avg_e2e_list = []

    for frame in tqdm(traj):
        positions = frame.particles.position
        e2e_distances = []

        for i in range(N_chains):
            start_idx = i * N_monomers
            end_idx = start_idx + N_monomers - 1
            r0 = positions[start_idx]
            r1 = positions[end_idx]
            dist = np.linalg.norm(r1 - r0)
            e2e_distances.append(dist)

        avg_e2e = np.mean(e2e_distances)
        avg_e2e_list.append(avg_e2e)

# Plotting (optional)
plt.plot(avg_e2e_list)
plt.xlabel("Frame")
plt.ylabel("Average End-to-End Distance")
plt.title("Polymer Chain E2E vs Time")
plt.savefig("e2e_vs_time.png")
