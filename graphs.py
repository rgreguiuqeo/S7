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
dt = 1e-3


with gsd.hoomd.open(name=filename, mode='r') as traj:
    avg_e2e_list = []

    t_list = []
    for i, frame in enumerate(tqdm(traj)):
        t_list.append(i*dt)

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

    radius_list = []

    for frame in tqdm(traj):
        positions = frame.particles.position

        # Compute center of mass of all polymer beads (assuming all particles are polymers)
        com = positions.mean(axis=0)

        # Compute radial distances from center of mass
        radii = np.linalg.norm(positions - com, axis=1)

        # Max radius is the furthest monomer from the center
        max_radius = np.max(radii)

        # Or optionally, average radius instead of max:
        # avg_radius = np.mean(radii)

        radius_list.append(max_radius)

# Plotting (optional)
plt.plot(t_list, avg_e2e_list)
plt.xlabel("Frame")
plt.ylabel("Average End-to-End Distance")
plt.title("Polymer Chain E2E vs Time")
plt.savefig("e2e_vs_time.png")
plt.cla()

# Plot
plt.plot(t_list, radius_list)
plt.xlabel("Frame")
plt.ylabel("Max Radius from Center")
plt.title("Polymer Extent Over Time")
plt.savefig("polymer_radius_vs_time.png")
