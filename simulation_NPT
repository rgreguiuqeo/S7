import hoomd
import gsd.hoomd
import numpy as np
from tqdm import tqdm
import argparse
import os

from ini_conf import ini_conf
from utils import tabulation_pot_dpd, MolecularMotors, setup_potentials, setup_writers

###################
#SIMULATION INPUTS#
###################

parser = argparse.ArgumentParser()

parser.add_argument('--dim',        type=int, choices=[2,3], default=3)
parser.add_argument('--density',    type=float, default=1e-2)

parser.add_argument('--N_polymers', type=int, default=1000)
parser.add_argument('--N_monomers', type=int, default=10)

parser.add_argument('--kT',         type=float, default=1)
parser.add_argument('--gamma',      type=float, default=1)
parser.add_argument('--Fact',       type=float, default=0)
parser.add_argument('--kappa_bond', type=float, default=5000)
parser.add_argument('--kappa_bend', type=float, default=0.001)
parser.add_argument('--sigma',      type=float, default=1)
parser.add_argument('--epsilon',    type=float, default=1)
parser.add_argument('--r0',         type=float, default=1)
parser.add_argument('--mass',       type=float, default=1e-2)
parser.add_argument('--A',          type=float, default=100)

parser.add_argument('--dt',         type=str, default='1e-3')
parser.add_argument('--n_steps',    type=float, default=100000)
parser.add_argument('--dump_freq',  type=float, default=10)

parser.add_argument('--sim_mode',   type=str, default='soft')
parser.add_argument('--sim_status', type=str, default='equilibration')

arguments           = parser.parse_args()
params              = vars(arguments)

dim                 = arguments.dim
density             = arguments.density

N_polymers          = arguments.N_polymers
N_monomers          = arguments.N_monomers

kT                  = arguments.kT
gamma               = arguments.gamma
Fact                = arguments.Fact
kappa_bond          = arguments.kappa_bond
kappa_bend          = arguments.kappa_bend
sigma               = arguments.sigma
epsilon             = arguments.epsilon
r0                  = arguments.r0
mass                = arguments.mass
A                   = arguments.A

dt                  = float(arguments.dt)
n_steps             = float(arguments.n_steps)
dump_freq           = arguments.dump_freq

box_size            = (N_polymers*N_monomers/density)**(1/3)
params['box_size']  = box_size

sim_mode            = arguments.sim_mode
sim_status          = arguments.sim_status

specified_params = {
    action.dest: vars(arguments).get(action.dest) != action.default
    for action in parser._actions
    if action.dest is not None 
}
##################
# INITIALIZATION #
##################

dev        = hoomd.device.CPU()
simulation = hoomd.Simulation(device=dev, seed=1)

if (sim_mode == 'soft'):

    frame = ini_conf(N_polymers, N_monomers, mass, r0, box_size, kappa_bend)  
    simulation.create_state_from_snapshot(frame)
    
elif (sim_mode == 'hard'):
    
    if (sim_status == 'equilibration'):

        simulation.create_state_from_gsd(filename='S7/traj_eq.gsd', frame=-1)

    elif (sim_status == 'production'):

        if os.path.exists('traj_prod.gsd'):
            simulation.create_state_from_gsd(filename='S7/traj_prod.gsd', frame=-1)

        else:
            simulation.create_state_from_gsd(filename='S7/traj_eq.gsd', frame=-1)


##############
# INTEGRATOR #
##############

P_target = 1.0 * kT / sigma**3

npt = hoomd.md.methods.NPT(
        filter=hoomd.filter.All(),
        kT=kT,          # target temperature
        tau=1.0,        # thermostat relaxation time (≈ 1 τ)
        S=P_target,     # target pressure
        tauS=10.0,      # barostat relaxation time (≈ 10 τ)
        couple='xyz')   # isotropic volume moves; use 'none' or 'xy' etc. if needed


forces     = setup_potentials(sim_mode, params)
integrator = hoomd.md.Integrator(dt=dt,
                                 methods=[npt],
                                 forces=forces)
simulation.operations.integrator = integrator

####################
# DEFINING WRITERS #
####################

thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
simulation.operations.add(thermo)
logger = hoomd.logging.Logger()
logger.add(thermo, quantities=['kinetic_temperature', 'pressure', 'potential_energy'])

write_thermo, write_traj = setup_writers(sim_mode, sim_status, logger, dump_freq)
simulation.operations.writers.append(write_traj)
simulation.operations.add(write_thermo)

####################
#RUNNING SIMULATION#
####################

for _ in tqdm(range(100), desc='Running simulation:'):
    simulation.run(int(n_steps / 100))


