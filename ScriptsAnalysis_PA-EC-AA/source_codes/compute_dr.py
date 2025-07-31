### This script computes the rotational diffusion coefficients of a crowded protein trajectory: whole trajectory, per chain

### Inputs: rotational autocorrelation function of each chain saved in separate temperature files
### Output: DR diffusion coefficients

import os
import numpy as np 
import matplotlib.pyplot as plt
from lmfit import Model


#### USER modifications ####################################

path            = "../../../02a-Simulations/Rotational/Rot_Diff_unitvectors/Psychro/System3/" # Path with rotational autocorrelation functions
path_indeces    = "../../../02a-Simulations/Systems/Psychro/Subbox3/chain_indeces.txt"

############################################################

chains          = count_lines(path_indeces)

def exp_decay(x, tau):
    return np.exp(-x/tau)
    
def count_lines(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            return len(lines)
    except FileNotFoundError:
        print("File not found.")
        return -1

Dr = [] 
path_parts = path.split("/")
org = path_parts[-3]
sys_num = path_parts[-2].split("System")[-1]
sys = f"System{sys_num}"

# Loop over Temperatures of a sytem
for folder in os.listdir(path):
    
    # Current Temperature & Files
    T = int(folder[:-1])
    folder_path = path + folder
    
    Dp = []
    for p in range(chains):
        
        # Load data from the xyz file using np.loadtxt
        data = np.loadtxt(folder_path + 'rotacf-P2_%s.xvg'%p, comments=('@', '#', '&'))

        # Extract x and y coordinates from the loaded data
        x_coordinates = data[:, 0]/1000 # ps to ns
        y_coordinates = data[:, 1]

        # Define the model using lmfit
        model = Model(exp_decay)

        # Set initial parameter values for the fit
        initial_tau = 10

        # Perform fit
        st = np.abs(x_coordinates - 0.3).argmin()
        en = np.abs(x_coordinates - 5).argmin()
        result = model.fit(y_coordinates[st:en], x=x_coordinates[st:en], tau=initial_tau)
        
        final_tau = result.best_values['tau']
        print("Final tau:", final_tau)

        fig, ax = plt.subplots(1,1,figsize = (3,3))
        ax.plot(x_coordinates[0:],y_coordinates[0:])
        ax.set_xlabel('Time [ns]')
        ax.plot(x_coordinates[st:en], result.best_fit, label=r'$\tau = $%s'%np.round(final_tau, 5))
        ax.legend()
        plt.show()

        Dr_curr = 1/(6*final_tau)
        Dp.append(Dr_curr)
    Dr.append(Dp)

Dr = np.array(Dr)
print("Final Dr:", Dr)
