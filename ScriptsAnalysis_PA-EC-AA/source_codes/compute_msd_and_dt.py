### This script computes the translational diffusion coefficients of a crowded protein trajectory: whole trajectory, per chain

### Inputs: TPR file, XTC files, protein_indeces.txt files, and Proteome Class code
### Output: DT diffusion coefficients

import sys
import os
sys.path.append(os.path.abspath('../../ProteomeClass/'))
from proteome import *
import re

#### USER modifications ####################################

directory    = '../../Data.Availability/Upload/TRJ_Psychro_sys4/System4/Folded/'                 
indeces_file = '../../Data.Availability/Upload/TRJ_Psychro_sys4/System4/Files/chain_indeces.txt' 
TPR_file     = directory + '300K/sys.tpr'
XTC_file     = '/traj_prot.xtc'

output_dir   = 'Results'

############################################################


pattern      = re.compile(r'^\d+K$')
temp_dirs    = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and pattern.match(name)]


prot1 = Proteome(tpr_file = TPR_file, 
                 indices_path = indeces_file, 
                 outputdir=output_dir, 
                 base='P_Arcticus',
                 plot=True,
                 selection = "protein"
                )
DT = []
for T_dir in temp_dirs:
    prot1.add_trajectory(directory + T_dir + XTC_file)
    prot1.center_of_masses(append=False)
    prot1.msd_com(n_blocks=1, HalfDt=True)
    Dt = prot1.dt_com()
    DT.append(Dt)

DT = np.array(DT)
