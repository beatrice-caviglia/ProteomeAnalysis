# Adapted from code originally shared by Stepan Timr (with permission)

# Load modules 
import numpy as np 
import MDAnalysis as mda
from MDAnalysis import *
from MDAnalysis import analysis
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis import align
import os
import glob
from helper_functions import * 

#### USER modifications ####################################

path          = "../../../Data.Availability/Upload/TRJ_Psychro_sys4/System4/Folded/" # path with temperature folders containing trajectories
indeces_path  = "../../../Data.Availability/Upload/TRJ_Psychro_sys4/System4/Files/"  # path with indeces file
trjfile       = "traj_prot.xtc" # filename of trajectory
grofile       = 'start_prot.gro'     # filename of GRO
output_folder = 'unitvectors/'        # output folder (already created)

############################################################


# Extract organism and system from path
path_parts = path.split("/")
org = path_parts[-3]
sys_num = path_parts[-2].split("Subbox")[-1]
sys = f"System{sys_num}"

protein_resids, protein_names = read_residue_ranges(indeces_path)
protein_names = list(rename_duplicates(protein_names))

# Loop over Temperatures of a sytem
for folder in os.listdir(path):
    
    T = int(folder[:-1])
    folder_path = path + folder + '/'
    trj =  folder_path + trjfile
    start = folder_path + grofile
    tpr = glob.glob(os.path.join(folder_path, '*.tpr'))[0].replace('\\', '/')

    # Universe with gro and xtc file
    u = mda.Universe(start, trj)
    protein_atoms = u.select_atoms('protein')

    # Define a Universe for initial frame (actually end frame)
    init = mda.Universe(start)
    protein_atoms_init = init.select_atoms('protein')

    #Generate random positions on a a unit sphere
    npoints = 1000
    z = 2*np.random.random(size=npoints) - 1
    phi = 2*np.pi*np.random.random(size=npoints)
    theta = np.arcsin(z)
    x = np.cos(theta)*np.cos(phi)
    y = np.cos(theta)*np.sin(phi)
    x = np.concatenate(([0], x))
    y = np.concatenate(([0], y))
    z = np.concatenate(([0], z))
    points = np.array([x,y,z])

    # Compute the rotational matrix and apply to unit vectors
    chain_points = np.zeros((u.trajectory.n_frames, len(protein_names), npoints+1, 3))

    # Loop over trajectory
    for (i,ts) in enumerate(u.trajectory):

        # Loop over proteins
        for (j,ids) in enumerate(protein_resids[:]):

            # Select protein pj
            pj = protein_atoms[ids[0]:ids[1]]
            pj_ref = protein_atoms_init[ids[0]:ids[1]]
            pj_ref = pj_ref.atoms

            REF = pj_ref
            CURRENT = pj

            ref0 = REF.positions - REF.center_of_mass()
            cur0 = CURRENT.positions - CURRENT.center_of_mass()
            R, rmsd = align.rotation_matrix(ref0, cur0)
            chain_points[i,j,:,:] = np.dot(R,points).T
    

    # Loop over the proteins and write 
    for (j, pnames) in enumerate(protein_names):
        # Write PDB reference and an XTC with points trajectories
        u_points = mda.Universe.empty(npoints+1, trajectory=True)
        u_points.load_new(chain_points[:,j,:,:],
                          format=mda.coordinates.memory.MemoryReader, dt=u.trajectory.dt)
        with mda.Writer(output_folder + "points_{}.xtc".format(j), len(u_points.atoms)) as W:
            for ts in u_points.trajectory:
                W.write(u_points)
    u_points.atoms.write(output_folder + 'reference_points.pdb')

    #Create index file
    with open(output_folder + 'index_points.ndx', 'w') as f:
        f.write("[ points ]\n")
        for i in range(npoints):
            f.write("1 {}\n".format(i+2))

# To create rotational autocorrelation functions of the unitvectors in GROMACS:
# gmx_mpi grompp -f grompp.mdp -c ../unitvectors/reference_points.pdb -maxwarn 2
# gmx_mpi rotacf -P 2 -f unitvectors/points_"${i}".xtc -s fake_tpr/topol.tpr -n unitvectors/index_points -o rotacf/rotacf-P2_"${i}" -d 

