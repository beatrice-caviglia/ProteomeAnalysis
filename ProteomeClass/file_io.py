import os
import MDAnalysis as mda
import re
import numpy as np

def separate_proteins(u, indeces_path):
    """
    Parses a file of residue ranges to extract individual protein atom groups from a universe.

    Args:
        u (MDAnalysis.Universe): Universe containing the system's atoms.
        indeces_path (str): Path to a file specifying protein names, residue index ranges,
                            and optionally protein masses.

    Returns:
        tuple:
            - proteins (list of AtomGroup): Atom groups for each protein.
            - protein_names (list of str): Unique names assigned to each protein.
            - protein_resids (list of arrays): Residue indices for each protein.
            - protein_masses (list of float or None, optional): Masses if provided; otherwise omitted.
    """
    filename = indeces_path
    atomgroups = {}
    protein_masses = []  # List to store protein masses if they exist

    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) < 3:  # Skip lines that don't have at least 3 columns
                continue
            
            protein_name = parts[0]
            start_residue = int(parts[1])
            end_residue = int(parts[2])

            # Check if there's a third column (protein mass)
            if len(parts) > 3:
                protein_mass_str = parts[3].replace(',', '')  # Remove commas from the mass string
                protein_mass = float(protein_mass_str)  # Convert the cleaned string to a float
                protein_masses.append(protein_mass)
            else:
                protein_masses.append(None)  # If no mass is given, append None

            new_protein_name = protein_name
            count = 1
            while new_protein_name in atomgroups:
                new_protein_name = f"{protein_name}_{count}"
                count += 1
            atomgroups[new_protein_name] = f"index {start_residue}-{end_residue}"

    # Create atom groups and extract names and residue indices
    proteins = [u.select_atoms(selection_string) for selection_string in atomgroups.values()]
    protein_names = list(atomgroups.keys())
    protein_resids = [protein.resids for protein in proteins]

    # Return protein masses only if they are populated and valid
    if protein_masses and any(mass is not None for mass in protein_masses):
        return proteins, protein_names, protein_resids, protein_masses
    else:
        return proteins, protein_names, protein_resids


def write_com_to_xtc(com_universe, outputdir, base, filename, n_proteins):
    """
    Write the center-of-mass (COM) trajectory to .xtc and .gro files.

    Args:
        com_universe (MDAnalysis.Universe): Universe containing the COM trajectory.
        outputdir (str): Directory where output files will be saved.
        base (str): Base filename prefix.
        filename (str): Name of the output file (without extension).
        n_proteins (int): Number of proteins in the system.
    """
    com_dir = os.path.join(outputdir, 'Center_of_Masses')
    os.makedirs(com_dir, exist_ok=True)
    base_filename = os.path.join(com_dir, base + filename)

    # Write .xtc trajectory file
    with mda.Writer(base_filename + '.xtc', n_proteins) as W:
        for ts in com_universe.trajectory:
            W.write(com_universe)

    # Write .gro coordinate file
    with mda.Writer(base_filename + '.gro', n_proteins) as W:
        W.write(com_universe.atoms)

    print(f'Proteome Center of Mass (COM) trajectory saved in {base_filename}.gro and {base_filename}.xtc')



def write_beta_pdb(filename, output, betas):
    """
    Write a PDB file with updated beta factors.

    Args:
        filename (str): Path to the input PDB file.
        output (str): Path to the output PDB file.
        betas (numpy.ndarray): Beta factor values to insert into the PDB file.

    """
    fr = -1  # Frame counter
    c = 0    # Counter for atoms within each frame

    with open(filename, 'r') as infile:
        with open(output, 'w') as outfile:
            for line in infile:
                line_split = line.split()

                if len(line_split) > 1 and 'CRYST1' not in line_split and 'MODEL' not in line_split and 'TITLE' not in line_split:
                    words = line.rsplit(' ', 1)
                    words[-2] = words[-2][0:-17] + "{:.2f}".format(betas[fr, c] / 10)  # Update beta factor
                    new_line = ' '.join(words)
                    outfile.write(new_line)
                    c += 1

                elif 'CRYST1' in line_split:
                    line_cryst = line  # Store CRYST1 line for later use

                elif "ENDMDL" in line_split:
                    outfile.write("END\n")

                else:
                    if 'MODEL' in line_split:
                        outfile.write(line)
                        outfile.write("HEADER\n")
                        outfile.write(line_cryst)  # Add CRYST1 line at the beginning of each model
                        print('New frame')
                        fr += 1
                        c = 0


def save_contacts(matrix_n_contacts, list_n_contacts, betas_array, outputdir, outputfile, base, create_trajectory, input_pdb, show_wrap, universe_wrap, skip, selection): # works
    """
    Save computed contacts and optionally create a trajectory file with beta factors.

    Args:
        matrix_n_contacts (np.ndarray): Matrix of average contacts between proteins.
        list_n_contacts (np.ndarray): Time-resolved contact matrix.
        betas_array (np.ndarray): Array for beta factors.
        outputdir (str): Directory to save the files.
        base (str): Base filename prefix.
        create_trajectory (bool, optional): Whether to save a trajectory file.
        input_pdb (str, optional): Input PDB file for writing output.
        show_wrap (bool, optional): Whether to use wrapped coordinates for trajectory.
        universe_wrap (MDAnalysis.Universe, optional): Universe containing wrapped proteins.
        skip (int, optional): Step size for trajectory frames.
    """

    cont_dir = os.path.join(outputdir, 'Contacts')
    os.makedirs(cont_dir, exist_ok=True)

    np.save(os.path.join(cont_dir, "Cntcts_time_"+base+".npy"), list_n_contacts)
    np.save(os.path.join(cont_dir, "Cntcts_avg_"+base+".npy"), matrix_n_contacts)

    if create_trajectory and input_pdb and universe_wrap:
        base_filename = os.path.join(cont_dir, base + outputfile + ".pdb")
        writer = mda.coordinates.PDB.PDBWriter(base_filename, multiframe=True)

        for i, ts in enumerate(universe_wrap.trajectory[::skip]):
            writer.write(universe_wrap.select_atoms(selection) if show_wrap else universe_wrap.select_atoms(selection))

        # Write beta factors
        base_filename2 = os.path.join(cont_dir, base + outputfile + "_BFactors.pdb")
        write_beta_pdb(base_filename, base_filename2, betas=betas_array)

        print(f"PDB written at {base_filename2}")

    print("Contacts computed and saved!")

def save_clusters(nclust_t, betas_array, outputdir, input_file_pdb, base):
    """
    Saves cluster trajectory data and writes a PDB file with beta factors.

    Args:
        nclust_t (array-like): Number of clusters over time (trajectory data).
        betas_array (array-like): Beta factors to embed in the PDB file (e.g. cluster metrics per atom).
        outputdir (str): Base directory where output files will be saved.
        input_file_pdb (str): Base name of the input PDB file (without extension).
        base (str): String used to construct output file names.

    Returns:
        None. Saves a `.npy` file with cluster counts and a `.pdb` file with beta values.
    """
    cont_dir = os.path.join(outputdir, 'Contacts/')
    input_file = cont_dir + input_file_pdb
    output_file = cont_dir + 'Clusters_Trajectory'
    os.makedirs(cont_dir, exist_ok=True)
    write_beta_pdb(input_file + '.pdb', output_file +'_BFactors.pdb', betas = np.array(betas_array))
    cont_dir = os.path.join(outputdir, 'Contacts')
    base_filename = os.path.join(cont_dir, "N_Clusters_time_"+base+".npy")
    np.save(base_filename, np.array(nclust_t))


def save_mindist(mind, outputdir, base):
    """
    Saves minimum distance data over time as a NumPy array.

    Args:
        mind (array-like): Minimum distance values over time (e.g. per frame).
        outputdir (str): Directory where the file will be saved.
        base (str): Base name to include in the output filename.

    Returns:
        None. Saves a `.npy` file named 'Mindist_time_<base>.npy' in the 'Contacts' subdirectory.
    """
    cont_dir = os.path.join(outputdir, 'Contacts')
    os.makedirs(cont_dir, exist_ok=True)
    base_filename = os.path.join(cont_dir, "Mindist_time_"+base+".npy")
    np.save(base_filename, np.array(mind))





