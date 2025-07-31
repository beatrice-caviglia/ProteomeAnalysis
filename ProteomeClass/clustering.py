import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.analysis import distances
import gc
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from file_io import *
import os
from MDAnalysis.lib.nsgrid import FastNS
import matplotlib.pyplot as plt
import networkx as nx


def calculate_protein_diameters(protein_list, boxdim): # works
    """Calculates max pairwise distances (diameters) for each protein in the list considering periodic boundaries."""
    diameters = []
    for p in protein_list:
        positions = p.positions
        diameters.append(np.max(distances.distance_array(positions, positions, box=boxdim)))
    return diameters


def contacts_within_cutoff(args): # works
    """Finds pairs of atoms within cutoff radius for a given frame and returns contact counts and binary contact arrays."""
    frame_index, ts, group_a, group_b, radius,u = args
    u.trajectory[frame_index]
    ns = FastNS(radius, group_a.wrap(), box=ts.dimensions, pbc=True)
    neighbors = ns.search(group_b.wrap())
    pairs = neighbors.get_pairs()
    # remove betas from output, not interested in this! for fast evaluation!
    betas = np.zeros(group_a.n_atoms + group_b.n_atoms, dtype=int)
    betas_a = np.zeros(group_a.n_atoms, dtype=int)
    betas_b = np.zeros(group_b.n_atoms, dtype=int)
    #print('WHY',group_b.n_atoms, group_b.ids[0])
    for ab, aa in pairs:
        #print(ab)
        betas_b[ab] = 1
        betas_a[aa] = 1
    return frame_index, len(pairs), betas_a, betas_b



def compute_contacts(proteins_list_wrap, universe_wrap, processes, radius_contact, skip, selection): # works
    """
    Compute contact maps between proteins based on a distance cutoff.

    Args:
        proteins_list_wrap (list): List of wrapped protein atom groups.
        universe_wrap (MDAnalysis.Universe): Universe containing wrapped protein positions.
        radius_contact (float, optional): Distance cutoff for contacts (default: 8 Å).
        skip (int, optional): Step size for iterating over trajectory frames (default: 1).

    Returns:
        tuple: (matrix_n_contacts, list_n_contacts, betas_array)
    """

    n_proteins = len(proteins_list_wrap)
    n_frames = len(universe_wrap.trajectory[::skip])
    betas_array = np.zeros((n_frames, universe_wrap.select_atoms(selection).n_atoms))

    # Compute protein diameters
    boxdim = universe_wrap.dimensions
    protein_diameters = calculate_protein_diameters(proteins_list_wrap, boxdim)
    
    matrix_n_contacts = np.zeros((n_proteins, n_proteins))
    list_n_contacts = np.zeros((n_proteins, n_proteins, n_frames))
    
    for j, p0 in enumerate(proteins_list_wrap):
        p0 = p0.select_atoms('protein')
        radius_p0 = protein_diameters[j] / 2

        for k, p1 in enumerate(proteins_list_wrap[j + 1:], start=j + 1):
            print(f"Processing contacts between Protein {j} and Protein {k}")

            p1 = p1.select_atoms('protein')
            radius_p1 = protein_diameters[k] / 2
            com_distance = distances.distance_array(
                np.array([p0.center_of_mass()]),
                np.array([p1.center_of_mass()]),
                box=universe_wrap.dimensions
            )[0, 0]
            
            n_cont = 0
            if com_distance < (radius_p0 + radius_p1 + 30):
                res_dist = distances.distance_array(p0.positions, p1.positions, box=universe_wrap.dimensions)
                if np.min(res_dist) < radius_contact:
                    print(f"Minimum distance between residues COM: {np.min(res_dist)}")

                with Pool(processes=processes) as pool:
                    args = [(i, ts, p0, p1, radius_contact, universe_wrap) for i, ts in enumerate(universe_wrap.trajectory[::skip])]
                    results = pool.map(contacts_within_cutoff, args)

                betas_i = np.array([result[2] for result in results])
                betas_j = np.array([result[3] for result in results])
                n_cont = np.mean([result[1] for result in results])
                indeces = np.array([result[0] for result in results])
                values = np.array([result[1] for result in results])

                ind1, ind2 = p0.ids[0], p0.ids[-1] + 1
                ind3, ind4 = p1.ids[0], p1.ids[-1] + 1

                betas_array[:, ind1:ind2] += betas_i
                betas_array[:, ind3:ind4] += betas_j
                list_n_contacts[j, k, indeces] = values

            matrix_n_contacts[j, k] = n_cont
            gc.collect()

    return matrix_n_contacts, list_n_contacts, betas_array



def clusters_one_frame(proteins_list, contacts, Ncrit):
    """
    Compute the number of clusters and assign cluster values to atoms based on the contact matrix.

    Args:
        contacts (np.ndarray): Contact matrix of shape (n_proteins, n_proteins).
        proteins_list (list): List of protein atom groups.
        Ncrit (int): Threshold number of contacts for cluster formation.

    Returns:
        tuple: (num_clusters, cluster_sizes, beta_array)
    """

    #print(f"Contact matrix shape: {contacts.shape}")

    beta_array = np.zeros(sum(p.n_atoms for p in proteins_list))  # 1D array for cluster values

    # Symmetrize the contact matrix
    contacts_sym = np.maximum(contacts, contacts.T)

    # Identify contacts above threshold
    adjacency_matrix = contacts_sym > Ncrit
    #plt.imshow(adjacency_matrix, cmap="binary", interpolation="nearest")

    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)
    connected_components = list(nx.connected_components(G))

    # Determine the number and sizes of clusters
    num_clusters = len(connected_components)
    cluster_sizes = [len(component) for component in connected_components]

    # Assign cluster values to atoms based on connected components
    for cluster_idx, component in enumerate(connected_components):
        cluster_value = cluster_idx  # Assign a unique cluster index

        for protein_idx in component:
            start_idx = proteins_list[protein_idx].ids[0]
            end_idx = proteins_list[protein_idx].ids[-1]
            beta_array[start_idx:end_idx] = cluster_value

    #print(f"Clusters defined based on contacts: {num_clusters} clusters found.")
    
    return num_clusters, cluster_sizes, beta_array


def clusters(proteins_list, Contacts, NCrit, input_file_pdb, outputdir, base, create_trajectory, show_wrap, skip):
        nclust_t = []
        sizes_t = []
        betas_t = []
        for t in range(Contacts.shape[-1]):
            nclus, sizes, betas = clusters_one_frame(proteins_list, Contacts[:, :, t], NCrit)  # Assuming n_clusters is predefined
            nclust_t.append(nclus)
            sizes_t.append(sizes)  # Flatten the sizes directly
            betas_t.append(betas)

        sizes_t_flattened = [item for sublist in sizes_t for item in sublist]
        unique_sizes, counts = np.unique(sizes_t_flattened, return_counts=True)
        fractions = (unique_sizes * counts) / np.sum(unique_sizes * counts)
        betas_array = np.array(betas_t)/np.max(np.array(betas_t))
        #if create_trajectory:
        #    cont_dir = os.path.join(outputdir, 'Contacts/')
        #    input_file = cont_dir + input_file_pdb
        #    output_file = cont_dir + 'Clusters_Trajectory'
        #    os.makedirs(cont_dir, exist_ok=True)
        #    write_beta_pdb(input_file + '.pdb', output_file +'_BFactors.pdb', betas = np.array(betas_array))#*10)
        #cont_dir = os.path.join(outputdir, 'Contacts')
        # base_filename = os.path.join(cont_dir, base + "N_Clusters_time.npy")
        # np.save(base_filename, np.array(nclust_t))
        # base_filename = os.path.join(cont_dir, self.base + "S_Clusters_time.npy")
        # np.save(base_filename, np.array(sizes_t))
        return nclust_t, sizes_t, fractions, unique_sizes, betas_array


def mindistances(args):
    frame_index, ts, group_a, group_b, radius,u = args
    u.trajectory[frame_index]
    pairwise_distances = distances.distance_array(group_a.positions, group_b.positions, box=u.dimensions)
    min_distance = np.min(pairwise_distances)
    return frame_index, min_distance

def compute_mindist(proteins_list_wrap, universe_wrap, processes, radius_contact, skip, selection, verbose):
    """
    Compute contact maps between proteins based on a distance cutoff.

    Args:
        proteins_list_wrap (list): List of wrapped protein atom groups.
        universe_wrap (MDAnalysis.Universe): Universe containing wrapped protein positions.
        radius_contact (float, optional): Distance cutoff for contacts (default: 8 Å).
        skip (int, optional): Step size for iterating over trajectory frames (default: 1).

    Returns:
        tuple: (matrix_n_contacts, list_n_contacts, betas_array)
    """

    n_proteins = len(proteins_list_wrap)
    n_frames = len(universe_wrap.trajectory[::skip])
    betas_array = np.zeros((n_frames, universe_wrap.select_atoms(selection).n_atoms))

    # Compute protein diameters
    boxdim = universe_wrap.dimensions
    protein_diameters = calculate_protein_diameters(proteins_list_wrap, boxdim)
    
    matrix_n_contacts = np.zeros((n_proteins, n_proteins))
    list_n_contacts = np.zeros((n_proteins, n_proteins, n_frames))

    #print('Protein diameters:', protein_diameters)
    
    for j, p0 in enumerate(proteins_list_wrap):
        p0 = p0.select_atoms('protein')
        radius_p0 = protein_diameters[j] / 2

        for k, p1 in enumerate(proteins_list_wrap[j + 1:], start=j + 1):
            if verbose:
                print(f"Processing distances between Protein {j} and Protein {k}")

            p1 = p1.select_atoms('protein')
            radius_p1 = protein_diameters[k] / 2

            with Pool(processes=processes) as pool:
                args = [(i, ts, p0, p1, radius_contact, universe_wrap) for i, ts in enumerate(universe_wrap.trajectory[::skip])]
                results = pool.map(mindistances, args)

            frames = np.array([result[0] for result in results])
            mindist_ij = np.array([result[1] for result in results])
            list_n_contacts[j, k, frames] = mindist_ij
    return list_n_contacts
