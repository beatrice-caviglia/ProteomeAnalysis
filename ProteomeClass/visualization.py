import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.optimize import curve_fit

# === Global Plot Settings === (LaTeX required)
# tex_fonts = {
#     "text.usetex": True,
#     "font.family": "serif",
#     "axes.labelsize": 12,
#     "font.size": 12,
#     "legend.fontsize": 11,
#     "xtick.labelsize": 11,
#     "ytick.labelsize": 11
# }
# plt.rcParams.update(tex_fonts)
# plt.rcParams['axes.linewidth'] = 1.2
# plt.rcParams['lines.linewidth'] = 1.2
# mpl.rcParams['lines.markersize'] = 3
# plt.rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def plot_com_trajectories(results, outputdir, base, plot): # works
    """
    Plot center of mass (COM) trajectories for all proteins in (X,Y), (Y,Z), and (X,Z) views.

    Args:
        results (dict): Dictionary containing COM trajectories for each protein.
        outputdir (str): Directory to save the plot.
        base (str): Base filename prefix.
    """
    
    # Check if 'com' data exists in results
    if not any("com" in results for results in results.values()):
        print("No COM data found. Run center of mass computation first.")
        return
    
    # Extract COM data
    com_data = {protein: res["com"] for protein, res in results.items() if "com" in res}

    # Define colors for proteins
    colors = plt.cm.get_cmap("tab20b", len(com_data))

    # Create subplots for (X, Y), (Y, Z), (X, Z)
    fig, axes = plt.subplots(1, 3, figsize=(7.008, 3))

    for idx, (protein, com_values) in enumerate(com_data.items()):
        axes[0].plot(com_values[:, 0], com_values[:, 1], label=protein, alpha=0.7, color=colors(idx), linewidth=2)
        axes[1].plot(com_values[:, 1], com_values[:, 2], label=protein, alpha=0.7, color=colors(idx), linewidth=2)
        axes[2].plot(com_values[:, 0], com_values[:, 2], label=protein, alpha=0.7, color=colors(idx), linewidth=2)

    # Set labels
    axes[0].set_xlabel(r"X ($\mathrm{\AA}$)")
    axes[0].set_ylabel(r"Y ($\mathrm{\AA}$)")
    axes[1].set_xlabel(r"Y ($\mathrm{\AA}$)")
    axes[1].set_ylabel(r"Z ($\mathrm{\AA}$)")
    axes[2].set_xlabel(r"X ($\mathrm{\AA}$)")
    axes[2].set_ylabel(r"Z ($\mathrm{\AA}$)")


    plt.tight_layout()
    plt.show()

    # Save the plot
    if outputdir is not None:
        com_dir = os.path.join(outputdir, 'Center_of_Masses')
        os.makedirs(com_dir, exist_ok=True)
        base_filename = os.path.join(com_dir, base + "center_of_masses.png")
        fig.savefig(base_filename)
        print(f"Plot of COM saved in {base_filename}.")



def plot_msd(msd, proteins_list, protein_names, trajectory_dt, outputdir, base, plot, sep_blocks, quant=None, msd_avg=None, msd_wavg=None, msd_err=None): # works
    """
    Plot the Mean Squared Displacement (MSD) for each protein and the weighted average MSD.

    Args:
        msd (np.ndarray): MSD matrix (n_proteins x n_frames).
        proteins_list (list): List of protein atom groups.
        trajectory_dt (float): Timestep of the trajectory (converted to ns).
        outputdir (str): Directory to save the plot.
        base (str): Base filename prefix.
        n_blocks (int, optional): Number of blocks for averaging (default: 1).
        quant (list, optional): List of MSD values for each block.
        msd_avg (np.ndarray, optional): Average MSD across all proteins.
        msd_wavg (np.ndarray, optional): Weighted average MSD.
        msd_err (np.ndarray, optional): Error in MSD calculation.

    """
    # Get the number of frames and the timestep
    nframes = msd.shape[1]
    lagtimes = np.arange(nframes) * (trajectory_dt / 1000)  # Convert to ns
    colormap = cm.get_cmap('tab20b', len(proteins_list))
    
    if not sep_blocks:
        fig, axs = plt.subplots(1, 1, figsize=(3.54, 3))
        axs = [axs]  # Ensure axs is always iterable
        quant = [msd]
        n_blocks = 1
    else:
        fig, axs = plt.subplots(1, len(quant), figsize=(7.008, 3))
        n_blocks = len(quant)
    # Loop over blocks and plot MSD
    for block_idx in range(n_blocks):
        current_msd = quant[block_idx]
        for i, protein in enumerate(proteins_list):
            axs[block_idx].plot(lagtimes, current_msd[i, :], alpha=0.7, label=protein_names[i], color = colormap(i))

        # Set titles and labels
        if n_blocks > 1:
            axs[block_idx].set_title(f'Block {block_idx + 1}')
        axs[block_idx].set_xlabel('Time (ns)')
        axs[block_idx].set_ylabel('MSD ($\\mathrm{\\AA}^2$)')
        axs[block_idx].tick_params(axis='x')
        axs[block_idx].tick_params(axis='y')
        #axs[block_idx].legend()

    # Final block with averaged MSD
    if len(quant) == 1 and (msd_avg is not None or msd_wavg is not None or msd_err is not None):
        if msd_err is not None:
            for i, protein in enumerate(proteins_list):
                axs[block_idx].fill_between(
                    lagtimes, 
                    msd[i, :] - msd_err[i, :],
                    msd[i, :] + msd_err[i, :],
                    alpha=0.2,
                    label=f"Protein {i+1} Error",
                    color = colormap(i)
                )

        if msd_avg is not None:
            axs[block_idx].plot(lagtimes, msd_avg, '-', color='black', label='avg', linewidth=2)

        if msd_wavg is not None:
            axs[block_idx].plot(lagtimes, msd_wavg, '-', color='navy', label='weighted avg', linewidth=2)

        #axs[block_idx].legend()

    # Adjust layout and save plot
    plt.tight_layout()
    plt.show()

    if outputdir is not None:
        msd_dir = os.path.join(outputdir, 'MSD')
        os.makedirs(msd_dir, exist_ok=True)
        base_filename = os.path.join(msd_dir, base + "msd.png")
        fig.savefig(base_filename)
        print(f"Plot of MSD saved in {base_filename}.")


def plot_dt_fit(msd, com, proteins_list, trajectory_dt, outputdir, base, plot, start, end): # works
    """
    Plot the diffusion coefficient (Dt) fitting for each protein.

    Args:
        msd (np.ndarray): MSD matrix (n_proteins x n_frames).
        com (list): COM trajectory data.
        proteins_list (list): List of protein atom groups.
        trajectory_dt (float): Timestep of the trajectory (converted to ns).
        outputdir (str): Directory to save the plot.
        base (str): Base filename prefix.
    """
    
    def linear_func_through_origin(x, a):
        return a * x  # Linear function passing through origin

    #nframes = len(com)
    nframes = len(msd[0,:])
    lagtimes = np.arange(nframes) * (trajectory_dt / 1000)  # Convert to ns
    st = int(start / (trajectory_dt / 1000))
    en = int(end / (trajectory_dt / 1000))
    n_proteins = len(proteins_list)

    fig, axs = plt.subplots(1, 1, figsize=(3.54, 3))
    colormap = cm.get_cmap('tab20b', n_proteins)

    for i, protein in enumerate(proteins_list):
        popt, pcov = curve_fit(linear_func_through_origin, lagtimes[st:en], msd[i, st:en])
        axs.plot(lagtimes[st:en], msd[i, st:en], label=f'Protein {i+1}', color=colormap(i))
        axs.plot(lagtimes[st:en], linear_func_through_origin(lagtimes[st:en], *popt), 
                 label=f'Linear Fit {i+1}', color=colormap(i), linestyle='--')

    axs.set_xlabel('Time (ns)')
    axs.set_ylabel('MSD ($\\mathrm{\\AA}^2$)')
    axs.tick_params(axis='x')
    axs.tick_params(axis='y')
    #axs.legend()

    plt.tight_layout()

    # Save the plot
    if outputdir is not None:
        diff_dir = os.path.join(outputdir, 'Diffusion')
        os.makedirs(diff_dir, exist_ok=True)
        base_filename = os.path.join(diff_dir, base + "Dt_fit.png")
        fig.savefig(base_filename)

        print(f"Plot of translational diffusion extraction saved in {base_filename}.")


def plot_dt_vals(protein_names, D, D_avg, outputdir, base, plot): # works
    """
    Plot the diffusion coefficient (Dt) as a bar chart.

    Args:
        protein_names (list): List of protein names.
        D (np.ndarray): Array of diffusion coefficients for each protein.
        D_avg (float): Average diffusion coefficient.
        outputdir (str): Directory to save the plot.
        base (str): Base filename prefix.
    """
    colormap = cm.get_cmap('tab20b', len(protein_names))
    colors = [colormap(i) for i in range(len(protein_names))]
    
    fig, axs = plt.subplots(1, 1, figsize=(3.54, 3))

    axs.bar(protein_names, D, color = colors)
    axs.axhline(y=D_avg, color='black', linestyle='--', label="Average Dt")
    axs.set_xticklabels(protein_names, rotation=60, ha='right') 
    axs.set_ylabel('$D_t$ ($\\mathrm{\\AA}^2$/ns)')

    plt.tight_layout()

    # Save the plot
    if outputdir is not None:
        diff_dir = os.path.join(outputdir, 'Diffusion')
        os.makedirs(diff_dir, exist_ok=True)
        base_filename = os.path.join(diff_dir, base + "Dt.png")
        fig.savefig(base_filename)
    
        print(f"Plot of translational diffusion values saved in {base_filename}.")


def plot_avg_contacts(contacts_avg, protein_names, outputdir, base, plot):
    """
    Plot the average contact matrix between proteins.

    Args:
        contacts_avg (np.ndarray): Matrix of average contacts between proteins.
        protein_names (list): List of protein names.
        outputdir (str): Directory to save the plot.
        base (str): Base filename prefix.
    """

    fig, axs = plt.subplots(1, 1, figsize=(3.54, 3))

    # Create contact map
    cax = axs.imshow(contacts_avg, cmap="viridis")

    # Set the x and y axis labels according to the protein names
    nproteins = len(protein_names)
    axs.set_xticks(np.arange(nproteins))
    axs.set_yticks(np.arange(nproteins))

    # Label the x and y axes with the protein names
    axs.set_xticklabels(protein_names, rotation=90, ha='center')
    axs.set_yticklabels(protein_names, va='center')

    # Add colorbar
    fig.colorbar(cax, ax=axs)

    plt.tight_layout()

    # Save the plot
    cont_dir = os.path.join(outputdir, 'Contacts')
    os.makedirs(cont_dir, exist_ok=True)
    base_filename = os.path.join(cont_dir, base + "Contacts_avg.png")
    fig.savefig(base_filename)

    print(f"Plot of avarage (trajectory) contact map saved in {base_filename}.")


def plot_radius_of_gyration(rg, protein_names, outputdir, base, plot): # works
    """
    Plot the histogram of radius of gyration (Rg) for each protein.

    Args:
        rg (np.ndarray): Array of Rg values (n_proteins x n_frames).
        protein_names (list): List of protein names.
        outputdir (str): Directory to save the plot.
        base (str): Base filename prefix.
    """

    fig, axs = plt.subplots(1, 1, figsize=(3.54, 3))

    num_proteins = rg.shape[0]  # Number of proteins
    colormap = cm.get_cmap('tab20b', num_proteins)

    for i in range(num_proteins):
        axs.hist(rg[i, :], bins=20, alpha=0.7, color=colormap(i), label=protein_names[i])

    #axs.legend()
    axs.set_xlabel('$R_g$ ($\\mathrm{\\AA}$)')
    axs.set_ylabel('Count')

    plt.tight_layout()

    # Save the plot
    # Save the plot
    if outputdir is not None:
        rg_dir = os.path.join(outputdir, 'Rg')
        os.makedirs(rg_dir, exist_ok=True)
        base_filename = os.path.join(rg_dir, base + "Rg_dist.png")
        fig.savefig(base_filename)
        print(f"Plot of Radius of gyration histogram saved in {base_filename}.")


def plot_clusters(nclust_t, time_ns, outputdir, base, plot):
    """
    Plots the number of clusters over time.

    Args:
        nclust_t (array-like): Number of clusters at each time point.
        time_ns (array-like): Corresponding time values in nanoseconds.
        outputdir (str): Directory where the plot will be saved.
        base (str): Base name for the output file.
        plot (bool): Unused; included for compatibility or future use.

    Returns:
        None. Saves the plot as a PNG file in the 'Contacts' subdirectory.
    """
    fig, axs = plt.subplots(1, 1, figsize=(3.54, 3))
    axs.plot(time_ns, nclust_t, marker='o', color='navy')
    axs.set_xlabel('Time (ns)')
    axs.set_ylabel('$N_{clust}$')
    plt.tight_layout()
    cont_dir = os.path.join(outputdir, 'Contacts')
    base_filename = os.path.join(cont_dir, base + "Clusters.png")
    fig.savefig(base_filename)
    print(f"Plot of average (trajectory) clusters saved in {base_filename}.")

def plot_cluster_fractions(unique_sizes, fractions, outputdir, base, plot):
    """
    Plots the fraction of proteins in each cluster size.

    Args:
        unique_sizes (array-like): Unique cluster sizes (e.g. number of proteins per cluster).
        fractions (array-like): Corresponding fraction of proteins in each cluster size.
        outputdir (str): Directory where the plot will be saved.
        base (str): Base filename prefix for the output plot.
        plot (bool): Unused; included for compatibility or future extension.

    Returns:
        None. Saves a bar plot as a PNG file in the 'Contacts' subdirectory.
    """
    fig, axs = plt.subplots(1, 1, figsize=(3.54, 3))
    axs.bar(unique_sizes, fractions, color='navy')
    axs.set_xlabel('Cluster Size')
    axs.set_ylabel('Fraction$_p$')
    plt.tight_layout()
    cont_dir = os.path.join(outputdir, 'Contacts')
    base_filename = os.path.join(cont_dir, base + "Clusters_fractions.png")
    fig.savefig(base_filename)
    print(f"Plot of protein fractions in clusters saved in {base_filename}.")


def plot_avg_mindistance(distances, protein_names, masses = None):
    """
    Plots average minimum inter-protein distances, sorted by protein mass.

    Args:
        distances (ndarray): Minimum distances between proteins over time,
            shape (n_proteins, n_proteins, n_frames).
        masses (array-like): Masses of proteins, used to sort the plot.
        protein_names (list of str): Names or labels for each protein.

    Returns:
        None. Displays a bar plot of average minimum distances per protein.
    """
    # Average the minimum distances across all frames
    average_min_distances = np.mean(distances, axis=2)  # Shape (22, 22)
    
    # Symmetrize the matrix (min distance is the same in both directions)
    symmetrized_matrix = (average_min_distances + average_min_distances.T) / 2

    # Plot a bar plot for each protein showing its minimum distance to other proteins
    fig, axs = plt.subplots(1, 1, figsize=(3.54, 3))
    
    avg_protein_distances = []
    for protein_id in range(distances.shape[0]):
        protein_distances = symmetrized_matrix[protein_id, :]
        avg_protein_distances.append(np.mean(protein_distances))
    
    if masses is not None:
        sorted_indices = np.argsort(masses) 
        sorted_avg_distances = np.array(avg_protein_distances)[sorted_indices]
        print(sorted_indices)
        print(protein_names)
        sorted_protein_names = [protein_names[i] for i in sorted_indices]
        axs.bar(sorted_protein_names, sorted_avg_distances)
    else:
        axs.bar(protein_names, distances)
        
    axs.tick_params(axis='x', labelrotation=90)
    axs.set_xlabel('Proteins (sorted by mass)')
    axs.set_ylabel('Average Minimum Distances')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

