import numpy as np
import os
from scipy.optimize import curve_fit
import MDAnalysis as mda
from MDAnalysis import Universe
import MDAnalysis.analysis.msd as msd_MD

def compute_center_of_masses(proteins_universe, proteins_list, box_dim=None, append=True, ranges = None): 
    """
    Computes center of mass (COM) trajectory for all proteins.

    Args:
        proteins_universe (MDAnalysis.Universe): Universe with protein trajectories.
        proteins_list (list): List of protein atom groups.
        box_dim (array, optional): Simulation box dimensions; defaults to universe box.
        append (bool, optional): Append COMs to existing data (default: True).

    Returns:
        tuple:
            - new_universe (MDAnalysis.Universe): Universe with COM trajectory.
            - timeseries (ndarray): Array of COMs over time (nframes x nprot x dim).
    """

    if ranges is not None:
        timeseries = []
        for ts in mda.log.ProgressBar(proteins_universe.trajectory[ranges[0]:ranges[1]]):
            coms = [prot.center_of_mass() for prot in proteins_list]
            timeseries.append(coms)
    else:
        timeseries = []  
        for ts in mda.log.ProgressBar(proteins_universe.trajectory):
            coms = [prot.center_of_mass() for prot in proteins_list]
            timeseries.append(coms)

    # Reshape COM data for new universe creation
    n_proteins = len(proteins_list)
    positions = np.array(timeseries).reshape(-1, n_proteins, 3)

    # Create a new universe for COM trajectory
    new_universe = Universe.empty(n_atoms=n_proteins, n_residues=n_proteins, n_segments=1, trajectory=True)
    new_universe.load_new(positions, dt=proteins_universe.trajectory.dt / 1000)

    # Set box dimensions
    box_dimensions = proteins_universe.dimensions if box_dim is None else box_dim
    for ts in new_universe.trajectory:
        ts.dimensions = box_dimensions
    # if ranges is not None:
    #     timeseries = np.array(timeseries)
    #     timeseries = timeseries[ranges[0]:ranges[1], :,:]
    return new_universe, np.array(timeseries)


def compute_average_over_blocks(n_blocks, quantity_func, n_frames, n_proteins, *args, **kwargs): 
        """
        Computes block-averaged quantities over a trajectory.
    
        Args:
            n_blocks (int): Number of blocks to divide the trajectory into.
            quantity_func (callable): Function to compute the desired quantity for a block.
            n_frames (int): Total number of frames in the trajectory.
            n_proteins (int): Number of proteins (passed for context or downstream use).
            *args: Additional positional arguments for `quantity_func`.
            **kwargs: Additional keyword arguments for `quantity_func`.
    
        Returns:
            tuple:
                - average_quantity (ndarray): Mean quantity over all blocks.
                - sem_quantity (ndarray): Standard error of the mean across blocks.
                - quantities (list): List of quantities computed for each block.
        """
        block_size = n_frames // n_blocks
        quantities = []                                      
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            print(f"Processing block {i + 1}/{n_blocks}, frames {start}-{end}")
            quantity = quantity_func(ranges=(start, end), *args, **kwargs)
            quantities.append(quantity)
        average_quantity = np.mean(quantities, axis=0)
        std_quantity = np.std(quantities, axis=0, ddof=1)
        sem_quantity = std_quantity / np.sqrt(n_blocks)
        return average_quantity, sem_quantity, quantities


def compute_average_over_blocks2(n_blocks, *args, **kwargs): 
        quantities = []                                      
        for i in range(n_blocks):
            quantity = quantity_func(*args, **kwargs)
            quantities.append(quantity)
        average_quantity = np.mean(quantities, axis=0)
        std_quantity = np.std(quantities, axis=0, ddof=1)
        sem_quantity = std_quantity / np.sqrt(n_blocks)
        return average_quantity, sem_quantity, quantities

def shift_array(arr, dt): # works
    """Shifts array by dt and returns two equal-length segments: original and shifted."""
    arr1 = arr[dt:dt+int(len(arr)/2)]
    arr2 = arr[:len(arr1)]
    return arr2, arr1
    

def dist(x, y): # works
    """Function to calculate the squared displacement between two sets of points"""
    a = np.sum(np.square(y - x), axis=1)  # Sum of squared differences over all 3 dimensions
    dd = np.sum(a)
    return dd / len(x)


def compute_msd(com_universe, com_data, proteins_list, outputdir, base, ranges=None, n_blocks=1, HalfDt=False, plot_msd_func=None): # works
    """
    Compute Mean Squared Displacement (MSD) for each protein.
    
    Args:
        com_universe (MDAnalysis.Universe): Universe containing the COM trajectory.
        com_data (list or np.ndarray): Precomputed COM data for all proteins.
        proteins_list (list): List of protein atom groups.
        outputdir (str): Directory where output files will be saved.
        base (str): Base filename prefix.
        ranges (tuple, optional): Range of frames to consider.
        n_blocks (int, optional): Number of blocks for averaging (default: 1).
        HalfDt (bool, optional): Whether to compute half-time MSD.
        Plot (bool, optional): Whether to plot MSD results.
        compute_average_over_blocks (function, optional): Function for averaging MSD over blocks.
        plot_msd_func (function, optional): Function to plot MSD.

    Returns:
        np.ndarray: Computed MSD matrix (n_proteins x n_frames).
    """
    
    # Handle block-wise computation
    if n_blocks > 1:
        original_n_blocks = n_blocks
        n_blocks = 1  # Prevent recursion
        
        n_frames = len(com_data)
        n_proteins = len(proteins_list)
        
        average_quantity, sem_quantity, quantities = compute_average_over_blocks(
            original_n_blocks, compute_msd, n_frames, n_proteins, com_universe, com_data, proteins_list, outputdir, base, HalfDt=HalfDt
        )

        if outputdir is not None:
            msd_dir = os.path.join(outputdir, 'MSD')
            os.makedirs(msd_dir, exist_ok=True)
    
            np.save(os.path.join(msd_dir, "msd_"+base+".npy"), average_quantity)
            np.save(os.path.join(msd_dir, "msd_err_"+base+".npy"), sem_quantity)

        return average_quantity, sem_quantity, quantities

    # Initialize MSD matrix
    n_frames = len(com_data)
    n_proteins = len(proteins_list)

    MSD = np.zeros((n_proteins, n_frames if ranges is None else ranges[1] - ranges[0]))

    # Compute MSD using EinsteinMSD if HalfDt is False
    if not HalfDt:
        for i, protein in enumerate(proteins_list):
            msd_calculator = msd_MD.EinsteinMSD(com_universe, select=f'index {i}', msd_type='xyz', fft=True)
            msd_calculator.run(start=ranges[0], stop=ranges[1]) if ranges else msd_calculator.run(verbose=False)
            MSD[i, :] = msd_calculator.results.timeseries

    # Custom HalfDt MSD Calculation
    else:
        def shift_array(arr, dt):
            arr1 = arr[dt:dt + int(len(arr) / 2)]
            arr2 = arr[:len(arr1)]
            return arr2, arr1

        def dist(x, y):
            return np.sum(np.square(y - x), axis=1).sum() / len(x)

        MSD = np.zeros((n_proteins, int(n_frames / 2) if ranges is None else int((ranges[1] - ranges[0]) / 2)))

        for i, protein in enumerate(proteins_list):
            traj = np.array([el[i] for el in com_data])
            if ranges:
                traj = traj[ranges[0]:ranges[1], :]

            msd_pp = np.zeros(len(traj) // 2)
            for dt in range(1, len(msd_pp)):
                cmt1, cmt2 = shift_array(traj, dt)
                msd_pp[dt] = dist(cmt1, cmt2)

            MSD[i, :] = msd_pp

    # Save computed MSD
    if outputdir is not None:
        msd_dir = os.path.join(outputdir, 'MSD')
        os.makedirs(msd_dir, exist_ok=True)
        np.save(os.path.join(msd_dir, "msd_"+base+".npy"), MSD)

    print(f'Proteome time-windowed MSD matrix (n_proteins x n_frames) computed!')
    return MSD



def compute_weighted_msd(msd_matrix, proteins_list, outputdir, base, blocks = False): 
    """
    Compute the weighted mean squared displacement (MSD) based on the number of atoms per protein.

    Args:
        msd_matrix (np.ndarray): Precomputed MSD values (n_proteins x n_frames).
        proteins_list (list): List of protein atom groups.
        outputdir (str): Directory where output files will be saved.
        base (str): Base filename prefix.

    Returns:
        np.ndarray: Weighted average MSD across proteins.
    """
    
    n_proteins = len(proteins_list)
    if msd_matrix is None or msd_matrix.shape[0] != n_proteins:
        raise ValueError("MSD matrix does not match the number of proteins.")

    # Get number of atoms per protein
    n_atoms = np.array([p.n_atoms for p in proteins_list])

    # Compute weighted average MSD
    MSD_avg = np.sum(n_atoms * np.transpose(msd_matrix), axis=1) / np.sum(n_atoms)

    # Save weighted MSD
    if outputdir is not None:
        msd_dir = os.path.join(outputdir, 'MSD')
        os.makedirs(msd_dir, exist_ok=True)
        np.save(os.path.join(msd_dir, "msd_wavg_"+base+".npy"), MSD_avg)

    print("Proteome weighted (H-atoms) time-windowed MSD matrix (n_frames) computed!")
    return MSD_avg

def linear_func_through_origin(x, a):
        return a * x  # Linear function passing through origin
def linear_func_offset(x, a, b):
        return a * x + b

def compute_diffusion_coefficient(msd_matrix, proteins_list, trajectory_dt, outputdir, base, start=0.3, end=5, offset = False): # works
    """
    Compute the diffusion coefficient (Dt) for each protein using MSD slopes.

    Args:
        msd_matrix (np.ndarray): Precomputed MSD values (n_proteins x n_frames).
        proteins_list (list): List of protein atom groups.
        trajectory_dt (float): Timestep of the trajectory (converted to ns).
        outputdir (str): Directory where output files will be saved.
        base (str): Base filename prefix.
        start (float, optional): Start time for fitting (in ns).
        end (float, optional): End time for fitting (in ns).

    Returns:
        np.ndarray: Computed diffusion coefficients for each protein.
    """
    n_frames = msd_matrix.shape[1]
    lagtimes = np.arange(n_frames) * (trajectory_dt / 1000)  # Convert to ns
    st = int(start / (trajectory_dt / 1000))
    en = int(end / (trajectory_dt / 1000))

    D = []
    if not offset:
        for i, protein in enumerate(proteins_list):
            popt, _ = curve_fit(linear_func_through_origin, lagtimes[st:en], msd_matrix[i, st:en])
            D.append((1 / 6) * popt[0])  # Compute diffusion coefficient
    else:
        for i, protein in enumerate(proteins_list):
            popt, _ = curve_fit(linear_func_offset, lagtimes[st:en], msd_matrix[i, st:en])
            D.append((1 / 6) * popt[0])  # Compute diffusion coefficient        

    # Save diffusion coefficients
    if outputdir is not None:
        diff_dir = os.path.join(outputdir, 'Diffusion')
        os.makedirs(diff_dir, exist_ok=True)
        np.save(os.path.join(diff_dir, "Dt_"+base+".npy"), np.array(D))
        print(f"Proteome diffusion coefficients computed and saved in {diff_dir}.")
    return np.array(D)


def compute_weighted_diffusion_coefficient(msd_wavg, trajectory_dt, outputdir, base, start=0.3, end=5, offset = False): # works
    """
    Compute the weighted diffusion coefficient (Dt_wavg) from the weighted MSD.

    Args:
        msd_wavg (np.ndarray): Precomputed weighted MSD values.
        trajectory_dt (float): Timestep of the trajectory (converted to ns).
        outputdir (str): Directory where output files will be saved.
        base (str): Base filename prefix.
        start (float, optional): Start time for fitting (in ns).
        end (float, optional): End time for fitting (in ns).

    Returns:
        float: Computed weighted diffusion coefficient.
    """

    n_frames = len(msd_wavg)
    lagtimes = np.arange(n_frames) * (trajectory_dt / 1000)  # Convert to ns
    st = int(start / (trajectory_dt / 1000))
    en = int(end / (trajectory_dt / 1000))

    # Fit a linear function to obtain the slope
    if not offset:
        popt, _ = curve_fit(linear_func_through_origin, lagtimes[st:en], msd_wavg[st:en])
        D_wavg = (1 / 6) * popt[0]  # Compute diffusion coefficient
    else:
        popt, _ = curve_fit(linear_func_offset, lagtimes[st:en], msd_wavg[st:en])
        D_wavg = (1 / 6) * popt[0]  # Compute diffusion coefficient

    # Save weighted diffusion coefficient
    if outputdir is not None:
        diff_dir = os.path.join(outputdir, 'Diffusion')
        os.makedirs(diff_dir, exist_ok=True)
        np.save(os.path.join(diff_dir, "Dt_wavg_"+base+".npy"), np.array(D_wavg))
        print(f"Proteome weighted (H-atoms) diffusion coefficient computed and saved in {diff_dir}.")
    return D_wavg


def compute_radius_of_gyration(proteins_list, trajectory, outputdir, base, skip=1): 
    """
    Compute the radius of gyration (Rg) for each protein over the trajectory.

    Args:
        proteins_list (list): List of protein atom groups.
        trajectory (MDAnalysis.trajectory): MD trajectory to iterate over.
        outputdir (str): Directory where output files will be saved.
        base (str): Base filename prefix.
        skip (int, optional): Step size for iterating over frames (default: 1).

    Returns:
        np.ndarray: Radius of gyration values (n_proteins x n_frames).
    """
    n_proteins = len(proteins_list)
    radii_of_gyration = np.zeros((n_proteins, len(trajectory[::skip])))

    # Compute Rg over trajectory frames
    for frame_idx, frame in enumerate(trajectory[::skip]):
        for j, protein in enumerate(proteins_list):
            radii_of_gyration[j, frame_idx] = protein.radius_of_gyration()

    # Save results
    if outputdir is not None:
        rg_dir = os.path.join(outputdir, 'Rg')
        os.makedirs(rg_dir, exist_ok=True)
        np.save(os.path.join(rg_dir, "Rg_time"+base+".npy"), radii_of_gyration)
        print(f"Proteome radius of gyration computed and saved in {rg_dir}.")
    return radii_of_gyration
