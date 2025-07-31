import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MDAnalysis import Universe
from MDAnalysis.analysis import msd as msd_MD
from MDAnalysis.analysis import align, distances, contacts
from MDAnalysis.analysis import contacts
from MDAnalysis.transformations import wrap, unwrap
from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.coordinates import PDB
from scipy.optimize import curve_fit
from lmfit import Model
import networkx as nx
import warnings
from tqdm import tqdm  # For progress tracking
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import os
import re

# Custom module imports
from file_io import *
from calculations import *
from visualization import *
from clustering import *

warnings.filterwarnings("ignore")


class Proteome:
    
    def __init__(self, tpr_file, indices_path, outputdir=None, base="", plot=True, selection="protein"): 
        """Initialize the Proteome class with TPR file and indices file"""
        self.tpr_file     = tpr_file
        self.indices_path = indices_path
        self.outputdir    = outputdir
        self.base         = base
        self.plot         = plot
        self.selection    = selection
        # Load MD system universe and create protein-only universe
        self.universe          = mda.Universe(tpr_file)
        self.proteins_universe = mda.Merge(self.universe.select_atoms(selection))
        # Extract protein atom groups, names, and residue IDs
        self.proteins_list, self.protein_names, self.protein_resids, *opt = separate_proteins(self.proteins_universe, self.indices_path)
        if not opt:
            self.proteins_masses = None
        else:
            self.proteins_masses = opt
        # Initialize results and other attributes
        self.results = {name: {} for name in self.protein_names}
        # Setup output directory if provided
        if self.outputdir:
            os.makedirs(self.outputdir, exist_ok=True)  
        else:
            print("No output directory provided. Results will not be saved to disk.")
        print("Proteome created!\nProteome contains: proteins_universe, proteins_list, protein_names, protein_resids")

    
    def add_trajectory(self, trajectory_file): 
        """Add a trajectory to the proteins universe"""
        self.proteins_universe.load_new(trajectory_file)
        print('Proteome trajectory loaded! \n Proteome contains now: proteins_universe.trajectory')

    
    def add_wrapped_trajectory(self, trajectory_file):
        """Add a trajectory to the proteins universe and wrap it"""
        temp_u = mda.Universe(self.tpr_file) 
        self.proteins_universe_wrap = mda.Merge(temp_u.select_atoms(self.selection)) 
        self.proteins_universe_wrap.load_new(trajectory_file)
        self.proteins_universe_wrap.trajectory.add_transformations(wrap(self.proteins_universe_wrap.atoms))
        self.proteins_list_wrap, _, _, *opt = separate_proteins(self.proteins_universe_wrap, self.indices_path)

    
    def center_of_masses(self, box_dim=None, append=True, ranges = None):
        """Compute the center of masses of the proteins"""
        self.com_universe, com_data = compute_center_of_masses(self.proteins_universe, self.proteins_list, box_dim, append, ranges)
        if not append or not hasattr(self, 'com'):
            self.com = com_data
            for i, name in enumerate(self.results):
                self.results[name]["com"] = com_data[:, i, :]
        else:
            self.com = np.concatenate((self.com, com_data), axis=0)
            for i, name in enumerate(self.results):
                self.results[name]["com"] = np.concatenate((self.results[name]["com"], com_data[:, i, :]), axis=0)
        print('Proteome Center of Mass (COM) trajectory computed! \n Proteome contains now: com, com_universe')
        return com_data

        
    def plot_com_trajectory(self): 
        """Plot the center of masses of the proteins in (x,y), (y,z), (x,z)"""
        plot_com_trajectories(
            results   = self.results,
            outputdir = self.outputdir,
            base      = self.base,
            plot      = self.plot
        )

    
    def write_com_to_xtc(self, filename): 
        """Save center of masses trajectory to a file, works only if outputdir was set!"""
        if self.outputdir is None:
            print("Error: 'outputdir' must be set when initializing proteome before writing the COM trajectory.")
            return 
        write_com_to_xtc(
            com_universe = self.com_universe,
            outputdir    = self.outputdir,
            base         = self.base,
            filename     = filename,
            n_proteins   = len(self.proteins_list)
        )

    
    def msd_com(self, ranges=None, n_blocks=1, HalfDt=False):
        """Compute msd from center of masses"""
        if ranges is not None and n_blocks > 1:
            warnings.warn("Both `ranges` and `n_blocks > 1` are specified. "
                "`ranges` will be ignored, and the full trajectory will be used for block analysis.")
        if not hasattr(self, 'com_universe'):
            self.center_of_masses()
            
        result = compute_msd(
            com_universe  = self.com_universe,
            com_data      = self.com,
            proteins_list = self.proteins_list,
            outputdir     = self.outputdir,
            base          = self.base,
            ranges        = ranges,
            n_blocks      = n_blocks,
            HalfDt        = HalfDt,
            plot_msd_func = self.plot_msd
        )
        if n_blocks == 1:
            self.msd = result
        else:
            self.msd = result[0]
            self.msd_err = result[1]
            self.msds_blocks = result[2]
        self.msd_avg = np.mean(self.msd, axis=0)
        for i, name in enumerate(self.results):
            self.results[name]["msd"] = self.msd[i, :]
        return self.msd

    
    def msd_com_weighted(self):
        """Compute msd from center of masses weighted over H-atoms"""
        if not hasattr(self, 'msd'):
            raise ValueError("MSD must be computed first. Run msd_com() before msd_com_weighted().")
        self.msd_wavg = compute_weighted_msd(
            msd_matrix    = self.msd,
            proteins_list = self.proteins_list,
            outputdir     = self.outputdir,
            base          = self.base
        )
        if hasattr(self, 'msds_blocks'):
            self.msds_wavg_blocks = []
            for i in range(len(self.msds_blocks)):
                self.msds_wavg_blocks.append(compute_weighted_msd(
                    msd_matrix    = self.msds_blocks[i],
                    proteins_list = self.proteins_list,
                    outputdir     = self.outputdir,
                    base          = self.base
                ))
        return self.msd_wavg

    
    def plot_msd(self, separateblocks = False): 
        """Plot msd in either a unique plot or separated by blocks"""
        if not hasattr(self, 'msd'):
            raise ValueError("MSD must be computed first. Run msd_com() before plot_msd().")
        if separateblocks:
            quant = self.msds_blocks
        else:
            quant = None
        plot_msd(
            msd           = self.msd,
            proteins_list = self.proteins_list,
            protein_names = self.protein_names,
            trajectory_dt = self.proteins_universe.trajectory.dt,
            outputdir     = self.outputdir,
            base          = self.base,
            plot          = self.plot,
            sep_blocks    = separateblocks,
            quant         = quant,
            msd_avg       = self.msd_avg if hasattr(self, 'msd_avg') else None,
            msd_wavg      = self.msd_wavg if hasattr(self, 'msd_wavg') else None,
            msd_err       = self.msd_err if hasattr(self, 'msd_err') else None
        )

        
    def dt_com(self, start=0.3, end=5, offset = False): 
        """Compute diffusion coefficient by fitting msd (blockwise) in a given range (ns)"""
        if not hasattr(self, 'msd'):
            raise ValueError("MSD must be computed first. Run msd_com() before dt_com().")
        self.D = compute_diffusion_coefficient(
            msd_matrix    = self.msd,
            proteins_list = self.proteins_list,
            trajectory_dt = self.proteins_universe.trajectory.dt,
            outputdir     = self.outputdir,
            base          = self.base,
            start         = start,
            end           = end,
            offset        = offset
        )
        self.D_avg = np.mean(self.D)
        for i, name in enumerate(self.results):
            self.results[name]["Dt"] = self.D[i]
        return self.D

    
    def dt_com_weighted(self, start=0.3, end=5, offset = False): 
        """Compute diffusion coefficient by fitting msd (blockwise) weighted over H-atoms"""
        if not hasattr(self, 'msd_wavg'):
            raise ValueError("Weighted MSD must be computed first. Run msd_com_weighted() before dt_com_weighted().")
        self.D_wavg = compute_weighted_diffusion_coefficient(
            msd_wavg      = self.msd_wavg,
            trajectory_dt = self.proteins_universe.trajectory.dt,
            outputdir     = self.outputdir,
            base          = self.base,
            start         = start,
            end           = end,
            offset        = offset
        )
        if hasattr(self, 'msds_blocks'):
            self.D_wavg_blocks = []
            for i in range(len(self.msds_blocks)):
                self.D_wavg_blocks.append(compute_weighted_diffusion_coefficient(
                    msd_wavg      = self.msds_wavg_blocks[i],
                    trajectory_dt = self.proteins_universe.trajectory.dt,
                    outputdir     = self.outputdir,
                    base          = self.base,
                    start         = start,
                    end           = end,
                    offset        = offset
                ))
            self.D_wavg = np.mean(self.D_wavg_blocks) 
            self.D_wavg_err = np.std(self.D_wavg_blocks)/np.sqrt(len(self.msds_wavg_blocks))
        return self.D_wavg

        
    def plot_dt_fit(self, start=0.3, end=5): 
        """Plot the linear fit of msd to find the diffusion coefficient"""
        if not hasattr(self, 'msd'):
            raise ValueError("MSD must be computed first. Run msd_com() before plot_dt().")
        plot_dt_fit(
            msd           = self.msd,
            com           = self.com,
            proteins_list = self.proteins_list,
            plot          = self.plot,
            trajectory_dt = self.proteins_universe.trajectory.dt,
            outputdir     = self.outputdir,
            base          = self.base, 
            start         = start,
            end           = end
        )

    
    def plot_dt(self, start=0.3, end=5): 
        """Plot diffusion coefficients in a bar plot"""
        if not hasattr(self, 'D'):
            raise ValueError("Diffusion coefficients (D) must be computed first. Run dt_com() before plot_dt_vals().")
        plot_dt_vals(
            protein_names = self.protein_names,
            D             = self.D,
            D_avg         = self.D_avg,
            plot          = self.plot,
            outputdir     = self.outputdir,
            base          = self.base
        )

    
    def contacts(self, type_cont='allres', radius_contact=8, skip=1, processes = 15, 
                     create_trajectory=False, show_wrap=False, output_file="contacts"): 
        """Compute the number of contacts between proteins for a given radius"""
        
        if not hasattr(self, 'proteins_list_wrap') or not hasattr(self, 'proteins_universe_wrap'):
            raise ValueError("Wrapped proteins must be available. Run add_wrapped_trajectory() first.")
        self.contacts_avg, list_n_contacts, betas_array = compute_contacts(
            proteins_list_wrap  = self.proteins_list_wrap,
            universe_wrap       = self.proteins_universe_wrap,
            processes           = processes,
            radius_contact      = radius_contact,
            skip                = skip,
            selection           = self.selection
        )
        save_contacts(
            matrix_n_contacts   = self.contacts_avg,
            list_n_contacts     = list_n_contacts,
            betas_array         = betas_array,
            outputdir           = self.outputdir,
            outputfile          = output_file,
            base                = self.base,
            create_trajectory   = create_trajectory,
            input_pdb           = self.input_file_pdb if hasattr(self, 'input_file_pdb') else None,
            show_wrap           = show_wrap,
            universe_wrap       = self.proteins_universe_wrap,
            skip                = skip,
            selection           = self.selection
        )
        self.betas_input_file = output_file
        return self.contacts_avg, list_n_contacts, betas_array

    
    def plot_average_contacts(self): 
        """Plot a map of the average contacts (trajectory) between proteins"""
        if not hasattr(self, 'contacts_avg'):
            raise ValueError("Contact map must be computed first. Run get_contacts() before plot_avg_contacts().")
        plot_avg_contacts(
            contacts_avg   = self.contacts_avg,
            protein_names  = self.protein_names,
            outputdir      = self.outputdir,
            base           = self.base,
            plot           = self.plot
        )

        
    def clusters_one_frame(self, contacts, Ncrit): 
        """Computes the clusters of proteins in one frame for a given critical number of contacts to define a cluster"""
        if not hasattr(self, 'proteins_list'):
            raise ValueError("Proteins list must be available. Ensure the proteome is initialized correctly.")
        return clusters_one_frame(
            proteins_list = self.proteins_list,
            contacts      = contacts,
            Ncrit         = Ncrit
        )

    
    def clusters(self, Contacts, NCrit, create_trajectory = False, show_wrap = True, skip = 1):
        """Computes the number of clusters and fractions of proteins belonging to the clusters over all frames"""
        input_file = self.betas_input_file
        outputdir = self.outputdir
        base = self.base
        nclust_t, sizes_t, fractions, unique_sizes, betas_array = clusters(
            proteins_list     = self.proteins_list, 
            Contacts          = Contacts,
            NCrit             = NCrit,
            input_file_pdb    = input_file, 
            outputdir         = outputdir, 
            base              = base,
            create_trajectory = create_trajectory, 
            show_wrap         = show_wrap, 
            skip              = skip
        )
        save_clusters(
            nclust_t          = nclust_t, 
            betas_array       = betas_array, 
            outputdir         = outputdir, 
            input_file_pdb    = input_file, 
            base=base
        )
        return nclust_t, sizes_t, fractions, unique_sizes, betas_array

    
    def plot_clusters(self, nclust_t):
        """Plots the number of clusters vs time"""
        outputdir = self.outputdir
        base      = self.base
        time_ns   = np.linspace(0, self.proteins_universe.trajectory.totaltime/1000, len(nclust_t)) 
        plot_clusters(nclust_t, time_ns, outputdir, base, plot=self.plot)

    
    def plot_cluster_fractions(self, unique_sizes, fractions):
        """Plots the fractions of proteins belonging to each cluster"""
        outputdir = self.outputdir
        base      = self.base
        plot_cluster_fractions(unique_sizes, fractions, outputdir, base, plot=self.plot)


    def radius_of_gyration(self, skip=1): 
        """Computes the radius of gyration of the proteins over trajectory"""
        if not hasattr(self, 'proteins_list'):
            raise ValueError("Proteins list is missing. Make sure proteins have been loaded.")
        self.rg = compute_radius_of_gyration(
            proteins_list = self.proteins_list,
            trajectory    = self.proteins_universe.trajectory,
            outputdir     = self.outputdir,
            base          = self.base,
            skip          = skip
        )
        for i, name in enumerate(self.results):
            self.results[name]["Rg"] = self.rg[i, :]
        return self.rg

    
    def plot_radius_of_gyration(self): 
        """Plots the radius of gyration of the proteins in histogram over trajectory"""
        if not hasattr(self, 'rg'):
            raise ValueError("Radius of gyration must be computed first. Run calculate_radius_of_gyration() first.")
        plot_radius_of_gyration(
            rg            = self.rg,
            protein_names = self.protein_names,
            outputdir     = self.outputdir,
            base          = self.base,
            plot          = self.plot
        )

    
    def minimum_distances(self, skip, radius_contact = 3, processes = 25, verbose = True): 
        outputdir = self.outputdir
        base      = self.base
        mind = compute_mindist(
            proteins_list_wrap = self.proteins_list_wrap, 
            universe_wrap      = self.proteins_universe_wrap,
            processes          = processes, 
            radius_contact     = radius_contact, 
            skip               = skip, 
            selection          = self.selection,
            verbose            = verbose
        )
        if outputdir is not None:
            save_mindist(mind      = mind, 
                         outputdir = outputdir, 
                         base      = base
                        )
        return mind

    
    def plot_avg_mindistance(self, mindistances):
        masses     = self.proteins_masses
        prot_names = self.protein_names
        plot_avg_mindistance(distances     = mindistances, 
                             masses        = masses,
                             protein_names = prot_names
                            )    
    
