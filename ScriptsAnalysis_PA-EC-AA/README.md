### OVERVIEW:
This folder contains the main Python and Jupyter Notebook scripts (.py and .ipynb) used in the article: "Cytoplasmic Fluidity and the Cold Life: Proteome Stability is Decoupled from Viability in Psychrophiles". 

### SCRIPTS AND FUNCTIONS:

1) 01_Evaluate-global-diffusion.ipynb:
   - Computes the global diffusion of protein chains using translational/rotational diffusion coefficients, and radial distribution functions.
   - Inputs:
     - Translational and rotational diffusion coefficients (computed with scripts: create_unit_vectors.py, compute_dr.py, compute_msd.py)
     - Radial distribution function
     - Folders containing: TPR file, Chain indices file (txt file with the indeces of each chain in the system), Temperatures

2) 02_Translational-diffusion-blockwise.ipynb
   - Performs a blockwise analysis of translational diffusion coefficients for proteins within the simulation.
   - Inputs:
     - Trajectory files (.xtc)
     - Protein index file (file containing index ranges of proteins)
     - TPR file

3) 03_RunContacts.ipynb
   - Computes inter-protein contacts of proteins during a simulation.
   - Inputs:
     - Trajectory files (.xtc)
     - Protein index file (file containing index ranges of proteins)
     - TPR file
    - Note: This code is computationally expensive and runs parallel with multiprocessing. A high skipping rate of frames can be 
      selected to run this faster.

Additional analysis tools are provided in the 'source_codes' folder. These include:
- Scripts to create unit vectors, compute rotational diffusion coefficients, compute translational diffusion coefficients over whole trajectory
- Utilities for correcting diffusion coefficients for periodic boundary conditions (PBC)
- Optimization of weighted global diffusion coefficients to fit experimental data

### Operating System tested:
Ubuntu 20.04.5 LTS  
Windows 11 (except for code that required multiprocessing, e.g.: Codes 1) and 3).)

### Software Dependencies:
Python: 3.12.1  
Jupyter Notebook: 7.0.8  
MDAnalysis: 2.7.0  
numpy: 1.26.4  
pandas: 2.2.1  
matplotlib: 3.8.0   
scipy: 1.12.0  
lmfit: 1.2.2  
networkx: 3.4.2  
tqdm: 4.65.0  

### Demo:
The scripts can be tested with available data on Zenodo:   
https://zenodo.org/records/16603402  
https://zenodo.org/records/16600218  
https://zenodo.org/records/16605838  
Note: Adjust the paths in the scripts according to the downloading path and the System. The first link includes specific examples for runnning specific codes.



