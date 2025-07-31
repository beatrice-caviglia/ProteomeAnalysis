import numpy as np 
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis import *
from MDAnalysis import analysis
from scipy.optimize import curve_fit
#from diffusion import *
#from translational_diffusion import *
from DC_correction_PBC import *
from scipy.integrate import quad
from scipy.special import spherical_jn
from scipy.optimize import minimize
import os
import time
import glob


def read_residue_ranges(path, chains = True):
    filename = path + "protein_indeces.txt"
    if chains:
        filename = path + "chain_indeces.txt"
    residue_ranges = []
    protein_names = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            protein_name = parts[0]
            start_residue = int(parts[1])
            end_residue = int(parts[2])
            residue_ranges.append((start_residue, end_residue))
            protein_names.append(protein_name)
    return residue_ranges, protein_names

def rename_duplicates(old):
    seen = {}
    for x in old:
        if x in seen:
            seen[x] += 1
            yield "%s_%d" % (x, seen[x]+1)
        else:
            seen[x] = 0
            yield x
            
# READ the RDF file 
def read_rdf_xvg(file_path):
    rdf_data = np.loadtxt(file_path, comments=['#', '@'], skiprows=23)
    return rdf_data


# Define the radial distribution function
def rho_h(r, rdf_data):
    r_values, rdf_values = rdf_data[:, 0], rdf_data[:, 1]
    rdf_at_r = np.interp(r, r_values, rdf_values)
    return rdf_at_r

# Define Bn function
def Bn(n, q, rdf_data):
    def integrand(r):
        return rho_h(r, rdf_data) * (spherical_jn(n, q*r)**2)
    integral, _ = quad(integrand, 0, np.inf)
    return (2*n + 1) * integral


def sum_function2(Dx, q, Dr, Dt, rdf_data):
    total_sum = 0
    for l in range(551): 
        numerator = (Dr*l*(l+1)) + ((Dt - Dx)*(q**2))
        denominator = ((Dr*l*(l+1)) + ((Dt + Dx)*(q**2))) ** 2
        total_sum += Bn(l, q, rdf_data) * (numerator / denominator)
    # print('Total sum', total_sum)
    return total_sum

def objective_function(Dx, q, Dr, Dt, rdf_data):
    return abs(sum_function2(Dx, q, Dr, Dt, rdf_data))

def find_temperatures(path, pattern, unF):
    folders = []
    for item in os.listdir(path):
        if unF:
            if os.path.isdir(os.path.join(path, item)) and item.endswith(pattern) and "UnF" in item:
                temperature = int(item.replace(pattern, "")[4:])
                folders.append(temperature)
        else:
            if os.path.isdir(os.path.join(path, item)) and item.endswith(pattern) and "UnF" not in item:
                temperature = int(item.replace(pattern, "")[:])
                folders.append(temperature)
            
    return folders


