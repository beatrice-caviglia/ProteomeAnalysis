import numpy as np 
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from DC_correction_PBC import *
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from lmfit import Model
import pickle
import lmfit
from scipy.interpolate import interp1d


# Linear model
def linear(x, a, b):
    return a * x + b

# Find temperatures for different system states
def find_temperatures(path, pattern):
    return [
        int(item.replace(pattern, ""))
        for item in os.listdir(path)
        if os.path.isdir(os.path.join(path, item)) and item.endswith(pattern) and "UnF" not in item
    ]

def find_temperatures_unF(path, pattern):
    return [
        int(item[4:-1].replace(pattern, ""))
        for item in os.listdir(path)
        if os.path.isdir(os.path.join(path, item)) and item.endswith(pattern) and "UnF" in item
    ]

def find_temperatures_punF(path, pattern, punf):
    full_path = os.path.join(path, 'partial_unfolding', punf)
    return [
        int(item.replace(pattern, ""))
        for item in os.listdir(full_path)
        if os.path.isdir(os.path.join(full_path, item)) and item.endswith(pattern)
    ]

# Plot translational diffusion
def plot_Dt(org, sys, ax, unF, punf='', color='orange', label='label'):
    # Map organism to short code
    org_map = {'Psychro': 'PA', 'Aquifex': 'AA', 'Ecoli': 'EC'}
    org_short = org_map.get(org, '')

    # Normalize punf string
    punf_clean = punf.replace('_', '').replace('-', '_')

    # Load data
    base_path = f"../../02a-Simulations/Systems/{org}/Subbox{sys}/"
    dt_path = f"../../02a-Simulations/Diffusion/Translational/Average_DT_blockwise/"
    DT = np.load(f"{dt_path}DT-{org}-System{sys}{punf}.npy")
    DT_err = np.load(f"{dt_path}DT-{org}-System{sys}{punf}_err.npy")

    # Load temperature data
    Temp = np.array(find_temperatures(base_path, 'K'))
    if 'unF' in punf:
        Temp = np.array(find_temperatures_unF(base_path, 'K'))
    if len(punf) == 7:
        Temp = np.array(find_temperatures_punF(base_path, 'K', punf[1:]))
        

    # Fix known data inconsistencies
    if org == 'Ecoli' and sys == '4':
        DT, DT_err = DT[:-1], DT_err[:-1]
    if org == 'Aquifex' and sys == '4':
        if punf == '':
            DT, DT_err, Temp = DT[3:], DT_err[3:], Temp[3:]
        elif punf == '-unF':
            DT, DT_err, Temp = DT[2:], DT_err[2:], Temp[2:]

    # Apply crowding correction
    protein_file = os.path.join(base_path, "protein_indeces.txt")
    correction = get_correction_crowded(17, Temp, protein_file, org_short, punF=punf_clean)
    DT += correction

    # Fit and plot
    popt, _ = curve_fit(linear, Temp, DT, sigma=DT_err, absolute_sigma=True)
    ax.plot(Temp, linear(Temp, *popt), color=color, linestyle='-', alpha=0.7)
    ax.errorbar(Temp, DT, DT_err, fmt='o', color=color, label=label)

    return DT, DT_err



# Helper to load pickle files
def load_pickle(filename, path):
    with open(os.path.join(path, filename), 'rb') as f:
        return pickle.load(f)
        
def load_step2(path):
    npz_data = np.load(os.path.join(path, 'fit_results.npz'))
    x_fit = npz_data['x_fit']
    fit = npz_data['fit']
    fit_err = npz_data['fit_err']
    data = load_pickle('val.pkl', path)
    data_err = load_pickle('err.pkl', path)
    all_temp = load_pickle('all_temp.pkl', path)
    Dt = load_pickle('dt.pkl', path)
    t0 = load_pickle('t0.pkl', path)
    return all_temp, data, data_err, x_fit, fit, fit_err, t0, Dt

def compute_experimental_au(data, err, all_temp, xfit, fit, fit_err, T0, Dt,
                            ax=None, p_color='black', var='dG', TCD=None, label=None,
                            skip=0, return_fit=False):
    
    dG = np.array([data, err])
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
        created_ax = True

    if TCD is not None:
        ax.axvline(TCD, color='black')
        ax.text(TCD + 1, 0.62, 'T$_{CD}$', fontsize=10, color='black')

    ax.errorbar(all_temp[skip:], dG[0][skip:], dG[1][skip:], fmt='s', fillstyle='none', label=label, color=p_color)

    # Compute and plot fitted curve
    T0_C = T0 - 273.15
    t_star = (np.array(xfit) - T0_C) / Dt
    a_u_exp = theta(t_star)
    ax.plot(xfit + 273.15, fit, '-', color=p_color)

    # Final formatting if axis was created here
    if created_ax:
        ax.set_ylim((0.1, 2))
        fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
        plt.show()

    # Define fitted function
    def au(T):
        return 1 / (1 + np.exp(-((T - 273.15) - T0_C) / Dt))

    # Optionally return fitting data
    if return_fit:
        return dG, T0_C, Dt, au, xfit, fit

    return au


def theta(x):
    return 1/(1 + np.e**(-x))

def power_law(x, p):
    return x**p

def find_au_vs_ru(org, sys, color = 'black', ax = None):

    # Read diffusion coefficients at various unfolding rates
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    punf_suffixes = ['', '-unF_25', '-unF_50', '-unF_75', '-unF']
    unF_flags = [False, False, False, False, True]    
    D_all = []
    D_all_err = []
    for punf, unF in zip(punf_suffixes, unF_flags):
        D, D_err = plot_Dt(org, sys, ax2, unF=unF, punf=punf, color='black', label="")
        D_all.append(D[:])
        D_all_err.append(D_err[:])
    plt.close(fig2)

    # Compute a_u paramter
    D, D_unF = D_all[0], D_all[-1]
    D_err, D_unF_err = D_all_err[0], D_all_err[-1]
    a_u = np.zeros((len(D_all), D_all[1].shape[0]))
    a_u_err = np.zeros((len(D_all), D_all[1].shape[0]))

    for (i,D_curr) in enumerate(D_all):
        nom = np.array((D_curr - D[:]))
        den = np.array((D_unF[:] - D[:]))
        a_u[i, :] = nom/den     
        

    # Fit a_u with power model and extract exponent p
    power_model = Model(power_law)
    params = power_model.make_params(p = 0.1)
    errors = np.std(a_u, axis=1)
    errors[0] = 1e-5
    errors[-1] = 1e-5
    result = power_model.fit(np.mean(a_u, axis = 1), params, weights = 1/errors, x=np.array([0, 0.25, 0.50, 0.75, 1.00]))
    x_eval = np.linspace(0, 1, 101)
    y_eval = result.eval(x=x_eval)
    p = result.params['p'].value

    return p
