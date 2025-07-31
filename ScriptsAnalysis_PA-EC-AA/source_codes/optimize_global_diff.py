# This script contains main functions to optimize the MD global diffusion coefficients of the different concentrations to the experimental diffusion curve.

import numpy as np 
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from lmfit import Model
import pickle
import lmfit
from scipy.interpolate import interp1d
import sys
sys.path.append('source_codes/') 
from helper_functions_plots import *
from scipy.optimize import minimize
from scipy.interpolate import interp1d


def concentration2(Ds_weighted, Ds_weighted_err, T_weighted, initial_weights, x_fit, fit, fit_err, TCD = 323.15):
    # Define objective function
    def objective_function(weights):

        # Weighted sum of MD Diffusion Coefficients (already with a_u weighting the folded and unfolded)
        D_model = weights@Ds_weighted

        # Element-wise multiplication using broadcasting for Error (?)
        weighted_errors = weights[:, np.newaxis] * Ds_weighted_err
        D_model_err = np.linalg.norm(weighted_errors, axis = 0)

        indeces = []
        for t in np.array(T_weighted):
            indx = np.argmin(np.abs(x_fit+ 273.15 - t))
            indeces.append(indx)
        fit_tmp = fit[np.array(indeces)]
        fit_tmp_err = fit_err[np.array(indeces)]
        
        # Calculate model prediction using given weights
        interp_func = interp1d(T_weighted, D_model, kind='linear', bounds_error=False)
        x_fit_md = np.linspace(T_weighted[0], T_weighted[-1], 1000)

        D_interp = interp_func(x_fit + 273.15)
        fig = plt.figure(figsize=(2.5, 1.9), dpi = 100)
        plt.plot(x_fit[:]+273.15, fit[:], '--', lw=2, color='crimson')
        plt.plot(x_fit+273.15, D_interp, '-', markersize = 1, color = 'purple')
        plt.plot(T_weighted, D_model, 'o', color = 'purple')
        plt.show()

        # MINIMIZE FUNCTION
        error = np.sum(((D_model - fit_tmp))** 2)
        return error



    # Bounds for  weights
    bounds = tuple((0, 1) for i in initial_weights)

    # Define equality constraint: sum of weights equals 1
    constraints = ({"type":"eq", "fun": lambda x: np.sum(x)-1})

    # Perform optimization with bounds and equality constraint
    result = minimize(fun = objective_function, x0 = initial_weights, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 5000, 'disp': True})
    optimized_weights = result.x
    

    # Use optimized weights to calculate model prediction
    D_final_optimized = optimized_weights@Ds_weighted
    D_final_err = np.sqrt((optimized_weights[0] * Ds_weighted_err[0])**2 +
                      (optimized_weights[1] * Ds_weighted_err[1])**2 +
                      (optimized_weights[2] * Ds_weighted_err[2])**2 + (optimized_weights[3] * Ds_weighted_err[3])**2)

    gellatio_fit_res = gellation_fit_3steps(T_weighted-273.15, D_final_optimized, D_final_err, xlim=(3,3), t0 = 100,dt = 5.73,
        fit_method='least_square', set_min_to_0=['a2', 'b2', 'a1', 'b1', 't0', 'dt'])

    x_fit_md = np.linspace(T_weighted[0], T_weighted[-1], 1000)-273.15
    fit_md = gellatio_fit_res.eval(x=x_fit_md)
    fit_md_err = gellatio_fit_res.eval_uncertainty(x=x_fit_md)
    
    
    fig = plt.figure(figsize=(2.5, 1.9), dpi = 200)
    plt.plot(x_fit_md+273.15, fit_md, '-', lw=2, color=c1)#
    plt.errorbar(T_weighted, D_final_optimized, D_final_err, fmt = 'o', color = c1, label = 'D$_{opt}$')
    plt.errorbar(all_temp[:], dG[0][:], dG[1][:], fmt = 's', fillstyle = 'none', color = 'black',  label = 'D$^{QENS}$')
    plt.plot(x_fit[2:]+273.15, fit[2:], '-', color='black')
    plt.xlabel('$T$ (K)')
    plt.ylabel('$D_G$ ($\\mathrm{\\AA}^2$/ns)')
    plt.axvline(TCD, 0, 1, color = 'black')
    plt.text(TCD + 2, 0.2, 'T$_{CD}$', rotation=0, fontsize=12, color = 'black')
    plt.ylim((0, 2.5))
    plt.show()
    fig.subplots_adjust(left=0.22, right=0.95, bottom = 0.25, top = 0.95, wspace = 0.35, hspace = 0.7)

    print("Optimized Weights:", optimized_weights)
    return optimized_weights


def gellation(x, t0, dt, a1, b1, a2, b2):
    t_star = (x - t0) / dt
    theta_star = theta(t_star)
    return (a1 * x + b1) * (1 - theta_star) + (a2 * x + b2) * theta_star

def gellation_fit_3steps(x,y, yerr, xlim=(3,4), fit_method='cg', t0=50, dt=4, set_min_to_0=['a1', 'b1', 'a2', 'b2', 't0', 'dt']):
    mod1  = lmfit.models.LinearModel()
    pars1 = mod1.guess(y[:xlim[0]], x=x[:xlim[0]])
    res1  = mod1.fit(y[:xlim[0]], pars1, weights=1/yerr[:xlim[0]], x=x[:xlim[0]])
    mod2  = lmfit.models.LinearModel()
    pars2 = mod2.guess(y[xlim[1]:], x=x[xlim[1]:])
    res2  = mod2.fit(y[xlim[1]:], pars2, weights=1/yerr[xlim[1]:], x=x[xlim[1]:])
    mod3  = lmfit.Model(gellation)
    min_val = 0 if 'a1' in set_min_to_0 else -np.inf
    mod3.set_param_hint(name='a1', min=min_val, value=res1.params['slope'].value)
    min_val = 0 if 'b1' in set_min_to_0 else -np.inf
    mod3.set_param_hint(name='b1', min=min_val, value=res1.params['intercept'].value)
    min_val = 0 if 'a2' in set_min_to_0 else -np.inf
    mod3.set_param_hint(name='a2', min=min_val, value=res2.params['slope'].value)
    min_val = 0 if 'b2' in set_min_to_0 else -np.inf
    mod3.set_param_hint(name='b2', min=min_val, value=res2.params['intercept'].value)
    min_val = 0 if 't0' in set_min_to_0 else -np.inf
    mod3.set_param_hint(name='t0', min=min_val, value=t0)
    min_val = 0 if 'dt' in set_min_to_0 else -np.inf
    mod3.set_param_hint(name='dt', min=min_val, value=dt)
    pars3 = mod3.make_params()
    res3  = mod3.fit(y, pars3, x=x, weights=1/yerr, method=fit_method)
    return res3
