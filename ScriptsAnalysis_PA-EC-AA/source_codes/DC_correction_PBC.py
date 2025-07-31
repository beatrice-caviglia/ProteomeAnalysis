import numpy as np

def total_mass(file):
    with open(file, 'r') as file:
        lines = file.readlines() 
    total_weight_mass = 0
    num_proteins = 0
    for line in lines:
        tokens = line.split()
        if len(tokens) >= 3:
            weight_mass = int(tokens[3].replace(',', ''))
            total_weight_mass += weight_mass
            num_proteins += 1
    return total_weight_mass

def average_weight_mass(file):
    with open(file, 'r') as file:
        lines = file.readlines()  
    total_weight_mass = 0
    num_proteins = 0
    
    # Iterate through each line and extract weight mass
    for line in lines:
        tokens = line.split()
        if len(tokens) >= 3:
            weight_mass = int(tokens[3].replace(',', ''))  
            total_weight_mass += weight_mass
            num_proteins += 1
    # Calculate the average weight mass
    average_weight_mass = total_weight_mass / num_proteins
    return average_weight_mass


def get_alpha(average_mass, T):
    file_path = 'source_codes/water_densities_CoolProp.txt'
    temperatures, densities = read_density_data(file_path)
    rho = interpolate_density(T, temperatures, densities)
    alpha = rho*(average_mass/1000)/0.0180153
    return alpha


def get_concetration(total_mass, boxsize = 17):
    conc = ((total_mass/1000)/(boxsize**3))*(10**24)*1.6605*10**(-21)
    return conc

def read_density_data(file_path):
    temperatures = []
    densities = []
    with open(file_path, 'r') as file:
        for line in file:
            if line and not line.startswith(';'): 
                temp, density = map(float, line.split())
                temperatures.append(temp)
                densities.append(density)
    return np.array(temperatures), np.array(densities)


def interpolate_density(T_kelvin, temperatures, densities):
    # Convert Kelvin to Celsius
    T_celsius = T_kelvin - 273.15
    # Interpolate density
    return np.interp(T_celsius, temperatures, densities)

def get_correction_crowded(L, T, file_mass, visc_file):
    # L is the edge length of the cubic box
    # T is the temperature in Kelvin
    # file_mass is a file containing the protein masses
    # visc_file contains the array of the viscosity parameters
    
    kb = 1.380649e-23
    xi = 2.837297    

    avg_mass = average_weight_mass(file_mass)
    alpha = get_alpha(avg_mass, T)
    tot_mass = total_mass(file_mass)
    conc = get_concetration(tot_mass, boxsize = L) # conc is protein concentration (in g/L)
    v = 1.417e-3                                   # m^3/kg             
    
    pars = np.load(visc_file)
    Bt = pars[0]
    Dt = pars[1]
    dEt = pars[2]

    Bw = 14.75   # [-]
    Dw = 0.0054  # K-1
    dEw = 12.8   # kJ/mol
    
    RT = 8.3144598e-3*T # kJ/mol
    beta = alpha*v - 1

    eta_0 = np.exp(-Bw + Dw*T + dEw/(RT))
    eta_r = np.exp((conc/(alpha-(beta*conc)))*(-Bt + Dt*T + (dEt/(RT))))
    eta = eta_0*eta_r # Multiply by the viscosity of the solvent
    
    return (kb*T*xi)/(6*np.pi*eta*L)*1e20

def get_correction_crowded_rot(L, T, file_avg_mass, visc_file):
    # L is the edge length of the cubic box
    # T is the temperature in Kelvin
    # file_mass is a file containing the protein masses
    # visc_file contains the array of the viscosity parameters
    
    kb = 1.380649e-23
    xi = 2.837297

    avg_mass = average_weight_mass(file_avg_mass)
    alpha = get_alpha(avg_mass, T)
    tot_mass = total_mass(file_avg_mass)
    conc = get_concetration(tot_mass, boxsize = L)
    v = 1.417e-3     # m^3/kg    
    
    pars = np.load(visc_file)
    Bt = pars[0]
    Dt = pars[1]
    dEt = pars[2]
    
    Bw = 14.75   # [-]
    Dw = 0.0054  # K-1
    dEw = 12.8   # kJ/mol
    

    RT = 8.3144598e-3*T # kJ/mol
    beta = alpha*v - 1

    eta_0 = np.exp(-Bw + Dw*T + dEw/(RT))
    eta_r = np.exp((conc/(alpha-(beta*conc)))*(-Bt + Dt*T + (dEt/(RT))))
    eta = eta_0*eta_r # Multiply by the viscosity of the solvent
        
    return ((kb*T)/(6*eta*(L**3)))*1e18
    
