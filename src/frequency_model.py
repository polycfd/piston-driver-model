import numpy as np
from scipy.optimize import root_scalar

# ============================================================
# ANALYTICAL FREQUENCY MODEL
# ============================================================

#Parameters#
c_water = 1500  # [m/s]
c_lith = 4544  # [m/s]
L_small = 0.05  # [m] - liquid length
L = 0.15  # [m] - liquid length
gamma=1.4
rho_water=1000

#data#
mass_ratios_ms_small_column = [0, 0.1, 0.2, 0.5, 1, 2, 5, 7, 8, 9, 10] 
mass_ratios_ms_usual_pda = [0,0.1, 0.2, 0.5, 1, 2, 3]
M_data_ms_small_column = np.array(mass_ratios_ms_small_column)
M_data_ms_usual_pda = np.array(mass_ratios_ms_usual_pda)

#MS model#
#water#
#small liquid column#
frequencies_ms_water_small_column=[14550, 13663, 12650, 10862, 9600, 8675,7950, 7800, 7750, 7712, 7675] #frequencies computed with the FFT in signal_processing
beta_data_ms_water_small_column = []
frequencies_ms_water_small_column=np.array(frequencies_ms_water_small_column)
beta_data_ms_water_small_column = c_water/(frequencies_ms_water_small_column * L_small)

#usual pda geometry#
frequencies_ms_water_usual_pda=[4962, 4550, 4237, 3650, 3225, 2912, 2787]
beta_data_ms_water_usual_pda = []
frequencies_ms_water_usual_pda=np.array(frequencies_ms_water_usual_pda)
beta_data_ms_water_usual_pda = c_water/(frequencies_ms_water_usual_pda * L)

#lithium#
#small liquid column#
frequencies_ms_lithium_small_column=[44089,41364, 38276, 32888, 29051, 26251, 24088, 23625, 23475, 23363,23263] 
beta_data_ms_lithium_small_column = []
frequencies_ms_lithium_small_column=np.array(frequencies_ms_lithium_small_column)
beta_data_ms_lithium_small_column = c_lith/(frequencies_ms_lithium_small_column * L_small)
#usual pda geometry#
frequencies_ms_lithium_usual_pda=[15000, 13813, 12788, 11012, 9750, 8825, 8437]
beta_data_ms_lithium_usual_pda = []
frequencies_ms_lithium_usual_pda=np.array(frequencies_ms_lithium_usual_pda)
beta_data_ms_lithium_usual_pda = c_lith/(frequencies_ms_lithium_usual_pda * L)

#def equation to solve#
def g(beta, M):
    X = 2 * np.pi / beta
    if np.isclose(beta, 0.0) or np.isclose(np.cos(X), 0.0):
        return np.inf
    return np.tan(X) + M * X - 1/X


beta_bracket = [2.01, 3.99] #solution search range

M_unique_ms_small_column = np.sort(np.unique(M_data_ms_small_column)) #Sort M so that the system is solved once for each value of M
M_unique_ms_usual_pda = np.sort(np.unique(M_data_ms_usual_pda))

beta_analytical_results = []
M_analytical_plot = []

M_many_points = np.linspace(0, 10, 100)

#solve with bisection method#
for M in M_many_points:
    if np.isclose(M, 0.0):     #if there is no piston, beta=2#   
        beta_analytical_results.append(2.0)        
        M_analytical_plot.append(M)
    try:
        sol = root_scalar(g, args=(M,), bracket=beta_bracket, method='bisect')       
        if sol.converged:
            beta_analytical_results.append(sol.root)
            M_analytical_plot.append(M)
    except ValueError:
        continue