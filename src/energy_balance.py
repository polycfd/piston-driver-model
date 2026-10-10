import matplotlib.pyplot as plt
import numpy as np
from src import parameters as params

# ============================================================
# ENERGY BALANCE FOR THE TOTAL SYSTEM AND IN THE CONTROL VOLUME 
# ============================================================


#Total system energy balance#
def compute_energy_balance(t, p_t, V_t, p_b, V_b, m_piston, u_piston, v_liq, x_piston, x_liq):
    E_gas_top    = np.array(p_t) * np.array(V_t) / (params.gamma - 1)
    E_gas_bottom = np.array(p_b) * np.array(V_b) / (params.gamma - 1)
    E_kinetic = 0.5 * m_piston * np.array(u_piston)**2 + np.sum(0.5 * params.m_e_water * v_liq**2, axis=0)
    E_el_1 = 0.5 * params.k_s_water * (x_liq[0,:] - x_piston - params.L0)**2
    E_el_inter = np.sum(0.5 * params.k_s_water * (np.diff(x_liq, axis=0) - params.L0)**2, axis=0)
    E_elastic_liq = E_el_1 + E_el_inter
    E_total = E_gas_top + E_gas_bottom + E_kinetic +E_elastic_liq
    E_gas_top_normalized=E_gas_top/E_total[0]
    E_gas_bottom_normalized=E_gas_bottom/E_total[0]
    E_kinetic_normalized=E_kinetic/E_total[0]
    E_elastic_liq_normalized=E_elastic_liq/E_total[0]
    E_total_normalized=E_total/E_total[0]
    return t, E_total

def compute_energy_balance_no_piston(t, p_t, V_t, p_b, V_b, v_liq, x_0, x_liq):
    E_gas_top    = np.array(p_t) * np.array(V_t) / (params.gamma - 1)
    E_gas_bottom = np.array(p_b) * np.array(V_b) / (params.gamma - 1)
    E_kinetic = np.sum(0.5 * params.m_e_water * v_liq**2, axis=0)
    E_el_1 = 0.5 * params.k_s_water * (x_liq[0,:] - x_0 - params.L0)**2
    E_el_inter = np.sum(0.5 * params.k_s_water * (np.diff(x_liq, axis=0) - params.L0)**2, axis=0)   
    E_elastic_liq = E_el_1 + E_el_inter

    E_total = E_gas_top + E_gas_bottom + E_kinetic+E_elastic_liq
    return t, E_total

#Compute control volume energy with volume integral method#
def compute_acoustic_energy(p_array, p_0, u_array, rho, c, L):
    n_space = p_array.shape[0]
    if n_space == 0:
        return np.array([0.0])
    Delta_x = L / n_space
    p_array_acoustic = p_array - p_0
    ep_density = (p_array_acoustic**2) / (2 * rho * c**2) 

    u_array = u_array[1:]
    u_mean = np.mean(u_array, axis=0)
    u_fluctu = u_array - u_mean [None, :]
    ec_density = (rho * u_fluctu**2) / 2 
    
    # Integration  
    energy_density_sum = np.sum(ep_density + ec_density, axis=0)
    e_ac_t = Delta_x * energy_density_sum
    
    return e_ac_t, u_fluctu


#Compute control volume energy with energy balance#
def cumulative_trapezoid(y, t):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if y.shape != t.shape:
        raise ValueError(f"Shapes incompatibles: y{y.shape} vs t{t.shape}")
    if len(t) < 2:
        return np.zeros_like(t)
    dt = np.diff(t)
    incr = 0.5 * (y[1:] + y[:-1]) * dt
    I = np.zeros_like(y)
    I[1:] = np.cumsum(incr)
    return I

def compute_e_ac_from_signals(t, u_piston, u_out, p_bottom, p_out, v_liq, rho_pist, L_pist, a_piston):
    u_bar = np.mean(v_liq, axis=0)
    a_piston=np.array(a_piston)
    u_piston=np.array(u_piston)
    u_out=np.array(u_out)
    p_bottom = np.array(p_bottom) 
    p_out    = np.array(p_out)
    
    p_ac_bottom = p_bottom - params.p_0
    p_ac_out    = p_out - params.p_0
    power_piston_corr = (rho_pist * L_pist) * a_piston * (u_piston - u_bar)
    
    e_drive_ac = cumulative_trapezoid(p_ac_bottom * (u_piston - u_bar), t)
    e_piston_corr = cumulative_trapezoid(power_piston_corr, t)
    e_out_ac = cumulative_trapezoid(p_ac_out * (u_out - u_bar), t)
    return e_drive_ac - e_out_ac - e_piston_corr, e_drive_ac, e_piston_corr, e_out_ac


def compute_e_ac_from_signals_no_piston(t, u_bottom, u_out, p_bottom, p_out, v_liq):
    u_bar = np.mean(v_liq, axis=0)
    p_bottom = np.array(p_bottom) 
    p_out    = np.array(p_out)
    p_ac_bottom = p_bottom - params.p_0
    p_ac_out    = p_out - params.p_0
    
    u_ac_piston = u_bottom - u_bar 
    u_ac_out    = u_out - u_bar   

    e_drive_ac = cumulative_trapezoid(p_ac_bottom * u_ac_piston, t)
    e_out_ac   = cumulative_trapezoid(p_ac_out * u_ac_out, t)
    
    return e_drive_ac - e_out_ac, e_drive_ac, e_out_ac

#Mean energy between two compression peaks#
def mean_e_from_peak_to_peak(t, e, t_start, t_end):
    T = (t_end-t_start)

    #Find index for interval [t_start, t_start + T]
    idx_start = np.searchsorted(t, t_start)
    idx_end = np.searchsorted(t, t_end)

    t_period = t[idx_start:idx_end]
    e_period = e[idx_start:idx_end]

    integral_e = np.trapezoid(e_period, t_period)

    mean_e = integral_e / T

    return mean_e