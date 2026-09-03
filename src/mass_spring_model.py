import numpy as np
from src import parameters as params

# ============================================================
# ODE SOLVER FOR THE MASS-SPRING MODEL 
# ============================================================

#Initial positions#
def equilibrate_fluid_positions(n, L0, k_s, m_e, m_piston, g, p_t, p_p):
    K = np.zeros((n+1, n+1))
    K[0, 0] = k_s
    for i in range(1, n+1):
        K[i, i] = 2*k_s
    for i in range(n):
        K[i, i+1] = -k_s
        K[i+1, i] = -k_s
    K[n, n] = k_s
    K[0, :] = 0  # Dirichlet
    K[0, 0] = 1
    F = np.ones(n+1)*(-m_e*g)
    th_piston = m_piston/(params.A_p*params.rho_aluminium)
    F[0] = th_piston + 1e-6
    F[n] = -p_t*params.A_p - m_e*g + k_s*L0
    x_fluid_eq = np.linalg.solve(K, F)
    return x_fluid_eq


def equilibrate_fluid_positions_no_piston(n, L0, k_s, m_e, g, p_t, p_p):
    K = np.zeros((n+1, n+1))
    K[0, 0] = k_s
    for i in range(1, n+1):
        K[i, i] = 2*k_s
    for i in range(n):
        K[i, i+1] = -k_s
        K[i+1, i] = -k_s
    K[n, n] = k_s
    K[0, :] = 0  # Dirichlet
    K[0, 0] = 1
    F = np.ones(n+1)*(-m_e*g)
    F[0] = params.x_air_bottom
    F[n] = -p_t*params.A_p - m_e*g + k_s*L0
    x_fluid_eq = np.linalg.solve(K, F)
    return x_fluid_eq


#Volumes#
def volumes(x, H, th_piston, n):
    x_p = x[0]  # piston top position
    x_n = x[n]
    x_t = H - x_n  # liquid surface position
    V_p = params.A_p * (x_p - th_piston)  # lower gas volume
    V_t = params.A_p * x_t  # upper gas volume
    return V_p, V_t


def volumes_no_piston(x):
    x_0 = x[0]  # first liquid element position
    x_n = x[params.n]
    x_t = params.H_no_piston - x_n  # liquid surface position
    V_p = params.A_p * x_0  # lower gas volume
    V_t = params.A_p * x_t  # upper gas volume
    return V_p, V_t


#Upper gas pressure#
def pressure_on_top(x, H, th_piston, V_t0, n):
    V_p, V_t = volumes(x, H, th_piston, n)
    p_t = params.p_t0 * (V_t0 / V_t)**params.gamma
    return p_t


def pressure_on_top_no_piston(x, V_t0):
    V_p, V_t = volumes_no_piston(x)
    p_t = params.p_t0 * (V_t0 / V_t)**params.gamma
    return p_t


#Lower gas pressure#
def pressure_under_piston(x, H, th_piston, V_p0, n, p_p0):
    V_p, V_t = volumes(x, H, th_piston, n)
    p_p = p_p0 * (V_p0 / V_p)**params.gamma
    return p_p


def pressure_gas_bottom(x, V_p0, p_p0):
    V_p, V_t = volumes_no_piston(x)
    p_p = p_p0 * (V_p0 / V_p)**params.gamma
    return p_p


#System#
def system_water(t, y, m_piston, H, V_t0, V_p0, k_s_water, m_e_water, n, p_p0):
    # y = [x_p, x1, ..., xn, v_p, v1, ..., vn]
    x = y[:n+1]  # Positions [x_p, x1, ..., xn]
    v = y[n+1:2*(n+1)]  # Velocity [v_p, v1, ..., vn]
    th_piston = m_piston/(params.A_p*params.rho_aluminium)
    p_t = pressure_on_top(y, H, th_piston, V_t0, n)
    p_p = pressure_under_piston(y, H, th_piston, V_p0, n, p_p0)
    a = np.zeros(n+1)
    a[0] = (p_p * params.A_p + k_s_water * (x[1] - x[0] - params.L0) - m_piston * params.g) / m_piston
    for i in range(1, n):
        a[i] = (-k_s_water * (x[i] - x[i-1] - params.L0) + k_s_water * (x[i+1] - x[i] - params.L0) - m_e_water * params.g) / m_e_water
    a[n] = (-p_t * params.A_p - k_s_water * (x[n] - x[n-1] - params.L0) - m_e_water * params.g) / m_e_water

    return np.concatenate((v, a))


def system_water_no_piston(t, y, V_t0, V_p0, p_p0):
    x = y[:params.n+1]
    v = y[params.n+1:]
    a = np.zeros(params.n+1)
    V_p, V_t = volumes_no_piston(y)
    p_t = pressure_on_top_no_piston(y, V_t0)
    p_p = pressure_gas_bottom(y, V_p0, p_p0)
    a[0] = (p_p * params.A_p + params.k_s_water * (x[1] - x[0] - params.L0) - params.m_e_water * params.g) / params.m_e_water
    for i in range(1, params.n):
        a[i] = (-params.k_s_water * (x[i] - x[i-1] - params.L0) + params.k_s_water * (x[i+1] - x[i] - params.L0) - params.m_e_water * params.g) / params.m_e_water
    a[params.n] = (-params.k_s_water * (x[params.n] - x[params.n-1] - params.L0) - p_t * params.A_p - params.m_e_water * params.g) / params.m_e_water
    return np.concatenate((v, a))


def system_lith(t, y, m_piston, H, V_t0, V_p0, n, p_p0):
    x = y[:n+1]
    v = y[n+1:2*(n+1)]
    th_piston = m_piston/(params.A_p*params.rho_aluminium)
    p_t = pressure_on_top(y, H, th_piston, V_t0, n)
    p_p = pressure_under_piston(y, H, th_piston, V_p0, n, p_p0)
    a = np.zeros(n+1)
    a[0] = (p_p * params.A_p + params.k_s_lithium * (x[1] - x[0] - params.L0) - m_piston * params.g) / m_piston
    for i in range(1, n):
        a[i] = (-params.k_s_lithium * (x[i] - x[i-1] - params.L0) + params.k_s_lithium * (x[i+1] - x[i] - params.L0) - params.m_e_lithium * params.g) / params.m_e_lithium
    a[n] = (-p_t * params.A_p - params.k_s_lithium * (x[n] - x[n-1] - params.L0) - params.m_e_lithium * params.g) / params.m_e_lithium

    return np.concatenate((v, a))


#Piston acceleration#
def piston_acceleration_water(y, m_piston, H, V_p0, n, k_s_water, p_p0):
    x = y[:n+1]
    v = y[n+1:2*(n+1)]
    th_piston = m_piston/(params.A_p*params.rho_aluminium)
    p_p = pressure_under_piston(y, H, th_piston, V_p0, n, p_p0)
    a0 = (p_p * params.A_p + k_s_water * (x[1] - x[0] - params.L0) - m_piston * params.g) / m_piston
    return a0

def piston_acceleration_lith(y, m_piston, V_p0, p_p0):
    x = y[:params.n+1]
    v = y[params.n+1:2*(params.n+1)]
    p_p = pressure_under_piston(y, params.H_ref, params.th_piston_ref, V_p0, params.n, p_p0)
    a0 = (p_p * params.A_p + params.k_s_lithium * (x[1] - x[0] - params.L0) - m_piston * params.g) / m_piston
    return a0

#First liquid element acceleration#
def liquid_acceleration_water(y, n, k_s_water, m_e_water, L0, V_p0, p_p0):
    x = y[:n+1]
    v = y[n+1:2*(n+1)]
    p_p = pressure_gas_bottom(y, V_p0, p_p0)
    a0 = (p_p * params.A_p + k_s_water * (x[1] - x[0] - L0) - m_e_water * params.g) / m_e_water
    return a0


def liquid_acceleration_lithium(y, n, k_s_lithium, m_e_lithium, L0, V_p0, p_p0):
    x = y[:n+1]
    v = y[n+1:2*(n+1)]
    p_p = pressure_gas_bottom(y, V_p0, p_p0)
    a0 = (p_p * params.A_p + k_s_lithium * (x[1] - x[0] - L0) - m_e_lithium * params.g) / m_e_lithium
    return a0

#Liquid acceleration#
def all_accelerations_water(y, m_piston, H, V_t0, V_p0, k_s_water, m_e_water, n, p_p0):
    x = y[:n+1]
    v = y[n+1:2*(n+1)]
    th_piston = m_piston / (params.A_p * params.rho_aluminium)
    p_t = pressure_on_top(y, H, th_piston, V_t0, n)
    p_p = pressure_under_piston(y, H, th_piston, V_p0, n, p_p0)
    a = np.zeros(n+1)
    a[0] = (p_p * params.A_p + k_s_water * (x[1] - x[0] - params.L0) - m_piston * params.g) / m_piston
    for i in range(1, n):
        a[i] = (-k_s_water * (x[i] - x[i-1] - params.L0) + k_s_water * (x[i+1] - x[i] - params.L0) - m_e_water * params.g) / m_e_water
    a[n] = (-p_t * params.A_p - k_s_water * (x[n] - x[n-1] - params.L0) - m_e_water * params.g) / m_e_water
    return a

def all_accelerations_water_no_piston(y, V_t0, V_p0, p_p0):
    x = y[:params.n+1]
    v = y[params.n+1:]
    a = np.zeros(params.n+1)
    p_t = pressure_on_top_no_piston(y, V_t0)
    p_p = pressure_gas_bottom(y, V_p0, p_p0)
    a[0] = (p_p * params.A_p + params.k_s_water * (x[1] - x[0] - params.L0) - params.m_e_water * params.g) / params.m_e_water
    for i in range(1, params.n):
        a[i] = (-params.k_s_water * (x[i] - x[i-1] - params.L0) + params.k_s_water * (x[i+1] - x[i] - params.L0) - params.m_e_water * params.g) / params.m_e_water
    a[params.n] = (-params.k_s_water * (x[params.n] - x[params.n-1] - params.L0) - p_t * params.A_p - params.m_e_water * params.g) / params.m_e_water
    return a 