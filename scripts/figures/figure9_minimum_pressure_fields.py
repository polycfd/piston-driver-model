import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
import gc

from src.parameters import A_p, n
from src import mass_spring_model as ms
from src import parameters as param
from src import plotting
from scripts.run_parametric_study import T_10, T_50, T_100
from scripts.run_parametric_study import sol_10bar_water_no_piston, sol_50bar_water_no_piston, sol_100bar_water_no_piston, x_fluid_eq_10bar_water_no_piston, x_fluid_eq_50bar_water_no_piston, x_fluid_eq_100bar_water_no_piston

# ============================================================
# GENERATION OF FIGURE 9
# ============================================================

def compute_p_field_full(sol_y, dx_eq, n):
    N_t     = sol_y.shape[1]
    x_p     = sol_y[0, :]
    x_fluid = sol_y[1:n+1, :]

    dx    = np.zeros((n, N_t))
    dx[0] = x_fluid[0] - x_p
    for i in range(1, n):
        dx[i] = x_fluid[i] - x_fluid[i - 1]

    p_field = np.zeros((n, N_t))
    for i in range(n):
        p_field[i] = (
            param.p_0
            - param.rho_l_water * param.c_water**2
            * (dx[i] * A_p - dx_eq[i] * A_p)
            / (dx_eq[i] * A_p)
        )
    return p_field


def find_min_ptop_between_first_two_peaks(p_top_t, prominence_frac=0.3):
    peaks_top, _ = find_peaks(p_top_t, prominence=prominence_frac * np.max(p_top_t))

    if len(peaks_top) >= 2:
        idx_peak1, idx_peak2 = peaks_top[0], peaks_top[1]
    elif len(peaks_top) == 1:
        idx_peak1 = peaks_top[0]
        idx_peak2 = len(p_top_t) - 1
        print("only one peak, fallback idx_peak2 = end of the signal")
    else:
        idx_peak1 = np.argmax(p_top_t)
        idx_peak2 = len(p_top_t) - 1
        print("no peak, fallback argmax")

    idx_min_ptop = idx_peak1 + np.argmin(p_top_t[idx_peak1:idx_peak2])
    return idx_min_ptop, idx_peak1, idx_peak2


driving_pressures = [10e5, 50e5, 100e5]
labels_pressure   = ['10 bar', '50 bar', '100 bar']
T_pressure = {10e5: T_10, 50e5: T_50, 100e5: T_100}

mass_ratios  = np.linspace(0.0, 2.0, 101)
matrix_worst = {}

sol_no_piston_by_pressure = {
    10e5:  sol_10bar_water_no_piston,
    50e5:  sol_50bar_water_no_piston,
    100e5: sol_100bar_water_no_piston,
}

x_fluid_eq_no_piston_by_pressure = {
    10e5:  x_fluid_eq_10bar_water_no_piston,
    50e5:  x_fluid_eq_50bar_water_no_piston,
    100e5: x_fluid_eq_100bar_water_no_piston,
}


for p_p0, label in zip(driving_pressures, labels_pressure):
    T_sim = T_pressure[p_p0]
    p_snapshot_matrix = []

    for M in mass_ratios:
        if M < 1e-9:
            # Case M=0
            sol_np = sol_no_piston_by_pressure[p_p0]
            x_eq_np = x_fluid_eq_no_piston_by_pressure[p_p0]
            N_t    = sol_np.y.shape[1]

            x_eq_np     = sol_np.y[:n+1, 0]
            dx_eq_np    = np.zeros(n)
            dx_eq_np[0] = x_eq_np[1] - x_eq_np[0]
            for i in range(1, n):
                dx_eq_np[i] = x_eq_np[i] - x_eq_np[i-1]

            p_field = compute_p_field_full(sol_np.y, dx_eq_np, n)
            V_t0= param.A_p * (param.H_no_piston - x_eq_np[param.n])
            p_top_t = np.array([ms.pressure_on_top_no_piston(sol_np.y[:, k], V_t0) for k in range(N_t)],)

            _, idx_peak1, idx_peak2 = find_min_ptop_between_first_two_peaks(p_top_t)
            del p_top_t

            p_window = p_field[:, idx_peak1:idx_peak2]
            idx_min  = np.argmin(p_window.min(axis=0))
            p_snapshot_matrix.append(p_window[:, idx_min])

            print(f"{label} — M = 0.00"
                  f"  |  idx_peak1 = {idx_peak1} ({sol_np.t[idx_peak1]*1e3:.3f} ms)"
                  f"  |  idx_peak2 = {idx_peak2} ({sol_np.t[idx_peak2]*1e3:.3f} ms)"
                  f"  |  idx_min = {idx_peak1 + idx_min} ({sol_np.t[idx_peak1 + idx_min]*1e3:.3f} ms)")
            del p_field, p_window
            gc.collect()
            continue

        # Cases M>0
        th_piston = M * param.h * param.rho_water / param.rho_aluminium
        m_piston  = th_piston * A_p * param.rho_aluminium
        H         = param.h + th_piston + param.h_gt + param.x_air_bottom

        x_eq = ms.equilibrate_fluid_positions(
            n, param.L0, param.k_s_water, param.m_e_water, m_piston, param.g, param.p_t0, p_p0
        )
        offset = x_eq[0] - (param.x_air_bottom + th_piston)
        x_eq  -= offset

        dx_eq    = np.zeros(n)
        dx_eq[0] = x_eq[1] - x_eq[0]
        for i in range(1, n):
            dx_eq[i] = x_eq[i] - x_eq[i - 1]

        V_p0_sim = A_p * (x_eq[0] - th_piston)
        V_t0_sim = A_p * (H - x_eq[n])
        y0       = np.concatenate((x_eq, [0.0], np.zeros(n)))

        sol = solve_ivp(
            ms.system_water, [0, T_sim], y0,
            args=(m_piston, H, V_t0_sim, V_p0_sim, param.k_s_water, param.m_e_water, n, p_p0),
            atol=1e-6, rtol=1e-6, max_step=1e-5
        )

        N_t = sol.y.shape[1]
        p_field = compute_p_field_full(sol.y, dx_eq, n)
        p_top_t = np.array([ms.pressure_on_top(sol.y[:, k], H, th_piston, V_t0_sim, n) for k in range(N_t)])

        _, idx_peak1, idx_peak2 = find_min_ptop_between_first_two_peaks(p_top_t)
        del p_top_t

        p_window   = p_field[:, idx_peak1:idx_peak2]
        idx_min    = np.argmin(p_window.min(axis=0))
        p_snapshot_matrix.append(p_window[:, idx_min])

        print(f"{label} — M = {M:.2f}"
              f"  |  idx_peak1 = {idx_peak1} ({sol.t[idx_peak1]*1e3:.3f} ms)"
              f"  |  idx_peak2 = {idx_peak2} ({sol.t[idx_peak2]*1e3:.3f} ms)"
              f"  |  idx_min = {idx_peak1 + idx_min} ({sol.t[idx_peak1 + idx_min]*1e3:.3f} ms)")

        del sol, p_field, p_window
        gc.collect()

    matrix_worst[p_p0] = np.array(p_snapshot_matrix)  # shape (n_M, n)

for p_p0, label in zip(driving_pressures, labels_pressure):
    P_str      = str(int(p_p0/1e5))
    vmin_local = matrix_worst[p_p0].min() / 1e6

    fig, ax = plt.subplots(figsize=(18, 10))
    im = ax.pcolormesh(
        mass_ratios, np.linspace(0, 1, n), matrix_worst[p_p0].T / 1e6,
        cmap='Blues_r', vmin=vmin_local, vmax=0.0, shading='auto'
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Min. pressure [MPa]', fontsize=35, fontweight='bold')
    plt.setp(cbar.ax.get_yticklabels(), fontweight='bold', fontsize=35)

    ax.set_xlabel('M', fontsize=35, fontweight='bold')
    ax.set_ylabel('x/L', fontsize=35, fontweight='bold')
    plotting.style_axes(ax)

    plt.tight_layout()
    plotting.savefig(fig, f'figure9_minimum_pressure_fields_{P_str}bar')
    plt.show()