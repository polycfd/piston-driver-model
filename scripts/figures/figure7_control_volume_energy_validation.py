import matplotlib.pyplot as plt
from scripts import data_energy as data_en
from scripts.run_parametric_study import (
    sol_10bar_water_no_piston,
    sol_10bar_water_09,
    sol_100bar_water_no_piston,
    sol_100bar_water_09,
)
from src import plotting

# ============================================================
# GENERATION OF FIGURE 7 
# ============================================================

panels = [
    dict(
        sol=sol_10bar_water_no_piston,
        e_flux=data_en.e_ac_sig_water_10bar_no_piston,
        e_vol=data_en.e_ac_water_10bar_no_piston,
        xlim=(0, 20),
        title='(a) 1 MPa — M = 0',
    ),
    dict(
        sol=sol_10bar_water_09,
        e_flux=data_en.e_ac_sig_water_10bar_09,
        e_vol=data_en.e_ac_water_10bar_M09,
        xlim=(0, 28),
        title='(b) 1 MPa — M = 0.9',
    ),
    dict(
        sol=sol_100bar_water_no_piston,
        e_flux=data_en.e_ac_sig_water_100bar_no_piston,
        e_vol=data_en.e_ac_water_100bar_no_piston,
        xlim=(0, 6),
        title='(c) 10 MPa — M = 0',
    ),
    dict(
        sol=sol_100bar_water_09,
        e_flux=data_en.e_ac_sig_water_100bar_09,
        e_vol=data_en.e_ac_water_100bar_M09,
        xlim=(0, 9),
        title='(d) 10 MPa — M = 0.9',
    ),
]

fig, axes = plt.subplots(2, 2, figsize=(22, 16))

for ax, panel in zip(axes.flat, panels):
    t_ms = panel['sol'].t * 1e3

    ax.plot(t_ms, panel['e_flux'], label=r'$e_{\text{cv}}$ (Flux balance)', lw=5)
    ax.plot(t_ms, panel['e_vol'], label=r'$e_{\text{cv}}$ (Volume integral)', lw=3.5)

    ax.set_xlim(*panel['xlim'])
    ax.set_title(panel['title'], fontsize=35, fontweight='bold')
    ax.set_xlabel('Time [ms]', fontsize=35, fontweight='bold')
    ax.set_ylabel('Control-volume energy\nper unit area [J/m²]', fontsize=35, fontweight='bold')

    plotting.style_axes(ax, tick_labelsize=35)
    plotting.style_grid(ax, linestyle='--', alpha=0.6)

    legend = ax.legend(loc='best', fontsize=35)
    plotting.style_legend(legend, fontsize=35)

plt.tight_layout()
plotting.savefig(fig, 'figure7_control_volume_energy_validation')
plt.show()