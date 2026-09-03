import csv
import numpy as np
import matplotlib.pyplot as plt

from scripts.run_parametric_study import sol_50bar_water_09, p_t_50bar_water_09, p_p_50bar_water_09
from src import plotting


# ============================================================
# GENERATION OF THE MASS-SPRING MODEL VALIDATION FIGURES
# ============================================================

CFD_DIR = "data/cfd"

def read_column(filepath, col_index):
    values = []
    with open(filepath, newline='', encoding='utf-8') as fichier:
        lecteur = csv.reader(fichier)
        for ligne in lecteur:
            values.append(float(ligne[col_index]))
    return np.array(values)


p_t_water_5MPa_M089_CFD = read_column(f"{CFD_DIR}/gastop_5MPa_water_M089.csv", col_index=3)
time_water_5MPa_M089_CFD = read_column(f"{CFD_DIR}/gasbot_5MPa_water_M089.csv", col_index=0)
p_p_water_5MPa_M089_CFD = read_column(f"{CFD_DIR}/gasbot_5MPa_water_M089.csv", col_index=3)

def plot_cfd_vs_ms(t_ms, p_ms, t_cfd, p_cfd, ylabel, filename):
    plotting.new_figure(figsize=(18, 9))

    plt.plot(t_ms, p_ms / 1e6, lw=6, linestyle='--', label='MS')
    plt.plot(t_cfd, p_cfd / 1e6, lw=6, label='CFD')

    plt.xlim(0, 8)
    plt.xlabel('Time [ms]', fontsize=35, fontweight='bold')
    plt.ylabel(ylabel, fontsize=35, fontweight='bold')
    plt.yscale('log')

    ax = plt.gca()
    plotting.style_axes(ax)
    plotting.style_grid(ax, linestyle='--', alpha=0.6)

    legend = plt.legend(loc='best')
    plotting.style_legend(legend)

    plotting.savefig(plt.gcf(), filename)
    plt.show()

plot_cfd_vs_ms(
    t_ms=sol_50bar_water_09.t * 1e3,
    p_ms=p_t_50bar_water_09,
    t_cfd=time_water_5MPa_M089_CFD * 1e3,
    p_cfd=p_t_water_5MPa_M089_CFD,
    ylabel='Upper gas pressure [MPa]',
    filename='p_t_water_5MPa_M089_CFD_vs_ms',
)

plot_cfd_vs_ms(
    t_ms=sol_50bar_water_09.t * 1e3,
    p_ms=p_p_50bar_water_09,
    t_cfd=time_water_5MPa_M089_CFD * 1e3,
    p_cfd=p_p_water_5MPa_M089_CFD,
    ylabel='Lower gas pressure [MPa]',
    filename='p_p_water_5MPa_M089_CFD_vs_ms',
)