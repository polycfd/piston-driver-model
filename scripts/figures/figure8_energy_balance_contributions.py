import matplotlib.pyplot as plt
import numpy as np
from scripts import data_energy as data_en
from src import plotting

# ============================================================
# GENERATION OF FIGURE 8
# ============================================================

configs = ['M=0', 'M=0.1', 'M=0.5', 'M=1', 'M=1.5', 'M=2']
x = np.arange(len(configs))

data = {
    100: {
        'e_ac':   data_en.E_ac_sig_water_over_E_ref_100bar_from_peak_to_peak_different_M,
        'e_in':   data_en.E_in_water_over_E_ref_100bar_from_peak_to_peak_different_M,
        'e_out':  data_en.E_out_water_over_E_ref_100bar_from_peak_to_peak_different_M,
        'e_pist': data_en.E_piston_water_over_E_ref_100bar_from_peak_to_peak_different_M,
    },
}

for P in data:
    e_ac   = np.array(data[P]['e_ac'])   * 100
    e_in   = np.array(data[P]['e_in'])   * 100
    e_out  = np.array(data[P]['e_out'])  * 100
    e_pist = np.array(data[P]['e_pist']) * 100

    refl          = np.where(e_out <= 0, np.abs(e_out), 0)
    leak          = np.where(e_out > 0, e_out, 0)
    piston_source = np.where(e_pist <= 0, np.abs(e_pist), 0)
    piston_cost   = np.where(e_pist > 0, e_pist, 0)

    fig, ax = plt.subplots(figsize=(18, 10))
    w = 0.5

    ax.bar(x, e_in, w, label=r'$E^*_{\text{lower}}$', color='#C0392B', alpha=0.88, zorder=3)
    ax.bar(x, refl, w, bottom=e_in, label=r'$E^*_{\text{return}}$', color='#D4AC0D', alpha=0.85, zorder=3)
    ax.bar(x, piston_source, w, bottom=e_in + refl, label=r'$E^*_{\text{piston}}$', color='#2980B9', alpha=0.85, zorder=3)

    ax.plot(x, e_ac, 'o--', color='black', lw=4, ms=20, label=r'$E^*_{\text{cv}}$', zorder=5)

    for i, val in enumerate(e_ac):
        ax.annotate(f'{val:.3f}', xy=(x[i], val), xytext=(0, 12),
                    textcoords='offset points', ha='center',
                    fontsize=25, fontweight='bold', color='black')

    ax.axhline(0, color='black', linewidth=2, zorder=4)
    ax.set_ylabel(r'$\langle E^* \rangle$ between' + '\ncompression peaks [%]',
                  fontsize=35, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=35, fontweight='bold')

    plotting.style_axes(ax)
    plotting.style_grid(ax, linestyle='--', alpha=0.6)

    legend = ax.legend(loc='best', fontsize=35, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    plotting.style_legend(legend)

    plt.tight_layout()
    plotting.savefig(fig, 'figure8_energy_balance_contributions')
    plt.show()