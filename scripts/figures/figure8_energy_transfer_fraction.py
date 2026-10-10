import matplotlib.pyplot as plt
from src.parameters import M_list, colors
from scripts import data_energy as data_en
from src import plotting

# ============================================================
# GENERATION OF FIGURE 8
# ============================================================

plotting.new_figure(figsize=(18, 10))

pressures_bar = [10, 20, 30, 40, 50, 60, 75, 100]
labels_MPa    = ['1MPa', '2MPa', '3MPa', '4MPa', '5MPa', '6MPa', '7.5MPa', '10MPa']

for i, (p, label) in enumerate(zip(pressures_bar, labels_MPa)):
    attr_name = f'E_ac_sig_water_over_E_ref_{p}bar_from_peak_to_peak_different_M'
    values = getattr(data_en, attr_name) * 100
    plt.scatter(M_list, values, s=350, color=colors[i], label=label)

plt.xlabel('M', fontsize=35, fontweight='bold')
plt.ylabel(r'$\langle E_{\text{cv}}^* \rangle$ between' + '\ncompression peaks [%]',
           fontsize=35, fontweight='bold')
plt.yscale('log')

ax = plt.gca()
plotting.style_axes(ax)
plotting.style_grid(ax, linestyle='--', alpha=0.6)

legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=35)
plotting.style_legend(legend)

plt.tight_layout()
plotting.savefig(plt.gcf(), 'figure8_energy_transfer_fraction')
plt.show()