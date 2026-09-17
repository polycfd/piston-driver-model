import matplotlib.pyplot as plt
from src import frequency_model as freq_model
from src import plotting

# ============================================================
# GENERATION OF FIGURE 6
# ============================================================

plotting.new_figure(figsize=(18, 8))

plt.scatter(freq_model.M_data_ms_usual_pda, freq_model.beta_data_ms_water_usual_pda,
            facecolors='none', edgecolors='blue', label='Water — L = 0.15 m',
            s=250, marker='o', linewidths=3.5)
plt.scatter(freq_model.M_data_ms_usual_pda, freq_model.beta_data_ms_lithium_usual_pda,
            facecolors='none', edgecolors='darkgreen', label='Lithium — L = 0.15 m',
            s=250, marker='o', linewidths=3.5)
plt.scatter(freq_model.M_data_ms_small_column, freq_model.beta_data_ms_water_small_column,
            facecolors='none', edgecolors='blue', label='Water — L = 0.05 m',
            s=250, marker='s', linewidths=3.5)
plt.scatter(freq_model.M_data_ms_small_column, freq_model.beta_data_ms_lithium_small_column,
            facecolors='none', edgecolors='darkgreen', label='Lithium — L = 0.05 m',
            s=250, marker='s', linewidths=3.5)
plt.plot(freq_model.M_analytical_plot, freq_model.beta_analytical_results,
         color='red', linestyle='-', linewidth=6,
         label=r'$\tan(\frac{2\pi}{\beta}) + M \frac{2\pi}{\beta} = 0$')

plt.xlabel('M', fontsize=35, fontweight='bold')
plt.ylabel(r'$\beta$', fontsize=35, fontweight='bold')

ax = plt.gca()
plotting.style_axes(ax)
plotting.style_grid(ax, linestyle='--', alpha=1)

legend = plt.legend(loc='best')
plotting.style_legend(legend)

plt.xlim(-0.1, 10.1)
plt.ylim(1.9, 4.1)

plotting.savefig(plt.gcf(), 'figure6_frequency_model_validation')
plt.show()