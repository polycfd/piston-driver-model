import matplotlib.pyplot as plt
from src import plotting
from scripts import run_parametric_study as param_study

# ============================================================
# GENERATION OF FIGURE 5
# ============================================================

plotting.new_figure()
plt.plot(param_study.sol_50bar_water_09.t*1e3, param_study.p_50bar_water_09[0]/1e6, label='x/L=0', lw=5)
plt.plot(param_study.sol_50bar_water_09.t*1e3, param_study.p_50bar_water_09[49]/1e6, label='x/L=0.5', lw=5)
plt.plot(param_study.sol_50bar_water_09.t*1e3, param_study.p_50bar_water_09[99]/1e6, label='x/L=1', lw=5)

plt.xlabel('Time [ms]', fontsize=35, fontweight='bold')
plt.ylabel('Pressure [MPa]', fontsize=35, fontweight='bold')

ax = plt.gca()
plotting.style_axes(ax)
plt.grid(True)

legend = plt.legend(loc='best')
plotting.style_legend(legend)

plt.xlim(5, 10)
plt.ylim(-50, 50)
plt.savefig("figures/figure5_standing_wave_pressure_time.pdf", bbox_inches='tight', pad_inches=0.05, dpi=300)
plt.show()