import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scripts.run_parametric_study import sol_50bar_water_09, x_fluid_50bar_water_09, p_50bar_water_09, x_p_50bar_water_09
from src.parameters import h, c_water

# ============================================================
# GENERATION OF FIGURE 4
# ============================================================

t = sol_50bar_water_09.t                     
x_fluid = x_fluid_50bar_water_09              
p = p_50bar_water_09                          
x_piston = x_p_50bar_water_09                  

n, nt = x_fluid.shape

t_flat = np.broadcast_to(t[np.newaxis, :], x_fluid.shape).flatten()
x_flat = x_fluid.flatten()
p_flat = p.flatten()

x_flat_norm = (x_flat - 0.25) / h
t_flat_norm = (t_flat / (h / c_water))-38.3

x_top = x_fluid[-1, :] 
x_bottom = x_piston        

colors = [
    (0.0, "midnightblue"),
    (0.2, "dodgerblue"),
    (0.5, "white"),
    (0.7, "lightsalmon"),
    (1.0, "darkred")
]
vmin = -35e6  
vmax = 125e6  

neg_ticks = np.array([ -1.5e7,-2.5e7,-3.5e7])
pos_ticks = np.array([25e6, 50e6, 75e6, 100e6, 125e6])
zero_tick = np.array([0.0])
ticks = np.concatenate((neg_ticks, zero_tick, pos_ticks))
tick_labels = [f"{int(t/1e6)} MPa" for t in ticks]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_red_blue", colors)
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

plt.figure(figsize=(18, 9))

ticks = ticks
tick_labels = tick_labels

plt.plot((t / (h / c_water))-38.3, (x_top - 0.25) / h, 'k-', linewidth=2)
plt.plot((t / (h / c_water))-38.3, (x_bottom - 0.25) / h, 'k-', linewidth=2)
plt.scatter(t_flat_norm, x_flat_norm, c=p_flat, cmap=cmap, norm=norm, s=5)

cbar = plt.colorbar()
cbar.set_label("Pressure", fontsize=35, fontweight='bold')
cbar.set_ticks(ticks)
cbar.ax.set_yticklabels(tick_labels, fontsize=35, fontweight='bold')

plt.xlabel(r"$\tau - \tau_{\mathrm{peak}}$", fontsize=35, fontweight='bold')
plt.ylabel("x/L", fontsize=35, fontweight='bold')

ax = plt.gca()
ax.tick_params(labelsize=32)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.ylim(0.2, 2.25)
plt.xlim(-5, 20)
plt.savefig('figures/figure4_spatiotemporal_pressure_map.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
plt.show()