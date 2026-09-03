# CFD validation data

`data/cfd/` contains the pressure traces used to validate the mass-spring model against CFD results from the [ECOGEN](https://github.com/code-mphi/ECOGEN) solver. The files correspond to the water case with a 5 MPa driving pressure and piston-to-liquid mass ratio `M = 0.89`.

| File | Contents |
| --- | --- |
| `gasbot_5MPa_water_M089.csv` | Lower-gas pressure trace. |
| `gastop_5MPa_water_M089.csv` | Upper-gas pressure trace. |

Both files are headerless, comma-separated numeric data with four columns:

| Column (zero-based index) | Confirmed meaning | Unit |
| --- | --- | --- |
| 0 | Simulation time | s |
| 1 | Not documented in this repository; not consumed by the Python workflows | — |
| 2 | Not documented in this repository; not consumed by the Python workflows | — |
| 3 | Gas pressure | Pa |

`scripts/figures/figure_cfd_vs_ms_validation.py` reads column 0 from `gasbot_5MPa_water_M089.csv` as the time vector and column 3 from each file as the lower- and upper-gas pressure traces. It uses these data to generate the two mass-spring-versus-CFD comparison figures. The complete `python -m scripts.make_figures` workflow includes this comparison.

The repository records the case fluid, driving pressure, and mass ratio above. For this CFD validation context, air is modeled as an ideal gas with $\gamma = 1.4$ and water with a stiffened-gas equation of state ($\gamma = 4.4$, $p_\infty = 6 \times 10^8$ Pa). 