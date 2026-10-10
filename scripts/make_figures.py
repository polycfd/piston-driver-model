import runpy

# ============================================================
# GENERATION OF THE PAPER'S FIGURES
# ============================================================

figure_modules = [
    "scripts.figures.figure4_spatiotemporal_pressure_map",
    "scripts.figures.figure5_standing_wave_pressure_time",
    "scripts.figures.figure6_frequency_model_validation",
    "scripts.figures.figure7_control_volume_energy_validation",
    "scripts.figures.figure8_energy_transfer_fraction",
    "scripts.figures.figure9_energy_balance_contributions",
    "scripts.figures.figure10_minimum_pressure_fields",
    "scripts.figures.figure_cfd_vs_ms_validation",
]

if __name__ == "__main__":
    for module_name in figure_modules:
        print(f"--- Generating {module_name} ---")
        runpy.run_module(module_name, run_name="__main__")
    print("All figures generated in figures/")