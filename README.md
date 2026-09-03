# Piston Driver Mass-Spring Model

## Overview

This repository contains a mass-spring (MS) model of pressure oscillations in a confined liquid column subjected to impulsive gas loading. The model represents the liquid column as coupled mass-spring elements and is compared with selected CFD results produced with the [ECOGEN](https://github.com/code-mphi/ECOGEN) solver.

## Capabilities

- Simulate water and lithium column responses over selected driving pressures and piston mass ratios.
- Compare mass-spring frequencies with an analytical frequency model.
- Compute control-volume energy contributions and transfer fractions.
- Compare predicted upper and lower gas pressures with the included ECOGEN CFD traces.
- Regenerate the research figures from the model and validation data.

## Requirements and installation

Run the commands below from the repository root. This repository is not packaged for installation; use a Python environment with the direct dependencies installed:

```bash
python -m pip install numpy scipy matplotlib
```

## Quick start

Generate the analytical-frequency comparison figure:

```bash
python -m scripts.figures.figure6_frequency_model_validation
```

The result is written to `figures/figure6_frequency_model_validation.pdf`.

## Reproducing the figures

To generate the complete figure set, run:

```bash
python -m scripts.make_figures
```

Generated files are written to `figures/`. The complete workflow runs the parameter-study and energy-analysis calculations and is computationally substantial. It also includes the CFD-validation comparison, which requires the datasets in `data/cfd/`.

Individual figures can be generated as modules, for example:

```bash
python -m scripts.figures.figure_cfd_vs_ms_validation
```

## Model inputs and outputs

The principal geometry, material properties, initial pressures, driving pressures, and piston mass ratios are defined in `src/parameters.py`.

- `scripts/run_parametric_study.py` runs the predefined mass-spring simulation cases.
- `scripts/data_energy.py` derives control-volume energy quantities from those cases.
- `scripts/figures/` contains the individual research-figure workflows.
- `figures/` receives generated figures.

The included validation datasets are inputs to `scripts/figures/figure_cfd_vs_ms_validation.py`. Details of their contents and use are in [data/README.md](data/README.md).

## CFD validation data

`data/cfd/` contains upper- and lower-gas pressure traces for the water validation case at a 5 MPa driving pressure and mass ratio `M = 0.89`. They are used to compare the mass-spring results with ECOGEN CFD results. See [data/README.md](data/README.md) for filenames, columns, units, and workflow usage.

## Repository structure

```text
.
├── src/                  # Core model, analysis, and plotting modules
├── scripts/              # Parameter study, energy analysis, and figure workflows
│   └── figures/          # One script per research figure
├── data/
│   └── cfd/              # ECOGEN validation traces
├── figures/              # Generated figure outputs
├── README.md
└── LICENSE
```

## Authors

- Tiphaine Bloch
- [Fabian Denner](https://www.polymtl.ca/expertises/en/denner-fabian)

## License

The code in this repository is released under the [MIT License](LICENSE).
