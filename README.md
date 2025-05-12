# Density-based Scanning Probe Microscopy
## A software and Python package to simulate and analyze HR-AFM and BRSTM data.

This repository contains the main program and supporting code and scripts of the DBSPM package. The package is designed to simulate and analyze high-resolution atomic force microscopy (HR-AFM) and bond-resolved scanning tunneling microscopy (BRSTM), primarily with CO-functionalized tips but is easily extendable to other tips. The package is divided into two main parts: the simulation of the microscopy data and the analysis of the simulated data. The simulation part is based on the density functional theory (DFT), while the analysis part is based on Python scripts and Jupyter notebooks.

## Background

### HR-AFM

The HR-AFM part of the package is based on the Full-density-based model (FDBM) [1], and its current implementation [2]. The method is based on the charge density of the molecule and the tip, obtained with DFT, and the interaction between them.

### BRSTM

The BRSTM part of the package combines the tip position data of the FDBM simulation with the Tersoff-Hamann [3] and Chen [4] theories of tunneling current and tunneling through an adsorbed CO, respectively. This methodology is an extension of the model presented in [5].

### References

[1] *Molecular identification, bond order discrimination, and apparent intermolecular features in atomic force microscopy studied with a charge density based method.*
M Ellner, P Pou, R Pérez. (2019)
**ACS nano 13 (1), 786-795**

[2] *Hydrogen bonded trimesic acid networks on Cu (111) reveal how basic chemical properties are imprinted in HR-AFM images.* P Zahl, AV Yakutovich, E Ventura-Macías, J Carracedo-Cosme, C Romero-Muñíz, P Pou, ..., R Pérez. (2022) **Nanoscale 13 (44), 18473-18482**

[3] *Theory of the scanning tunneling microscope.* J Tersoff, DR Hamann. (1985) **Physical Review B 31, 805**

[4] *Tunneling matrix elements in three-dimensional space: The derivative rule and the sum rule.*
C. Julian Chen. (1990)
**Physical Review B 42, 8841**

[5] *High-Resolution Molecular Orbital Imaging Using a p-Wave STM Tip.*
L Gross, N Moll, F Mohn, A Curioni, G Meyer, F Hanke, and M Persson. (2011) **Physical Review Letters 107, 086101**

## Structure

The program files are organized in the following way:

```
|DBSPM/
|--- pydbspm/
|------ calculators/
|--------- vasp.py
|------ __init__.py
|------ calculate.py
|------ grid.py
|------ input_parser.py
|------ interactions.py
|------ parallel.py
|------ sample.py
|------ setup_calculation.py
|------ SPMGrid.py
|------ tools.py
|--- example/
|--- brstm_example/
|--- scripts/
|------ fdbmfit
|------ vaspfz
|--- dbspm
|--- README.md
|--- setup.py
```

The most important files are:

- `dbspm`:
  The main program that runs the simulation.

- `pydbspm/SPMGrid.py`:
  The main class that contains the methods to analyze data.

- `example/`:
  A folder with the example of how to use the package.

- `brstm_example/`':
  A folder with the example of how to use the package with stm and brstm calculations.

- `scripts/vaspfz`:
  A script to automate a Force spectroscopy with VASP to fit the FDBM parameters.

- `scripts/fdbmfit`:
  A script to fit the FDBM parameters to the DFT data obtained with `vaspfz`.

## Installation

The software can be installed directly through PIPI with the following command:

>pip install git+https://github.com/SPMTH/DBSPM

The `dbspm`, `vaspfz`, and `fdbmfit` scripts should appear in your PATH if your Python installation is set up appropriately. The package will be installed in the Python environment.

### Dependencies

PIP will try to install all the dependencies, but it may fail. In this case, you can install the dependencies manually (listed below).

- Standalone programs
    - python (>=3.6)
    - OpenMPI
    - Vasp (optional)
- Python packages
    - NumPy
    - SciPy
    - pandas
    - ASE (Recommended 3.22.1)
    - Matplotlib
    - mpi4py
    - dftd3
    - tricubic
    - vaspwfc

### Environment Setup

To use the VASP interface, ASE needs to know the location of the pseudopotentials. So, we need to add in `~/.bashrc` the following line:

>export VASP_PP_PATH="*path_to_pseudopotentials*"

See ASE documentation for more information.

## Usage

We recommend using the notebook `DBSPM_Basic_Quick-start.ipynb` in the `example/` folder to learn how to use the package. The notebooks are divided into two parts: the simulation and the analysis of the data. To use stm and brstm, use the notebook `DBSPM-BRSTM_Basic_Quick-start.ipynb` in the `brstm_example/` folder

### The dbspm command

The main program is `dbspm`. It is a command-line program that runs the simulation. The program takes as arguments the step or steps that will be run and the input file through the `-i` option. It writes the output in the same folder that is run from (using the LABEL parameters specified in the input file).

> dbspm *step* -i *input_file*

The program supports defining multiple steps separated by spaces. For example:

> dbspm *step1* *step2* -i *input_file*

Other options are available to control the simulation parameters. These can be printed with the `-h` option.

> dbspm -h

Note that any option provided in the command line will override the values in the input file.

### Calculation steps

- `sample`: Calculates the sample data with the specified DFT calculator.
- `tip`: Calculates the tip data with the specified DFT calculator.
- `grid`: Reads the sample and tip data and sets up the calculation grid.
- `sr`: Calculates the short-range interaction between the sample and the tip.
- `es`: Calculates the electrostatic interaction.
- `vdw`: Calculates the van der Waals interaction with DFTD3.
- `relax`: Relax the probe position on the static potential.
- `stm`: Calculates the STM *s* and *p*-wave components. 
- `brstm`: Approximates the bond-resolved STM signal with the HR-AFM and STM data. 

#### Step aliases

| Alias | sample | tip | grid | sr | es | vdw | relax | stm | brstm |
|-------:|:--------:|:-----:|:------:|:----:|:----:|:-----:|:-------:|:-----:|:-------:|
| fullauto* | X | X | X | X | X | X | X | | |
| prepdft | X | X | | | | | | | |
| fromdft | | | X | X | X | X | X | | |
| prep | | | X | | | | | | |
| fullprep | X | X | X | | | | | | |
| pauli | | | | X | | | | | |
| dens | | | | X | X | | | | |
| disp | | | | | | X | | | |
| d3 | | | | | | X | | | |
| static | | | | X | X | X | | | |
| afm | | | | X | X | X | X | | |
| fdbm | | | | X | X | X | X | | |

\* Due to a performance bug, the `fullauto` step is not currently recommended, although the results are correct if used. The most efficient way to run the simulation at the moment is to run the `prepdft` and `fromdft` steps separately.

#### MPI parallelization

The `vdw` and `relax` steps need to be run in parallel mode with MPI. The `dbspm` program should be run with the `mpirun` (or equivalent) command if one of these step is to be performed.

> mpirun -np *nprocs* dbspm *step* -i *input_file*

Note that running the program in this way does not affect the other steps and can be used even if any of them is included in the command.

### Input file

The input file should contain all the required parameters for the calculation. If a parameter is not provided, the program will default to the values in the `pydbspm/input_parser.py` file.

In this file, each flag or variable should be in ALL-CAPS followed by an equal sign with at least one blank space on each side (` = `), and the value should be in python format (except for strings that should not have quote marks ' or ").

Please check out the `input.in` file provided in the example for reference. The basic settings in the file are the following:

- LABEL
    - String to label all the output files of pyFDBM
- Z0 / ZF
    - Smallest (Z0) and largest (ZF) tip-sample separation in the calculation
- ZREF
    - Position along the z-axis of the sample
- ALPHA / V
    - Parameters for the short-range interaction

#### VASP input

VASP parameters can be included in the input file. They have to be preceded with a `$` character (without any space before the parameter name). The supported parameters are the same as in the ASE VASP calculator, as they will be passed on to it, and should follow its format.

## DFT Calculators

The DFT calculators are used to calculate the sample and tip data in their corresponding steps. 

Currently, the package only supports the VASP calculator. However, the package is designed to be easily extendable to other calculators. The calculators should be placed in the `pydbspm/calculators/` subfolder and should have the same structure as the `vasp.py` file. In short, the calculator should have `calculate_sample` and `calculate_tip` functions that interface with the DFT program and take the `params` dictionary from the main program. And two functions, `read_sample` and `read_tip`, that read the DFT output and return the needed data.

The dbspm program also supports defining a custom DFT command (`CMD` in the input file or `--dft-cmd` option in the command line) to run the DFT calculation. The string in this parameter will be passed to the DFT calculator, so it can be used as needed. For example, to run an MPI VASP calculation, the command should be:

> dbspm *step* -i *input_file* --dft-cmd "mpirun -np *nprocs* vasp_gam"

## Disclaimer

This software is in active development and may contain bugs. The developer and their colleagues will do their best to solve any issues reported in the GitHub repository, but we cannot guarantee a quick response.

## Credits

**Main Developer**

- Emiliano Ventura-Macias, Ph.D.

**Contributors and testers**

- Jara Trujillo-Mulero
- Manuel E. Gonzalez Lastre

**Supervisors**

- Pablo Pou, Ph.D.
- Rubén Pérez, Ph.D.


## Acknowledgments

- Michael Ellner Ph.D., for the initial implementation of the FDBM method.
- Prokop Hapala Ph.D., for the useful help and discussions.
