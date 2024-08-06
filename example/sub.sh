#!/bin/bash
#SBATCH --job-name=pyFDBM
#SBATCH --chdir=.
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
#SBATCH -N 1
#SBATCH -p spmth
#SBATCH --time=30:00
#SBATCH --mem-per-cpu=3900
#SBATCH --get-user-env

#PSEUDOS='/apps/exported/installed/software/vasp/PP/potpaw_PBE.54/'
#cat $PSEUDOS > POTCAR

source /home/eventura/.bashrc
module load intel/oneAPI/conda/ vasp/6.3.0
export VASP_PP_PATH='/apps/exported/installed/software/vasp/PP/'

mpirun -n 1 ../dbspm sample -i input.in --dft-cmd="mpirun -np 32 vasp_gam" > test.out
