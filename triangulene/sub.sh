#!/bin/bash
#SBATCH --job-name=dbspm      ## Job name: BATCH_JOB-MP 
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
#SBATCH -N 1
#SBATCH --partition=low_priority
#SBATCH --get-user-env            ## Exports all local SHELL vars
#SBATCH --output=%j.out     ## Slurm STDOUT
#SBATCH --error=%j.error    ## Slurm STDERR
#SBATCH --time=24:00:00           ## MaxTime for this JOB
#SBATCH --mem-per-cpu=7968        ## RAM/NUM_CORES ratio: 3936 for 'ram256GB' AND 7968 for 'ram512GB'
#NOTE: 
# - '#SBATCH' is NOT a commented line
# - '##SBATCH' this one YES
# - '# SBATCH' this one also.

#PSEUDOS='/apps/exported/installed/software/vasp/PP/potpaw_PBE.54/'
#cat $PSEUDOS > POTCAR

source /home/eventura/.bashrc
module load intel/oneAPI/conda/ vasp/6.3.0
conda activate pyFDBMenv
export VASP_PP_PATH='/apps/exported/installed/software/vasp/PP/'

#mpirun -n 1 ../dbspm brstm -d SPIN -b 1.5 -i input.in  > test.out
mpirun -n 32 ../dbspm relax -i input.in  > vdw.out