#!/bin/bash
#! Name of the job:
#SBATCH -J recursive_indep_only
#! Partition
#!SBATCH -p skylake
#! Number of tasks
#SBATCH --ntasks-per-node=1
#! Number of cores per task (use for multithreaded jobs, by default 1)
#SBATCH -c 1
#!How much wallclock time will be required (HH:MM:SS)
#SBATCH --time=12:00:00
#!Output file name
#SBATCH --output=./Slurm/R-%x.%j.out
#SBATCH --mem-per-cpu=64G

source activate python38-env
srun python3 RunRecursiveConstraintsIndepOnly.py
