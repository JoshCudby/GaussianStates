#!/bin/bash
#! Name of the job:
#SBATCH -J constraintsjob
#! Number of required nodes (can be omitted in most cases)
#SBATCH -N 1
#! Number of tasks
#SBATCH --ntasks-per-node=1
#! Number of cores per task (use for multithreaded jobs, by default 1)
#SBATCH -c 1
#!How much wallclock time will be required (HH:MM:SS)
#SBATCH --time=10:00:00
#!Output file name
#SBATCH --output=R-%x.%j.out
#SBATCH --mem=1000G

source activate python38-env
srun python3 RunGaussianConstraints.py