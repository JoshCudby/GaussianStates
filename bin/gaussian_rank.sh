#!/bin/bash
#! Name of the job:
#SBATCH -J small_set
#! Partition
#!SBATCH -p skylake
#! Number of tasks
#SBATCH --ntasks-per-node=1
#! Number of cores per task (use for multithreaded jobs, by default 1)
#SBATCH --cpus-per-task 1
#!How much wallclock time will be required (HH:MM:SS)
#SBATCH --time=12:00:00
#!Output file name
#SBATCH --output=Slurm/SmallSet/R-%x.%j.out

source activate python38-env
srun python3 ./bin/gaussian_rank.py
