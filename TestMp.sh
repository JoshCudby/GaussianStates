#!/bin/bash
#! Name of the job:
#SBATCH -J test_mp
#! Partition
#!SBATCH -p skylake
#! Number of tasks
#SBATCH --ntasks=1
#! Number of cores per task (use for multithreaded jobs, by default 1)
#SBATCH --cpus-per-task=2
#!How much wallclock time will be required (HH:MM:SS)
#SBATCH --time=12:00:00
#!Output file name
#SBATCH --output=./Slurm/R-%x.%j.out
#SBATCH --mem-per-cpu=64G

source activate python38-env

export OMP_PROC_BIND=true
echo 'Running on '$HOSTNAME

srun python3 TestMp.py
