#!/bin/bash

################
# Job will use a single-node (--nodes=1) and will run a single task (MPI process) (--ntasks=1).
# It and will use a single GPU on its single node (--gres=gpu:1) and will use 10 cores for 
# its single task (--cpus-per-task=10)
################


## Parameters for the entire job
#SBATCH --account=<account> # Specify your TRUBA account
#SBATCH --job-name=gputest # The name of the job you see in `squeue` output
#SBATCH --partition=akya-cuda # Partition of servers used
#SBATCH --reservation=computhon # Servers reserved for the people in the competition only!
#SBATCH --time=10:00 # Amount of time that the job will run for. NOTE: if your job doesn't 
                    # finish in time, it will be killed!
#SBATCH --output=gpu-test-%j.out # standard output of the job will be printed here
#SBATCH --error=gpu-test-%j.err # standard error of the job will be printed here

## Resources used by complete job
#SBATCH --nodes=1 # Number of nodes (servers) reserved for the job
#SBATCH --ntasks=1 # Maximum number of tasks (MPI processes) that will be used by the job

## Resources used on each server
#SBATCH --gres=gpu:1 # Number of GPUs reserverd per node
#SBATCH --ntasks-per-node=1 # Maximum number of tasks (MPI processes) that will run on each server 

## Resources used by each task  (MPI process)
#SBATCH --cpus-per-task=10 # Number of cores used per task (MPI process) each process

# Setup the environment
module purge
module load centos7.3/comp/gcc/7
module load centos7.3/lib/cuda/10.1
module load centos7.3/lib/openmpi/1.8.8-gcc-4.8.5

# Compile the code
nvcc filename.cu -o exe 
# for additional OpenMP support: nvcc filename -o exe -Xcompiler -fopenmp
# for the final version use -O3

# Execute with input file and output
srun nvprof ./exe

