#!/bin/bash

## Parameters for the entire job
#SBATCH --account=<account> # Specify your TRUBA account
#SBATCH --job-name=jaccard # The name of the job you see in `squeue` output
#SBATCH --partition=akya-cuda # Partition of servers used
#SBATCH --reservation=computhon # Servers reserved for the people in the competition only!
#SBATCH --time=<hh:mm:ss>     # Amount of time that the job will run for. NOTE: if your job doesn't 
                    # finish in time, it will be killed!
#SBATCH --output=jaccard-%j.out # standard output of the job will be printed here
#SBATCH --error=jaccard-%j.err # standard error of the job will be printed here

## Resources used by complete job
#SBATCH --nodes=<num_servers> # Number of nodes (servers) reserved for the job

## Resources used on each server
#SBATCH --gres=gpu:<num_gpus> # Number of GPUs reserverd per node
#SBATCH --ntasks-per-node=<num_tasks> # Maximum number of tasks (MPI processes) that will run on each server 

## Resources used by each task  (MPI process)
#SBATCH --cpus-per-task=<num_cores_per_task> # Number of cores used per task (MPI process) each process

# Setup the environment
module purge
module load centos7.3/comp/gcc/7
module load centos7.3/lib/cuda/10.1
module load centos7.3/lib/openmpi/1.8.8-gcc-4.8.5

# Compile the code
g++ jaccard.cpp -o exe

# Execute with input file and output
srun ./exe example_graphs/g0.txt out.txt
