#!/bin/bash

################
# Job will use 3 nodes (--nodes=3). It will run 3 tasks (MPI process) in total 
# (--ntasks-per-node=1 * 3 nodes), and will run 1 task at most on each node (--ntasks-per-node=1).
# It and will use four GPUs on each node (--gres=gpu:4) for a total of 12 GPUS for
# the entire job (4*3). It will use 20 cores for each task (--cpus-per-task=20) 
# for a total of 60 cores for the entire job (3*20).
################


## Parameters for the entire job
##SBATCH --account=aalabsialjundi # Specify your TRUBA account
#SBATCH --job-name=jaccard # The name of the job you see in `squeue` output
#SBATCH --partition=akya-cuda # Partition of servers used
#SBATCH --reservation=computhon # Servers reserved for the people in the competition only!
#SBATCH --time=10:00 # Amount of time that the job will run for. NOTE: if your job doesn't 
                    # finish in time, it will be killed!
#SBATCH --output=jaccard-%j.out # standard output of the job will be printed here
#SBATCH --error=jaccard-%j.err # standard error of the job will be printed here

## Resources used by complete job
#SBATCH --nodes=3 # Number of nodes (servers) reserved for the job

## Resources used on each server
#SBATCH --gres=gpu:4 # Number of GPUs reserverd per node
#SBATCH --ntasks-per-node=1 # Maximum number of tasks (MPI processes) that will run on each server 

## Resources used by each task  (MPI process)
#SBATCH --cpus-per-task=20 # Number of cores used per task (MPI process) each process

# Setup the environment
module purge
module load centos7.3/comp/gcc/7
module load centos7.3/lib/cuda/10.1
module load centos7.3/lib/openmpi/1.8.8-gcc-4.8.5

# Compile the code
g++ jaccard.cpp -o exe

# Execute with input file and output

mpirun ./exe example_graphs/g0.txt out.txt

