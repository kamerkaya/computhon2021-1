#!/bin/bash
#SBATCH -A <account> #your_user_account
##SBATCH --res=computhon
#SBATCH -J mpi-example
#SBATCH -N 2 #number of nodes to be used
#SBATCH --ntasks-per-node=2
#SBATCH -c 1 #number of cores per task.. Not mandatory
#SBATCH -p debug
#SBATCH --time=00:15:00
#SBATCH --output=./slurm/%j.out
#SBATCH --error=./slurm/%j.err

module unload centos7.3/lib/openmpi/1.8.8-gcc-4.8.5
module load centos7.3/lib/openmpi/1.8.8-gcc-4.8.5
#module unload centos7.3/lib/openmpi/4.0.1-gcc-7.0.1
#module load centos7.3/lib/openmpi/4.0.1-gcc-7.0.1

#mpicc -g -Wall -std=c11 -o communicator comm.c 
#mpirun ./communicator

mpicc -o pi pi.c
mpirun ./pi 

#mpicc -o broadcast broadcast.c
#mpirun ./broadcast

#mpicc -o send-receive reduceVSsendrecv.c
#mpirun ./send-receive 
#mpicc -o reduce reduceVSsendrecv.c -DREDUCE
#mpirun ./reduce 

#mpicc -o scatter-gather scatter-and-gather.c
#mpirun ./scatter-gather

#mpicc -o async async.c
#mpirun ./async

#mpicc -o derived derived.c
#mpirun ./derived

#mpicc -o pokemon pokemon.c
#mpirun ./pokemon

#mpicc -o degree_counting degree_counting.c
#mpirun ./degree_counting
exit
