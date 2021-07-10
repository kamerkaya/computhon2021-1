#!/bin/bash
#SBATCH --account=<account> #your_user_account
#SBATCH --job-name=omp-example
#SBATCH --nodes=1 #number of nodes to be used
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --reservation=computhon
#SBATCH --partition=akya-cuda #name of the job queue
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
echo "***********************"
echo "Serial"
g++ -o pi_serial pi_serial.cpp -fopenmp
srun ./pi_serial
echo "***********************"
echo "0) Naive: "
g++ -o pi 0_pi_omp_false_sharing.cpp -fopenmp
srun ./pi 
echo "***********************"
echo "1) Padding (solving false sharing): "
g++ -o pi_padded 1_pi_omp_padded.cpp -fopenmp
srun ./pi_padded
echo "***********************"
echo "2) Locks (bad): "
g++ -o pi_lock 2_pi_omp_lock.cpp -fopenmp
srun ./pi_lock
echo "***********************"
echo "3.a) Locks: "
g++ -o pi_lock_good 3_pi_omp_lock_good.cpp -fopenmp
srun ./pi_lock_good
echo "***********************"
echo "3.b) Critical: "
g++ -o pi_critical 4_pi_omp_critical.cpp -fopenmp
srun ./pi_critical
echo "***********************"
echo "3.c) Atomic: "
g++ -o pi_atomic 5_pi_omp_atomic.cpp -fopenmp
srun ./pi_atomic
echo "***********************"
echo "6) Reduction: "
g++ -o pi_reduction 6_pi_omp_loop.cpp -fopenmp
srun ./pi_reduction
echo "***********************"
exit
