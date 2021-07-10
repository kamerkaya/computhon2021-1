# computhon2021-1: Jaccard Similarity

## OMP parallelization of pi integraion
Different implementations of the Pi integration algorithm shown in the presentation on the first day. Each of the C++ files, with the exception of `pi_serial.cpp`, uses a different technique for parallelizing computation using OpenMP. The SLURM script included will compile the codes and execute them using a variable number of threads. 

### SLURM file
* `pi.slurm`: dispatches a job to TRUBA that will compile and execute all the C++ codes using 1, 2, 4, 8, and 16 cores and show their timings.

### C++ files
1. `pi_serial.cpp`: sequential implementation of pi integration.
2. `0_pi_omp_false_sharing.cpp`: OMP parallelized implementation using a shared array to store per-thread sums. Suffers from false sharing.
3. `1_pi_omp_padded.cpp`: OMP parallelized using a shared array with padding to solve the problem of false sharing.
4. `2_pi_omp_lock.cpp`: inefficient paralleization using OMP that uses a lock around the most compute intensive part of the loop.
5. `3_pi_omp_lock_good.cpp`: efficient OMP parallelization that uses a single private variable per thread to store intermediate sums and protects the summation of intermediate values into the shared result with a lock.
6. `4_pi_omp_critical.cpp`: similar to the above implementation, except that the summation of intermediate values step is inside a critical section instead of being protected by a lock.
7. `5_pi_omp_atomic.cpp`: just like the previous two implementations, uses a private variable per thread to store intermediate summations and protects the final step of adding all the summations with an atomic construct.
8. `6_pi_omp_loop.cpp`: uses the OMP for construct to carry out the integration and uses the reduction option to sum-up the intermediate values.
