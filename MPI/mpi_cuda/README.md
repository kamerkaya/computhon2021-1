# computhon2021-1: Jaccard Similarity

## Multi-node, multi-GPU job using MPI and CUDA
In this example, multiple nodes are cooperating to calculate the degrees of a graph and each is utilizing a GPU to accomplish its task. The `deg_cuda.cu` file contains MPI+CUDA code and the `deg_cuda.slurm` script will dispatch a job to SLURM on TRUBA that will compile and execute the code on multiple processes.
