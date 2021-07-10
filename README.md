# computhon2021-1: Jaccard Similarity

The source codes and simple test cases for the first Computhon. 

## Quickstart

We will clone the repo on TRUBA and dispatch a SLURM job that will compile and execute the Jaccard code on the sample `example_graphs/g0.txt` graph on a single node using 10 nodes.

First, while logged into TRUBA, we clone the repository and navigate to it:
```
git clone https://github.com/kamerkaya/computhon2021-1
cd computhon2021-1
```

Then, we dispatch the job by executing the following command with `<account>` being our account on TRUBA:

```
sbatch example_scripts/script_1node_10core_1gpu.slurm -A <account>
```

The outputs of the job will be in the file `jaccard-%j.out` and `jaccard-%j.err` where %j is the job ID.

### SLURM scripts

You can start sending jobs to the cluster directly using one of the given SLURM scripts in the directory `example_scripts`. You can either use the script `example_scripts/slurm_template.slurm` as a template and fill out the options you need (advanced), or use one of the ready made SLURM scripts that cover the three categories of the competition:

1. `example_scripts/script_1node_10core_1gpu.slurm`: this script will reserve a job that will use 1 node (server), 10 cores, 1 GPU, and a single task (process).
2. `example_scripts/script_1node_10core_4gpu.slurm`: reserves a job that will use 1 node, 1 process (task), 10 cores, and 4 GPUs.
3. `example_scripts/script_3node_60core_12gpu.slurm`: reserves a job that will use 3 nodes, 3 processes (tasks), 20 cores per process (60 cores total), and 4 GPUs per node (12 total).

The above three scripts include a paragraph at the top explaining the different options used to establish the job parameters. In addition, all the scripts will load the required modules to use OMP, CUDA, and MPU. Finally, the scripts, by default, will compile and execute the sample Jaccard code (`jaccard.cpp`) on the graph `example_graphs/g0.txt`. 

To execute any of the above scripts (as well as the template script) use the following command:
```
sbatch example_scripts/<script> -A <account>
```
where `<script>` is the script you'd like to run and `<account>` is your TRUBA account.

## Graphs
Three example graphs are given in the directory `example_graphs`.
