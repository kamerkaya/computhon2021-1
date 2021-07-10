#include "mpi.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define VERTEX_AMOUNT 32
#define ROOT 0

void print_matrix(unsigned int ** matrix, int size){
	printf("Matrix:\n");
	for(unsigned int i = 0; i < VERTEX_AMOUNT; i++){
		printf("r%d ", i);
		for(unsigned int j = 0; j < VERTEX_AMOUNT; j++){
			printf("%d ", matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
} 

// https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
#define FULL_MASK 0xffffffff
__global__ void count_degrees(unsigned int n_v, unsigned int * rows, unsigned int * degrees){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int val = rows[tid];
	for (int offset = 16; offset > 0; offset /= 2){
	    val += __shfl_down_sync(FULL_MASK, val, offset); 
	}
	if(tid % 32 == 0){
		degrees[tid / VERTEX_AMOUNT] = val;
	}
}

int main(int argc, char **argv)
{
	/* initialize the mpi environment and report */
	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	if(rank == ROOT){
		printf("Number of processes = %d\n", size);
	}

	unsigned int vertices_per_process = VERTEX_AMOUNT / size;
	unsigned int ** row_ptrs = (unsigned int**)malloc(sizeof(unsigned int*) * VERTEX_AMOUNT); 
	unsigned int * buff = (unsigned int*)malloc(sizeof(unsigned int) * VERTEX_AMOUNT * vertices_per_process);
	unsigned int * degrees;
	if(rank == ROOT){
    degrees = (unsigned int*)malloc(sizeof(unsigned int) * VERTEX_AMOUNT);
		//create an empty adjacency matrix
		//and fill it with random edges
		for(unsigned int i = 0; i < VERTEX_AMOUNT; i++){
			row_ptrs[i] = (unsigned int*)malloc(sizeof(unsigned int) * VERTEX_AMOUNT);
			for(unsigned int j = 0; j < VERTEX_AMOUNT; j++){
				row_ptrs[i][j] = rand() % 2; //unwiegted directed graph  	
			}
		}
		print_matrix(row_ptrs, VERTEX_AMOUNT);

		// distribute vertices and their neighborhood information to processes
		// for each process other than ROOT fill the message buffer with vertex neighborhoods
		// to calculate the degree of a vertex its neighborhood information is enough 
		// thus this problem is emberrasingly parallelisable
		for(unsigned int r = 1; r < size; r++){ 
			for(unsigned int i = 0; i < vertices_per_process; i++){
				unsigned int vertex_id = vertices_per_process * r + i;
				for(unsigned int j = 0; j < VERTEX_AMOUNT; j++){
					//printf("Process: %d, Vertex: %d, Column: %d\n", r, vertices_per_process * r + i, j);
					buff[i * VERTEX_AMOUNT + j] = row_ptrs[vertex_id][j];
				}
			}
			//distribute vertices to processes
			//MPI_Send(buff, VERTEX_AMOUNT * vertices_per_process, MPI_UNSIGNED, r, 0, MPI_COMM_WORLD);  
			MPI_Send(buff, VERTEX_AMOUNT * vertices_per_process, MPI_UNSIGNED, r, 0, MPI_COMM_WORLD);  
		}
	}

	MPI_Status status;
	// Processes other than the ROOT will recieve the vertex neighborhoods, count their neighbors
	// Finally they will return the degree to the ROOT
	if(rank != ROOT){
		// get vertex neighborhoods
		MPI_Recv(buff, VERTEX_AMOUNT * vertices_per_process, MPI_UNSIGNED, ROOT, 0, MPI_COMM_WORLD, &status);

		// calculate degrees of vertices
		//for(unsigned int i = 0; i < vertices_per_process; i++){
		//	unsigned int degree = 0;
		//	for(unsigned int j = 0; j < VERTEX_AMOUNT; j++){
		//		if(buff[i * VERTEX_AMOUNT + j] == 1){
		//			degree++;
		//		}
		//	}
		//	printf("Process: %d, Vertex: %d, Degree: %d\n", rank, vertices_per_process * rank + i, degree);
		//	// return the result to ROOT
		//	// what is inefficient with this?
		//	// what can we do to improve?
		//	MPI_Send(&degree, 1, MPI_UNSIGNED, ROOT, vertices_per_process * rank + i, MPI_COMM_WORLD);  
		//}

		// gpu counting
		unsigned int * degrees = (unsigned int*)malloc(sizeof(unsigned int*) * vertices_per_process);
		unsigned int * d_degrees;
		unsigned int * d_edges;
		cudaMalloc(&d_edges, vertices_per_process*VERTEX_AMOUNT*sizeof(unsigned int));
		cudaMalloc(&d_degrees, vertices_per_process*sizeof(unsigned int));
		cudaMemcpy(d_edges, buff, vertices_per_process*VERTEX_AMOUNT*sizeof(unsigned int), cudaMemcpyHostToDevice);
		count_degrees<<<1,32*vertices_per_process>>>(vertices_per_process, d_edges, d_degrees);
		cudaDeviceSynchronize();
		cudaMemcpy(degrees, d_degrees, vertices_per_process*sizeof(unsigned int), cudaMemcpyDeviceToHost);
		for(unsigned int i = 0; i < vertices_per_process; i++){
			MPI_Send(&degrees[i], 1, MPI_UNSIGNED, ROOT, vertices_per_process * rank + i, MPI_COMM_WORLD);  
		}
	}

	//Needed for correct stdout
	MPI_Barrier(MPI_COMM_WORLD);

	if(rank == ROOT) 
	{
		unsigned int degree = 0;
		// ROOT also computes the degree of vertices_per_process amount of vetices
		for(unsigned int i = 0; i < vertices_per_process; i++){
			degree = 0;
			for(unsigned int j = 0; j < VERTEX_AMOUNT; j++){
				if(row_ptrs[i][j] == 1){
					degree++;
				}	
			}
			printf("Process: %d, Vertex: %d, Degree: %d\n", rank, vertices_per_process * rank + i, degree);
			degrees[i] = degree;
		}

		// ROOT recieves the result from other processes and merges them
		for(unsigned int r = 1; r < size; r++){ //BUGGY
			for(unsigned int i = 0; i < vertices_per_process; i++){
				MPI_Recv(&degree, 1, MPI_UNSIGNED, r, vertices_per_process * r + i, MPI_COMM_WORLD, &status);  
				degrees[r * vertices_per_process + i] = degree;
			}
		}
		printf("\nDegrees: \n");
		for(unsigned int i = 0; i < VERTEX_AMOUNT; i++){
			printf("v: %d, d: %d\n", i, degrees[i]);
		}
	}

	//free dynamically allocated memory
	if(rank == ROOT){
		for(unsigned int i = 0; i < VERTEX_AMOUNT; i++){
			free(row_ptrs[i]);
		}
    free(degrees);
	}
	free(row_ptrs);
	free(buff);

	/* Clean up and exit */
	MPI_Finalize();
	return 0;
}
