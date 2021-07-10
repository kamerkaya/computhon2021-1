#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#define NPROCS 4

int main(int argc, char *argv[]) {
	int rank;
	int size;
	int new_rank;
	int sendbuf;
	int recvbuf;
	int count;

	MPI_Comm new_comm;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if(rank == 0){
		printf("MPI_COMM_WORLD size = %d\n", size);
	}

	// check that the size of MPI_COMM_WORLD matches with our grouping expecations
	if (size != NPROCS) {
		fprintf(stderr, "Error: Must have %d processes in MPI_COMM_WORLD\n",
				NPROCS);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	// we will send the rank of the process in MPI_COMM_WORLD
	sendbuf = rank;
	count = 1;

	// split the processes in half
	// new_comm will have different groups of processors
	int res = MPI_Comm_split(MPI_COMM_WORLD, rank < NPROCS / 2, rank, &new_comm);
	MPI_Comm_size(new_comm, &size);
	//if(rank == 0){
		printf("New comunicator success = %s, size = %d\n", res == MPI_SUCCESS ? "true": "false", size);
	//}


	MPI_Comm_rank(new_comm, &new_rank);
	printf("rank= %d newrank= %d\n", rank, new_rank);

	int result;
	MPI_Comm_compare(MPI_COMM_WORLD,new_comm,&result);
	if(rank == 0){
		printf("assign:    comm==copy: %d \n",result==MPI_IDENT);
		printf("            congruent: %d \n",result==MPI_CONGRUENT);
		printf("            not equal: %d \n",result==MPI_UNEQUAL);
	}

	// compute total of ranks in MPI_COMM_WORLD in the newer, smaller communicator
	MPI_Allreduce(&sendbuf, &recvbuf, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	printf("WORLD: rank= %d newrank= %d recvbuf= %d\n", rank, new_rank, recvbuf);

	MPI_Allreduce(&sendbuf, &recvbuf, count, MPI_INT, MPI_SUM, new_comm);
	printf("New Comm: rank= %d newrank= %d recvbuf= %d\n", rank, new_rank, recvbuf);

	MPI_Comm_free(&new_comm);

	MPI_Finalize();

	return 0;
}
