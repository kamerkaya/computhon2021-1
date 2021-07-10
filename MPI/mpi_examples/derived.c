#include <mpi.h>
#include <stdio.h>

struct Pair {
	 int first;
	 char second;
};

static int num_steps=100000;
int main(int argc, char** argv){
	int rank, num_procs, ms, id_sum=0;
	double start=0, end=0;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	// we have one int in the first field and one char in the second field
	MPI_Datatype typesig[2] = {MPI_INT, MPI_CHAR};
	int block_lengths[2] = {1, 1};

	struct Pair my_pair;

	// we cannot use pointer arithmetic to compute displacements. 
	// always keep in mind that your program might be run on heterogeneous architectures.
	// you have to program for correctness and portability.
	MPI_Aint displacements[2];
	MPI_Get_address(&my_pair.first, &displacements[0]);
	MPI_Get_address(&my_pair.second, &displacements[1]);

	MPI_Datatype mpi_pair;
	// Pair has two fields, hence count = 2 in the call to MPI_Type_create_struct. 
	// All array arguments to this function will have length 2.
	MPI_Type_create_struct(2, block_lengths, displacements, typesig, &mpi_pair);
	MPI_Type_commit(&mpi_pair);

	MPI_Type_free(&mpi_pair);

	MPI_Finalize();
	return 0;
}
