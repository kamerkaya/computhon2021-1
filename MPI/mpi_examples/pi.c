#include <mpi.h>
#include <stdio.h>

//#define REDUCE

static long long num_steps=100000000;
double step;
int main(int argc, char** argv){
	int i, myid, num_procs;
	double x, pi=0, remote_sum, sum=0, start=0, end=0, all_sum=0;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	start = MPI_Wtime();
	step = 1.0/(double) num_steps;
	for (i = myid; i< num_steps; i=i+num_procs){
		x =(i+0.5)*step;
		sum +=4.0/(1.0+x*x);
	}
#ifdef REDUCE
	MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	pi *= step;
#else
	if (myid==0){
		for (i = 1; i< num_procs;i++){
			MPI_Status status;
			MPI_Recv(&remote_sum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
			sum +=remote_sum;
		}
		pi=sum*step;
	} else {
		MPI_Send(&sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
#endif
	MPI_Finalize();

	if (myid ==0){
		end = MPI_Wtime();
#ifdef REDUCE
		printf("All Reduce: ");
#else
		printf("Send-Receive: ");
#endif
		printf("Processors %d, took %f, value: %lf\n", num_procs, end-start, pi);
	}
	return 0;
}
