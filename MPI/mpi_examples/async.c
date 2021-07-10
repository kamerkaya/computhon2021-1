#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
  int rank, n_ranks;
  int my_first, my_last;
  int numbers = 10;
  MPI_Request request;

  // First call MPI_Init
  MPI_Init(&argc, &argv);

  // Check that there are at least two ranks
  MPI_Comm_size(MPI_COMM_WORLD,&n_ranks);
  if( n_ranks < 2 ){
    printf("This example requires at least two ranks");
    MPI_Finalize();
    return(1);
  }

  // Get my rank
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  if( rank == 0 ){
     char *message = "Hello, world!\n";
     MPI_Isend(message, 16, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &request);
  }

  if( rank == 1 ){
     char message[16];
     MPI_Status status;
     MPI_Irecv(message, 16, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &request);
     MPI_Wait( &request, &status );
     printf("%s",message);
  }

  // Call finalize at the end
  return MPI_Finalize();
}
