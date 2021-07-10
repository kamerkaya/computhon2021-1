#include <stdio.h>
#include <omp.h>
#include <chrono>
#include "common.h"

#define N (256 * 1024 * 1024)
#define THREADS_PER_BLOCK 256

__global__ void vector_add(float *a, float *b, float *c) {
  int blockID =  gridDim.y * blockIdx.x +  blockIdx.y;
  int index = (THREADS_PER_BLOCK * blockID) + threadIdx.x;
  c[index] = a[index] + b[index];
}

int main() {
  cudaSetDevice(0);
  
  /******************************************************************************/
  /* Getting the device properties here */
  int device;
  cudaGetDevice(&device);
  
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("  Device Number: %d\n", 0);
  printf("  Device name: %s\n", prop.name);
  printf("  Warp size: %d\n", prop.warpSize);
  printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1000);
  printf("  Clock rate (MHz): %d\n", prop.clockRate / 1000);
  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6	\
	 );
  printf("  Global memory size (GB): %.2f\n", (prop.totalGlobalMem + .0f) / (1000000000));
  printf("  Shared mem per block: %ld\n", prop.sharedMemPerBlock);
  printf("  Major-minor: %d.%d\n", prop.major, prop.minor);
  printf("  Device overlap: %d\n", prop.deviceOverlap); //Device can concurrently copy memory and execute a kernel
  printf("  Compute mode: %d\n", prop.computeMode); //0 is cudaComputeModeDefault (Multiple threads can use cudaSetDevice() with this device)
  puts("");
/******************************************************************************/
  
  /******************************************************************************/
  //Preparing the memory
  float *a, *b, *c;
  float *d_a, *d_b, *d_c;
  size_t size = N * sizeof(float);
  printf("%f MBbytes will be allocated for a b c (size variable is %ld)\n", ((double)N / 1e+6) * sizeof(float) * 3, size );
  /* allocate space for device copies of a, b, c */

  cudaCheck(cudaMalloc( (void **) &d_a, size ));
  cudaCheck(cudaMalloc( (void **) &d_b, size ));
  cudaCheck(cudaMalloc( (void **) &d_c, size ));

  cudaCheck(cudaPeekAtLastError());
  
  /* allocate space for host copies of a, b, c and setup input values */
  a = (float *)malloc( size );
  b = (float *)malloc( size );
  c = (float *)malloc( size );
  
  for(int i = 0; i < N; i++ ) {
    a[i] = b[i] = 1; c[i] = 0;
  }
  
  /******************************************************************************/
  //Timing
  dim3 block( 1024, 1024 );
  
  /******************************************************************************/

  cudaCheck(cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice ));
  cudaCheck(cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice ));
  cudaCheck(cudaMemcpy( d_c, c, size, cudaMemcpyHostToDevice ));

  auto start = omp_get_wtime();
  vector_add<<<block, THREADS_PER_BLOCK >>>(d_a, d_b, d_c);
  cudaCheck(cudaDeviceSynchronize());
  auto end = omp_get_wtime();
  auto diff = end - start;

  cudaCheck(cudaPeekAtLastError());
  cudaCheck(cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost ));

  for( int i = 0; i < N; i++) {
    if(c[i] != 2) {
      printf("GPU Error: value c[%d] = %f\n", i, c[i]);
      break;
    }
  }
  printf("GPU time is: %lf seconds\n", diff);
  /******************************************************************************/
  
  for( int i = 0; i < N; i++) c[i] = 0;
  double start_time = omp_get_wtime();
#pragma omp parallel
  {
#pragma omp single
    printf("number of threads is: %d\n", omp_get_num_threads());
#pragma omp for schedule(static)
    for( int i = 0; i < N; i++ )
      c[i] = a[i] + b[i];
  }
  double end_time = omp_get_wtime();
  for( int i = 0; i < N; i++) {
    if(c[i] != 2) {
      printf("CPU Error: value c[%d] = %f\n", i, c[i]);
      break;
    }
  }
  printf("CPU time is: %lf seconds\n", end_time - start_time);
  
  free(a);
  free(b);
  free(c);

  cudaFree( d_a );
  cudaFree( d_b );
  cudaFree( d_c );
  
  return 0;
} /* end main */
