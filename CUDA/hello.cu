#include <stdio.h>

const char STR_LENGTH = 52;
__device__ const char *STR = "HELLO WORLD! HELLO WORLD! HELLO WORLD! HELLO WORLD! ";

__global__ void hello() {
  printf("%c", STR[blockIdx.x]);
}

int main(int argc, char** argv) {
  int device = atoi(argv[1]);
  cudaSetDevice(device);
  
  hello<<<STR_LENGTH, 1>>>();
  printf("\n");
  cudaDeviceSynchronize();		

  return 0;
}
