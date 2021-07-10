#include <random>
#include <iostream>
#include <omp.h>
using namespace std;

#define VECTOR_LENGTH (1024.0*1024*32)
#define NUM_GPUS 3
#define MIN -100
#define MAX 100
#define NUM_BLOCKS 256
#define NUM_THREADS 256

float generate_random_number(float min, float max);
size_t compare_vectors(float* v1, float* v2, int length, float error_margin);

__global__ void gpu_vector_addition(float *op1_subvector, float *op2_subvector, float *result_subvector, int length){
  int total_threads = blockDim.x * gridDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int elements_per_thread = ceilf(length / float(total_threads));
  int start = tid * elements_per_thread;
  int end = min((tid+1) * elements_per_thread, length);
  for (int i = start; i < end; i++) {
    result_subvector[i] = op1_subvector[i]+op2_subvector[i];
  }
}

void vector_addition(float * op1_vector, float* op2_vector, float* result_vector, int length){
  for (int i = 0;i <length; i++) result_vector[i] = op1_vector[i]+op2_vector[i];
}


int main(){

  float *op1_vector = new float[int(VECTOR_LENGTH)], *op2_vector = new float[int(VECTOR_LENGTH)], *result_vector = new float[int(VECTOR_LENGTH)];
  for (int i = 0; i < VECTOR_LENGTH; i++){
    op1_vector[i] = generate_random_number(MIN, MAX);
    op2_vector[i] = generate_random_number(MIN, MAX);
  }
  double start, end;

  start = omp_get_wtime(); 
  vector_addition(op1_vector, op2_vector, result_vector, VECTOR_LENGTH);
  end = omp_get_wtime();

  cout << "CPU vector addition using took " << end-start << " seconds" << endl;

  float *op1_subvectors_d[NUM_GPUS], *op2_subvectors_d[NUM_GPUS], *result_subvectors_d[NUM_GPUS];
  int elements_per_gpu[NUM_GPUS];
  for (int gpu = 0; gpu < NUM_GPUS; gpu++){
    elements_per_gpu[gpu] = ceil(VECTOR_LENGTH/NUM_GPUS);
    if (gpu == NUM_GPUS-1) elements_per_gpu[gpu] = VECTOR_LENGTH - (ceil(VECTOR_LENGTH/NUM_GPUS)*(gpu));
  }
  float* combined_result_vector = new float[int(VECTOR_LENGTH)];

  // Allocate memory on GPUs
  for (int gpu = 0; gpu < NUM_GPUS; gpu++){
    cudaSetDevice(gpu);
    cudaMalloc(&op1_subvectors_d[gpu], elements_per_gpu[gpu]*sizeof(float));
    cudaMalloc(&op2_subvectors_d[gpu], elements_per_gpu[gpu]*sizeof(float));
    cudaMalloc(&result_subvectors_d[gpu], elements_per_gpu[gpu]*sizeof(float));
  }
 /* for (int gpu = 0; gpu < NUM_GPUS; gpu++){
    int starting_index = int(gpu*ceil(VECTOR_LENGTH/NUM_GPUS));
    cudaSetDevice(gpu);
    cudaMemcpy(op1_subvectors_d[gpu], &(op1_vector[starting_index]), elements_per_gpu[gpu]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(op2_subvectors_d[gpu], &op2_vector[starting_index], elements_per_gpu[gpu]*sizeof(float), cudaMemcpyHostToDevice);
  }
  */

  start = omp_get_wtime();
  // Copy data to GPU
  for (int gpu = 0; gpu < NUM_GPUS; gpu++){
    int starting_index = int(gpu*ceil(VECTOR_LENGTH/NUM_GPUS));
    cudaSetDevice(gpu);
    cudaMemcpyAsync(op1_subvectors_d[gpu], &(op1_vector[starting_index]), elements_per_gpu[gpu]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(op2_subvectors_d[gpu], &op2_vector[starting_index], elements_per_gpu[gpu]*sizeof(float), cudaMemcpyHostToDevice);
    gpu_vector_addition<<<NUM_BLOCKS, NUM_THREADS>>>(op1_subvectors_d[gpu], op2_subvectors_d[gpu], result_subvectors_d[gpu], elements_per_gpu[gpu]);
    cudaMemcpyAsync(&combined_result_vector[starting_index], result_subvectors_d[gpu], elements_per_gpu[gpu]*sizeof(float), cudaMemcpyDeviceToHost);
  }
  for (int gpu = 0; gpu < NUM_GPUS; gpu++){
    cudaSetDevice(gpu);
    cudaDeviceSynchronize();
  }

  end = omp_get_wtime();  
  /*for (int gpu = 0; gpu < NUM_GPUS; gpu++){
    int starting_index = int(gpu*ceil(VECTOR_LENGTH/NUM_GPUS));
    cudaMemcpy(&combined_result_vector[starting_index], result_subvectors_d[gpu], elements_per_gpu[gpu]*sizeof(float), cudaMemcpyDeviceToHost);
  }
  */
  cout << "GPU vector addition using " << NUM_GPUS  << " took " << end-start << " seconds" << endl;

  size_t errors = compare_vectors(result_vector, combined_result_vector, VECTOR_LENGTH, 0.001);
  if (errors > 0){
    cout << errors << " Errors found\n";
  }
  
  return 0;
}

float generate_random_number(float min, float max){
  return min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max-min)));
}
size_t compare_vectors(float* v1, float* v2, int length, float error_margin){
  size_t errors = 0;
  for (int i = 0; i < length ; i++){
    if (abs(v1[i]-v2[i]) > error_margin){
      errors+=1;
    }
  }
  return errors;
}
