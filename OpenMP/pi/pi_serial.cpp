/*
  
  This program will numerically compute the integral of
  
  4/(1+x*x) 
  
  from 0 to 1.  The value of this integral is pi -- which 
  is great since it gives us an easy way to check the answer.
  
  History: Written by Tim Mattson, 11/99.
  
*/
#include <stdio.h>
#include <omp.h>
static long num_steps = 1024 * 1024 * 64;
double step;

int main () {
  const int MAX_T = 16;
  int i, t;
  double x, pi, sum = 0;
  double start_time, run_time;

  step = 1.0/(double) num_steps;

  start_time = omp_get_wtime();
  pi = 0.0;
        
  for (i = 0; i < num_steps; i ++){
    x = (i + 0.5) * step;
    sum += 4.0/(1.0+x*x);
  }
  
  pi = sum * step;
  
  run_time = omp_get_wtime() - start_time;
  printf("pi serially is %.16lf in %lf seconds\n",t , pi,run_time);
}
  
  
  

