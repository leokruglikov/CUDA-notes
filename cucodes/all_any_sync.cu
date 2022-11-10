#include "stdio.h"
#include <cuda.h>

__global__ void all_any_example(){
  int id = threadIdx.x;
  
  //0 if odd, 1 if even
  int id_is_even = !(id%2);

  //get the mask of active threads
  unsigned int mask = __activemask();

  //get the result of functins of interest
  int any = __any_sync(mask, id_is_even);
  int all = __all_sync(mask, id_is_even);
  __syncthreads();
  
  //the result is the same for all threads
  //thus only print the result from 1 thread
  if(id == 0){
    printf("Result of __any_sync: %d\n", any);
    printf("Result of __all_sync: %d\n", all);
  }
}

__device__ void by_bit_output(int result){
  int size_in_bits = 32;

  printf("\n");
  for(int i = 0; i<size_in_bits; i++){
    printf("%d\t", (result & ( 1 << i )) >> i);
  }
  printf("\n");

}

__global__ void ballot_example(){
  int id = threadIdx.x;

  //the condition to be true is the 
  //lane index to be equal to 12
  int target_thread_id = 12;

  //the predicate computed
  int predicate = (target_thread_id == id);

  //get the mask of active threads
  unsigned int mask = __activemask();
  int result_int = __ballot_sync(mask, predicate);

  __syncthreads();
  
  //print the result once
  if(id == 0){
    by_bit_output(result_int);
  }
}

int main(){
  int n_threads = 32;
  int n_blocks = 1;
  all_any_example<<<n_blocks, n_threads>>>();
  printf("\n\n");
  ballot_example<<<n_blocks, n_threads>>>();

  cudaDeviceSynchronize();
  
  return 0;
}
