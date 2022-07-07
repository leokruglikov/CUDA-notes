#include <iostream>

#define N_THREADS 3
#define N_BLOCKS 512 

__global__ 
void hi_from_gpu(){
    
    printf("Hi from GPU, from thread id %d and block id %d \n", threadIdx.x, blockIdx.x);
}

int main(){
    dim3 k;
    hi_from_gpu<<<N_BLOCKS, N_THREADS>>>();
    cudaDeviceSynchronize();
    return 0;
}
