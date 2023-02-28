#include "stdio.h"
#include <iostream>
#include <algorithm>

__device__ void swap_ptr(int* p_a, int* p_b){
    int id = blockDim.x*blockIdx.x + threadIdx.x;
    p_a[id] = p_b[id];
}

__device__ void print_arr(int* p_a, int* p_b, int size){
  printf("The arr in:\n");
  for(int i = 0; i<size; i++){
    printf("%d,", p_a[i]);
  } 
  printf("\n");
  
  printf("The arr out:\n");
  for(int i = 0; i<size; i++){
    printf("%d,", p_b[i]);
  } 
  printf("\n");
}

__global__ void reduce_block(int* input, int* output, int size){
  int id = threadIdx.x;
  for(int i = 1; i<=size/2; i*=2){
    if(id < (size)-i){
      output[i+id] = input[id]+input[id+i];
    }
    __syncthreads();
    swap_ptr(input, output);
    __syncthreads();
  }
}

void fill_arr(int* a, int size){
  for(int i = 0; i<size; i++){
    a[i]=i+1;
  }
}

void print(int *a, int size){
  std::cout<<"Array given by: \n";
  for(int i = 0; i<size; i++){
    std::cout<<a[i]<<",";
  }
  std::cout<<"\n";
}

int* scan_seq(int *in, int size){
  int* to_ret = (int*)malloc(size);
  int temp = 0;
  for(int i = 0; i<size; i++){
    to_ret[i] = temp+in[i];
    temp+=in[i];
  }
  return to_ret;
}

int main(){
int n_threads = 1024;
int *h_a; int *h_b; int* d_a; int* d_b;
h_a = (int*)malloc(n_threads*sizeof(int));
h_b = (int*)malloc(n_threads*sizeof(int));
std::cout<<"cudaMalloc\n";
cudaMalloc(&d_a, n_threads*sizeof(int));
cudaMalloc(&d_b, n_threads*sizeof(int));

std::cout<<"fill_arr\n";
fill_arr(h_a, n_threads);
fill_arr(h_b, n_threads);
auto temp = scan_seq(h_a, n_threads);

std::cout<<"cudaMemcpy\n";
cudaMemcpy(d_a, h_a, n_threads*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, n_threads*sizeof(int), cudaMemcpyHostToDevice);

std::cout<<"reduce_block\n";
reduce_block<<<1, n_threads>>>(d_a, d_b, n_threads);

std::cout<<"sync\n";
cudaDeviceSynchronize();
cudaMemcpy(h_a, d_a, n_threads*sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(h_b, d_b, n_threads*sizeof(int), cudaMemcpyDeviceToHost);


print(h_a, n_threads);
std::cout<<"Sequential:\n";
print(temp, n_threads);
cudaFree(d_a);
cudaFree(d_b);



}
