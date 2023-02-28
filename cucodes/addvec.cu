#include "stdio.h"
#define N_THREADS 512
#define N_BLOCKS 64 
void init_host_vector(double *a, double *b);
void check_result(double *res);

__global__ 
void add_vec(double *a, double *b, double *res){
    //compute the index
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < N_THREADS*N_BLOCKS){
        res[id] = a[id] + b[id];
    }

}

int main(){
    
    const int size_in_bytes =N_THREADS*N_BLOCKS*sizeof(double);
    //initialize the data on HOST
    //malloc() (C) or new (C++) 
    double *hst_a = (double *)malloc(size_in_bytes);
    double *hst_b = (double *)malloc(size_in_bytes);
    double *hst_res = (double *)malloc(size_in_bytes);

    init_host_vector(hst_a, hst_b);

    //allocate memory on GPU
    double* dv_a;    cudaMalloc(&dv_a, size_in_bytes);
    double* dv_b;    cudaMalloc(&dv_b, size_in_bytes);
    double* dv_res;  cudaMalloc(&dv_res, size_in_bytes);

    cudaMemcpy(dv_a, hst_a, size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dv_b, hst_b, size_in_bytes, cudaMemcpyHostToDevice);

    add_vec<<<N_BLOCKS, N_THREADS>>>(dv_a, dv_b, dv_res);
    cudaMemcpy( hst_res, dv_res, size_in_bytes, cudaMemcpyDeviceToHost );

    check_result(hst_res);
    cudaDeviceSynchronize();

    cudaFree(dv_res);   free(hst_res);
    cudaFree(dv_a);     free(hst_a);  
    cudaFree(dv_b);     free(hst_b);
    return 0;
}


void init_host_vector(double *a, double *b){
    for(int i=0; i<N_THREADS*N_BLOCKS; i++){
        a[i] = 1.0;
        b[i] = 1.0;
    }
}

void check_result(double *res){
    bool ok = true;
    for(int i=0; i<N_THREADS*N_BLOCKS; i++){
        if (res[i]<1.9 && res[i]>1.1)
        {
            printf("ERROR\n");
            ok = false;
        }
        
    }
    if (ok)
    {
        printf("OK\n");
    } else{
        printf("ERROR\n");
    }
    
}
