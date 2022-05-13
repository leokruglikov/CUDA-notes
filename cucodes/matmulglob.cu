#include "stdio.h"
#define BLOCK_SIZE 16

typedef struct{
    int height; int width; float* element;
}Matrix;

__global__ void matmul_global(const Matrix a,\
const Matrix b, Matrix c);

int main(){
    //init matrices A and B
    Matrix A; A.height = 32; A.width = 32;
    Matrix d_A; d_A.height = A.height; d_A.width = A.width;
    int size = sizeof(float)*d_A.height*d_A.width;
    cudaMalloc(&(d_A.element), size);
    cudaMemcpy(d_A.element, A.element, size, cudaMemcpyHostToDevice);
    /* same for d_B
        ...
    */

    //prepare memory, for device to write to
    Matrix d_C; d_C.height = d_A.height; d_C.width = d_A.width;
    cudaMalloc(&(d_C.element), size);

    //prepare dimensions of the kernel (2D indexing)
    dim3 block_dim = (BLOCK_SIZE, BLOCK_SIZE);  //dimension of block                 
    dim3 grid_dim = (A.width/BLOCK_SIZE, A.height/BLOCK_SIZE); //dim. of blocks grid 
    matmul_global<<<grid_dim, block_dim>>>(d_A, d_B, d_C);

    /*
    cudaMemcpy(...); free(...); cudaFree(...); //free the ressources
    */
}

__global__ void matmul_global(const Matrix A, const Matrix B, Matrix C){
    int row_id = blockDim.y*blockIdx.y + threadIdx.y;
    int col_id = blockDim.x*blockIdx.x + threadIdx.x;

    //accumulate sum for c_{row_id,col_id} element
    float tempsum = 0.0;
    for(int k = 0; k<A.width; k++){
        tempsum += A.element[row_id*A.width + k]*\
                   B.element[k*B.width + col_id];
    }
    C.element[row_id*C.width + col_id] = tempsum;
}