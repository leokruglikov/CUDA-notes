   /*Perform the necessary declarations, main(), before/after, etc...*/
void reduction(float *d_out, float *d_in, int n_thr, int size){
   cudaMemcpy(d_out, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);
   while(size > 1){
      int n_bl = (size + n_thr - 1)/n_thr;
      reduce_shared<<<n_bl, n_thr, n_thr*sizeof(float), 0>>>(d_out, d_out, size);
      size = n_bl;
   }
}

__global__ void reduce_shared(float* d_out, float* d_in, unsigned int size){
   int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
   extern __shared__ float s_data[];
   s_data[threadIdx.x] = (idx_x < size) ? d_in[idx_x] : 0.f;
      __syncthreads();

// do reduction
   for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
      // thread synchronous reduction
      if ( (idx_x % (stride * 2)) == 0 ){
         s_data[threadIdx.x] += s_data[threadIdx.x + stride];
      }
      __syncthreads();
   }
   if(threadIdx.x == 0){
      d_out[blockIdx.x] = s_data[0];
   }
}