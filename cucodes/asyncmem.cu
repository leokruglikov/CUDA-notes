__global__ void vec_add_kernel(double *c, double *a, double *b);

int main(){
    //----------------------------------//
    //PERFORM THE NECESSARY INITIALIZATION//

    //init the pointers as usual
    double *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
    int size = 1<<24; //a good number which is a power of 2
    int buffsize = size*sizeof(double);
    int n_streams = 4; //number of streams that we'll use

    //initialize the memory on host 
    //using the paged-locked transfer strategy
    cudaMallocHost((void**)&h_a, buffsize);
    cudaMallocHost((void**)&h_b, buffsize);
    cudaMallocHost((void**)&h_c, buffsize);

    //initialize memory on the device
    cudaMalloc((void**)&d_a, buffsize);
    cudaMalloc((void**)&d_b, buffsize);
    cudaMalloc((void**)&d_c, buffsize);
    //---------------------------------//
    //INITIALIZE THE STREAMS AND START KERNELS//

    cudaStram_t streams = new cudaStream_t[n_streams];

    for(int i=0; i<n_streams; i++){
        int offset = i*(size/n_streams); //the offset for every array
        
        //copy the memory asynchronously
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], bufsize, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_b[offset], &h_b[offset], bufsize, cudaMemcpyHostToDevice, streams[i]);

        //initialize the grid, on which the kernel will be executed
        dim3 dimBlock(256); dim3 dimGrid((size/n_streams)/dimBlock.x);
        //add a certain elements of the vector.
        //note that if one computes the sum multiple times, it doesn't cause any problems
        vec_add_kernel<<<dimGrid, dimBlock, 0, streams[i]>>>(&d_c[offset],
                                                             &d_a[offset],
                                                             &d_b[offset]);
        //retrieve the data 
        cudaMemcpyAsync(&h_c[offset], &d_c[offset], cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();

    //don't forget to free the resources on both host and device
}
