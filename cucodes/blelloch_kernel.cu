#define BLOCK_SIZE 256
__global__ void scan1 ( float * inData , float * outData , int n )
{
  //allocating the shared memory of size 2*BLOCK_SIZE
  __shared__ float temp [2*BLOCK_SIZE] ;
  int tid = threadIdx . x ; //assign the thread index
  int offset = 1 ;

  //
  temp [ tid ] = inData [ tid ] ;
  temp [ tid + BLOCK_SIZE ] = inData [ tid + BLOCK_SIZE ] ;
  for ( int d = n >> 1 ; d > 0 ; d >>= 1 ) {
  __syncthreads ( ) ;
    if ( tid < d ) {
      int ai = offset * ( 2 * tid + 1 ) - 1 ;
      int bi = offset * ( 2 * tid + 2 ) - 1 ;

      //perform the central sum-tree operation
      temp [ bi ] += temp [ ai ] ;
    }
  offset <<= 1 ;
  }
}
