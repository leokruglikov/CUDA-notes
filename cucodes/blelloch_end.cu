//first set the last element
  if ( tid == 0 ){
    temp [ n-1] = 0 ;
  }
  for ( int d = 1 ; d < n ; d <<= 1 )
  {
    offset >>= 1 ;
    __syncthreads ( ) ;
    if ( tid < d )
    {
      int ai = offset * ( 2 * tid + 1 ) - 1 ;
      int bi = offset * ( 2 * tid + 2 ) - 1 ;
      float t = temp [ ai ] ;
      //report the i'th elementreport the i'th element to j'th
      temp [ ai ] = temp [ bi ] ;
      //add the 2 elements
      temp [ bi ] += t ;
    }
  }
  __syncthreads() ;
  //set the data
  outData [ 2 * tid ] = temp [ 2 * tid ] ;
  outData [ 2 * tid + 1 ] = temp [ 2 * tid + 1 ] ;
