#include <sys/time.h>
#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
//#include <cuda.h>
//#include <helper_cuda.h>

// time stamp function in seconds
double getTimeStamp() {
	struct timeval tv ;
	gettimeofday( &tv, NULL ) ;
	return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}
// host side matrix addition
void h_addmat(float *A, float *B, float *C, int nx, int ny){

for (int i =0;i<nx;i++){

	for(int j=0;j<ny;j++){
		C[i*ny+j] = A[i*ny+j]+B[i*ny+j]; 
	}

}
return;
}
// device-side matrix addition
__global__ void f_addmat( float *A, float *B, float *C, int nx, int ny ){
	// kernel code might look something like this
	// but you may want to pad the matrices and index into them accordingly
	int ix = threadIdx.x + blockIdx.x*blockDim.x ;
	int iy = threadIdx.y + blockIdx.y*blockDim.y ;
	int idx = ix*ny + iy ; //iy*ny + ix previously with <= instead of =
	if( (ix<nx) && (iy<ny) )
		C[idx] = A[idx] + B[idx] ;
}

void initData(float *M, long x, long y, int flag ){ //remove and put it in main assigining values in a single lool
	
	if(flag)
	{	
		printf("A\n");
		for (int i=0;i<x;i++){
		        for (int j=0;j<y;j++){
			    M[i*y+j] = (float)(i+j)/3.0;
			    //printf("%f ",M[i*y+j]);
	                }
			//printf("\n");
		}
	}
	else
	{
		printf("B\n");
		for (int i=0;i<x;i++){
			for (int j=0;j<y;j++){
				M[i*y+j] = (float)3.14*(i+j) ;
				//printf("%f ",M[i*y+j]);
			}
			//printf("\n");
		}
	
	}
}

int main( int argc, char *argv[] ) {
	// get program arguments
	
	if (argc!=3){
		printf("Fail");
		exit(1);
		//printf("Fail");
	}
	
	int nx = atoi( argv[1] ) ; // should check validity
	int ny = atoi( argv[2] ) ; // should check validity
	int noElems = nx*ny ;
	int bytes = noElems * sizeof(float) ;

	// but you may want to pad the matrices…
	// alloc memory host-side

	float *h_A = (float *) malloc( bytes ) ;
	float *h_B = (float *) malloc( bytes ) ;
	float *h_hC = (float *) malloc( bytes ) ; // host result
	float *h_dC = (float *) malloc( bytes ) ; // gpu result
	
	// init matrices with random data
	initData(h_A,nx,ny,1); initData(h_B,nx,ny,0);
	// alloc memory dev-side
	float *d_A, *d_B, *d_C ;
	cudaMalloc( (void **) &d_A, bytes ) ;
	cudaMalloc( (void **) &d_B, bytes ) ;
	cudaMalloc( (void **) &d_C, bytes ) ;
	double timeStampA = getTimeStamp() ;
	//transfer data to dev
	
	cudaMemcpy( d_A, h_A, bytes, cudaMemcpyHostToDevice ) ;
	cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice ) ;
	// note that the transfers would be twice as fast if h_A and h_B
	// matrices are pinned
	double timeStampB = getTimeStamp() ;
	// invoke Kernel
	dim3 block( 16, 16) ; // you will want to configure this
	dim3 grid( (nx + block.x-1)/block.x, (ny + block.y-1)/block.y) ;
	printf("%d\n",(ny+block.y-1)/block.y);
	f_addmat<<<grid, block>>>( d_A, d_B, d_C, nx, ny ) ;
	cudaDeviceSynchronize() ;
	double timeStampC = getTimeStamp() ;
	//copy data back
	cudaMemcpyAsync(h_dC, d_C, bytes, cudaMemcpyDeviceToHost);
	//learn how to comment and uncomment in one go
	/*
	printf("C\n");
	for(int i=0;i<nx;i++){
		for(int j=0;j<ny;j++){
			//printf("%f ",h_dC[i*ny+j]);
		}
		//printf("\n");
	}
	*/

	double timeStampD = getTimeStamp() ;
	
	//for(int i=0; i<nx; i++){
	//	for(int j=0; j<ny; j++){
	//		printf("%f ",h_dC[i*ny+j]);
	//	}
	//	printf("\n");
	//}

	// free GPU resources
	cudaFree( d_A ) ; cudaFree( d_B ) ; cudaFree( d_C ) ;
	cudaDeviceReset() ;
	// check result
	printf("%f %f %f %f\n",(timeStampD-timeStampA),(timeStampB-timeStampA),(timeStampC-timeStampB),(timeStampD-timeStampC));
	h_addmat( h_A, h_B, h_hC, nx, ny ) ;
	int flag = 0;
	for(int i=0;i<nx;i++){
		for(int j=0;j<ny;j++){
			if(h_hC[i*ny+j] != h_dC[i*ny+j])
				flag=1;
		}
	
	}	
	printf("\n %d \n",flag);

	// print out results
}

