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
	int ix = threadIdx.x + blockIdx.x*(blockDim.x) ;
	int iy = threadIdx.y + blockIdx.y*(blockDim.y) ;
	int idx = ix*ny + iy ; //iy*ny + ix previously with <= instead of =
	//printf("Thread %d %d\n",ix,iy);
	if( (ix<nx) && (iy<ny) ){
		C[idx] = A[idx] + B[idx] ;
		//printf("Thread %d %d\n",ix,iy);
	}
}

void initData(float *M, int x, int y, int width, int flag ){ //remove and put it in main assigining values in a single lool
	
	if(flag)
	{	
		printf("A\n");
		for (int i=0;i<x;i++){
		        for (int j=0;j<y;j++){
			    M[i*width+j] = (float)(i+j)/3.0;
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
				M[i*width+j] = (float)3.14*(i+j) ;
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
	int mx=0,my=0;
	//if((nx%16) != 0){
	//	mx = 0;
	//	//mx = 16 - (nx%16);
	//}
	if((ny%1024) != 0){ 
                my = 1024 - (ny%1024);
		
        }
	int noElems = (nx+mx)*(ny+my) ;
	
	printf ("%d %d %d %d \n",(nx*ny),(noElems),mx,my);
	
	int bytes = noElems * sizeof(float) ;

	// but you may want to pad the matrices…
	// alloc memory host-side

	//float *h_A = (float *) malloc( bytes ) ;
	//float *h_B = (float *) malloc( bytes ) ;
	float *h_hC = (float *) malloc( bytes ) ; // host result
	//float *h_dC = (float *) malloc( bytes ) ; // gpu result
	
	// init matrices with random data
	//initData(h_A,nx,ny,1); initData(h_B,nx,ny,0);
	// alloc memory dev-side
	
	float *d_A, *d_B, *d_C ;
	cudaMalloc( (void **) &d_A, bytes ) ;
	cudaMalloc( (void **) &d_B, bytes ) ;
	cudaMalloc( (void **) &d_C, bytes ) ;
	
	float *h_Ap, *h_Bp, *h_dCp;
	cudaMallocHost( (float **) &h_Ap, bytes ) ;
	cudaMallocHost( (float **) &h_Bp, bytes ) ;
	cudaMemset(h_Ap,0,bytes);
	cudaMemset(h_Bp,0,bytes);
	initData(h_Ap,nx,ny,ny+my,1); initData(h_Bp,nx,ny,ny+my,0);
	for(int i=0;i<(nx+mx);i++){
                for(int j=0;j<(ny+my);j++){
                        //if(h_hC[i*(ny+my)+j] != h_dCp[i*(ny+my)+j])
                        //printf("%d ",j);
                        //printf("%f %f %d %d\n",h_Ap[i*(ny+my)+j],h_Bp[i*(ny+my)+j],i,j);
                        //if(h_hC[i*(ny+my)+j] != h_dCp[i*(ny+my)+j])
                        //        flag=1;
                }
                //printf("\n");

        }

	cudaMallocHost( (float **) &h_dCp, bytes ) ;
	cudaMemset(h_dCp,0,bytes);
	
	double timeStampA = getTimeStamp() ;
	//transfer data to dev
	cudaMemcpy( d_A, h_Ap, bytes, cudaMemcpyHostToDevice ) ;
	cudaMemcpy( d_B, h_Bp, bytes, cudaMemcpyHostToDevice ) ;
	// note that the transfers would be twice as fast if h_A and h_B
	// matrices are pinned
	
	double timeStampB = getTimeStamp() ;
	// invoke Kernel
	dim3 block( 1, 1024) ; // you will want to configure this
	dim3 grid( (nx+block.x-1)/block.x, (ny+my)/block.y) ; //(ny+block.y-1)/block.y)
	printf("Grid %d %d \n",(nx+mx)/block.x,(ny+my)/block.y);
	f_addmat<<<grid, block>>>( d_A, d_B, d_C, nx, ny+my ) ;
	cudaDeviceSynchronize() ;
	
	double timeStampC = getTimeStamp() ;
	//copy data back
	cudaMemcpyAsync(h_dCp, d_C, bytes, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize() ;
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
	//cudaFreeHost(h_Ap); cudaFreeHost(h_Bp); 
	//cudaFreeHost(h_dCp);
	//cudaDeviceReset() ;
	// check result
	printf("%f %f %f %f\n",(timeStampD-timeStampA),(timeStampB-timeStampA),(timeStampC-timeStampB),(timeStampD-timeStampC));
	h_addmat( h_Ap, h_Bp, h_hC, nx+mx, ny+my ) ;
	int flag = 0;
	for(int i=0;i<(nx+mx);i++){
		for(int j=0;j<(ny+my);j++){
			//if(h_hC[i*(ny+my)+j] != h_dCp[i*(ny+my)+j])
			//printf("%d ",j);
			//printf("%f %f %d %d\n",h_hC[i*(ny+my)+j],h_dCp[i*(ny+my)+j],i,j);
			if(h_hC[i*(ny+my)+j] != h_dCp[i*(ny+my)+j])	
				flag=1;
		}
		//printf("\n");
	
	}
	cudaFreeHost(h_Ap); cudaFreeHost(h_Bp);	cudaFreeHost(h_dCp);
        free(h_hC);
	cudaDeviceReset() ;	
	printf("\n %d \n",flag);

}

