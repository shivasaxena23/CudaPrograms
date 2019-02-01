#include <sys/time.h>
#include <cuda.h>
#include <stdio.h>

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
	
	int ix = threadIdx.x + blockIdx.x*(blockDim.x) ;
	int iy = threadIdx.y + blockIdx.y*(blockDim.y) ;
	int idx = ix*ny + iy ; 

	if( (ix<nx) && (iy<ny) ){
		C[idx] = A[idx] + B[idx] ;
		//printf("Thread %d %d\n",ix,iy);
	}
}

void initData(float *M, int x, int y, int width, int flag ){ 
	
	if(flag)
	{	
		//printf("A\n");
		for (int i=0;i<x;i++){
		        for (int j=0;j<y;j++){
			    M[i*width+j] = (float)(i+j)/3.0;
		        }
		}
	}
	else
	{
		//printf("B\n");
		for (int i=0;i<x;i++){
			for (int j=0;j<y;j++){
				M[i*width+j] = (float)3.14*(i+j) ;
			}
		}
	
	}
}

int main( int argc, char *argv[] ) {
	
	
	if (argc!=3){
		printf("Error: Invalid number of arguments.\n");
		exit(1);
	}
	
	int nx = atoi( argv[1] ) ; // should check validity
	int ny = atoi( argv[2] ) ; // should check validity
	
	if(nx <=0 || ny <=0){
		printf("Error: Dimension lessThanOrEqualto Zero.\n");
                exit(1);
	}

	int my=0;
	
	/*
	if((ny%16) != 0){ 
                my = 16 - (ny%16);		
        }
	*/

	int noElems = (nx)*(ny+my) ;
	int bytes = noElems * sizeof(float) ;
	//printf ("%d %d %d %d \n",(nx*ny),(noElems),mx,my);	

        // GPU and CPU memory Allocations
        
        float *d_A, *d_B, *d_C ;
        cudaMalloc( (void **) &d_A, bytes ) ;
        cudaMalloc( (void **) &d_B, bytes ) ;
        cudaMalloc( (void **) &d_C, bytes ) ;
	float *h_hC = (float *) malloc( bytes ) ; // host result

        float *h_Ap, *h_Bp, *h_dCp;
        cudaMallocHost( (float **) &h_Ap, bytes ) ;
        cudaMallocHost( (float **) &h_Bp, bytes ) ;
        cudaMallocHost( (float **) &h_dCp, bytes ) ;
	//cudaMemset(h_Ap,0,bytes);
        //cudaMemset(h_Bp,0,bytes);
        //cudaMemset(h_dCp,0,bytes);
	
	// init matrices with random data
	initData(h_Ap,nx,ny,ny+my,1); 
	initData(h_Bp,nx,ny,ny+my,0);

	double timeStampA = getTimeStamp() ;

	//transfer data to dev
	cudaMemcpy( d_A, h_Ap, bytes, cudaMemcpyHostToDevice ) ;
	cudaMemcpy( d_B, h_Bp, bytes, cudaMemcpyHostToDevice ) ;
		
	double timeStampB = getTimeStamp() ;
	
	// invoke Kernel
	dim3 block( 16, 16) ; // you will want to configure this
	dim3 grid( (nx+block.x-1)/block.x, (ny+block.y-1)/block.y) ; 
	//printf("Grid %d %d \n",(nx+block.x-1)/block.x,(ny+my)/block.y);
	f_addmat<<<grid, block>>>( d_A, d_B, d_C, nx, ny+my ) ;
	cudaDeviceSynchronize() ;
	
	double timeStampC = getTimeStamp() ;
	
	//copy data back
	cudaMemcpy(h_dCp, d_C, bytes, cudaMemcpyDeviceToHost);
	
	double timeStampD = getTimeStamp() ;
	
	// free GPU resources
	cudaFree( d_A ) ; cudaFree( d_B ) ; cudaFree( d_C ) ;
	
	// CPU Matrix add
	h_addmat( h_Ap, h_Bp, h_hC, nx, ny+my ) ;
	
	// Check results
	int flag = 0;
	for(int i=0;i<(nx);i++){
		for(int j=0;j<(ny+my);j++){
			if(h_hC[i*(ny+my)+j] != h_dCp[i*(ny+my)+j])	
				flag=1;
		}
	}
	
	if (flag == 0){
		printf("%.6f %.6f %.6f %.6f\n",(timeStampD-timeStampA),(timeStampB-timeStampA),(timeStampC-timeStampB),(timeStampD-timeStampC));
	}
	
	
	//free other resourses
	cudaFreeHost(h_Ap); cudaFreeHost(h_Bp);	cudaFreeHost(h_dCp);
        free(h_hC);
	cudaDeviceReset() ;	
	
}

