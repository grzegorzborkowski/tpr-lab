#include <iostream>
#include <fstream>
#include <cmath>
#include <sys/time.h>


void checkErrors(char *label)
{
// we need to synchronise first to catch errors due to
// asynchroneous operations that would otherwise
// potentially go unnoticed
cudaError_t err;
err = cudaThreadSynchronize();
if (err != cudaSuccess)
{
char *e = (char*) cudaGetErrorString(err);
fprintf(stderr, "CUDA Error: %s (at %s)\n", e, label);
}
err = cudaGetLastError();
if (err != cudaSuccess)
{
char *e = (char*) cudaGetErrorString(err);
fprintf(stderr, "CUDA Error: %s (at %s)\n", e, label);
}
}

double get_time()
{  struct timeval tim;
  cudaThreadSynchronize();
  gettimeofday(&tim, NULL);
  return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}

// GPU kernel
__global__ void copy_array(float *u, float *u_prev, int N)
{
        int i = threadIdx.x;
        int j = threadIdx.y;
        int I = blockIdx.y*BSZ*N + blockIdx.x*BSZ + j*N + i;
        if (I>=N*N){return;}    
        u_prev[I] = u[I];

}

__global__ void update (float *u, float *u_prev, int N, float h, float dt, float alpha)
{
	// Setting up indices
	int i = threadIdx.x;
	int j = threadIdx.y;
	int I = blockIdx.y*BSZ*N + blockIdx.x*BSZ + j*N + i;
	
	if (I>=N*N){return;}	

	__shared__ float u_prev_sh[BSZ][BSZ];

	u_prev_sh[i][j] = u_prev[I];
	
	__syncthreads();
	
	bool bound_check = ((I>N) && (I< N*N-1-N) && (I%N!=0) && (I%N!=N-1)); 
	bool block_check = ((i>0) && (i<BSZ-1) && (j>0) && (j<BSZ-1));

	// if not on block boundary do 
	if (block_check)
	{	u[I] = u_prev_sh[i][j] + alpha*dt/h/h * (u_prev_sh[i+1][j] + u_prev_sh[i-1][j] + u_prev_sh[i][j+1] + u_prev_sh[i][j-1] - 4*u_prev_sh[i][j]);
	}
	// if not on boundary 
	else if (bound_check) 
	//if (bound_check) 
	{	u[I] = u_prev[I] + alpha*dt/(h*h) * (u_prev[I+1] + u_prev[I-1] + u_prev[I+N] + u_prev[I-N] - 4*u_prev[I]);
	}
	
	// Boundary conditions are automatically imposed
	// as we don't touch boundaries
}

int main(int argc, char **argv)
{
	// Allocate in CPU
	if (argc < 3) {
            printf("Invalid number of arguments");
            return;
        }
    	// Allocate in CPU
    int N = atoi(argv[1]);
    int BLOCKSIZE = BSZ;

	cudaSetDevice(0);

	float xmin 	= 0.0f;
	float xmax 	= 3.5f;
	float ymin 	= 0.0f;
	//float ymax 	= 2.0f;
	float h   	= (xmax-xmin)/(N-1);
	float dt	= 0.00001f;	
	float alpha	= 0.645f;
	float time 	= 0.4f;

	int steps = (int) ceil(time/dt);
	steps = 100;
	int I;

	float *x  	= new float[N*N]; 
	float *y  	= new float[N*N]; 
	float *u  	= new float[N*N];
	float *u_prev  	= new float[N*N];


	// Generate mesh and intial condition
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = N*j + i;
			x[I] = xmin + h*i;
			y[I] = ymin + h*j;
			u[I] = 0.0f;
			if ( (i==0) || (j==0)) 
				{u[I] = 200.0f;}
		}
	}

	// Allocate in GPU
	float *u_d, *u_prev_d;
	
	cudaMalloc( (void**) &u_d, N*N*sizeof(float));
	cudaMalloc( (void**) &u_prev_d, N*N*sizeof(float));

	// Copy to GPU
	cudaMemcpy(u_d, u, N*N*sizeof(float), cudaMemcpyHostToDevice);

	// Loop 
	dim3 dimGrid(int((N-0.5)/BLOCKSIZE)+1, int((N-0.5)/BLOCKSIZE)+1);
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	double start = get_time();
	for (int t=0; t<steps; t++)
	{	copy_array <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N);
		update <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N, h, dt, alpha);
	}
	double stop = get_time();

	checkErrors("update");
	double elapsed = stop - start;
    // std::cout<<"time = "<<elapsed<<std::endl;
	// std::cout << N << "," << BLOCKSIZE << "," << elapsed << std::endl;
	// Copy result back to host
	cudaMemcpy(u, u_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    /*
	std::ofstream temperature("temperature_shared.txt");
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = N*j + i;
		//	std::cout<<u[I]<<"\t";
			temperature<<x[I]<<"\t"<<y[I]<<"\t"<<u[I]<<std::endl;
		}
		temperature<<"\n";
		//std::cout<<std::endl;
	}

	temperature.close();
    */
	// Free device
	cudaFree(u_d);
	cudaFree(u_prev_d);
}
