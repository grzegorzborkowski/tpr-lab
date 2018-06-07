/*** Calculating a derivative with CD ***/
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


texture<float, 2> tex_u;
texture<float, 2> tex_u_prev;

// GPU kernels
__global__ void copy_kernel (float *u, float *u_prev, int N, int BSZ, int N_max)
{
	// Setting up indices
	int i = threadIdx.x;
	int j = threadIdx.y;
	int x = i + blockIdx.x*BSZ;
	int y = j + blockIdx.y*BSZ;
	int I = x + y*N_max;
	

	//if (I>=N*N){return;}	
	//if ((x>=N) || (y>=N)){return;}	
	float value = tex2D(tex_u, x, y);

	u_prev[I] = value;

}

__global__ void update (float *u, float *u_prev, int N, float h, float dt, float alpha, int BSZ, int N_max)
{
	// Setting up indices
	int i = threadIdx.x;
	int j = threadIdx.y;
	int x = i + blockIdx.x*BSZ;
	int y = j + blockIdx.y*BSZ;
	int I = x + y*N_max;
	
	//if (I>=N*N){return;}	
	//if ((x>=N) || (y>=N)){return;}	
	

	float t, b, r, l, c;
	c = tex2D(tex_u_prev, x, y);	
	t = tex2D(tex_u_prev, x, y+1);	
	b = tex2D(tex_u_prev, x, y-1);	
	r = tex2D(tex_u_prev, x+1, y);	
	l = tex2D(tex_u_prev, x-1, y);


	//if ( (I>N) && (I< N*N-1-N) && (I%N!=0) && (I%N!=N-1))
	if ( (x!=0) && (y!=0) && (x!=N-1) && (y!=N-1))
	{	u[I] = c + alpha*dt/h/h * (t + b + l + r - 4*c);	
	}
}

int main(int argc, char **argv)
{
	// Allocate in CPU
	// int N = 128;		// For textures to work, N needs to be a multiple of
	// int BLOCKSIZE = 16;	// 32. As I will be using BLOCKSIZE to be a multiple of 8
						// I'll just look for the closest multiple of BLOCKSIZE (N_max)

    if (argc < 3) {
            printf("Invalid number of arguments");
            return;
        }
    	// Allocate in CPU
    int N = atoi(argv[1]);
    int BLOCKSIZE = atoi(argv[2]);

	int N_max = (int((N-0.5)/BLOCKSIZE) + 1) * BLOCKSIZE;

	float xmin 	= 0.0f;
	float xmax 	= 3.5f;
	float ymin 	= 0.0f;
	//float ymax 	= 2.0f;
	float h   	= (xmax-xmin)/(N-1);
	float dt	= 0.00001f;	
	float alpha	= 0.645f;
	float time 	= 0.4f;

	int steps = (int)ceil(time/dt);
	steps = 100;
	int I, J;

	float *x  	= new float[N*N]; 
	float *y  	= new float[N*N]; 
	float *u  	= new float[N_max*N_max];
	float *u_prev  	= new float[N*N];

	// Initialize
	for (int j=0; j<N_max; j++)
	{	for (int i=0; i<N_max; i++)
		{	I = N_max*j + i;
			u[I] = 0.0f;
			if ( ((i==0) || (j==0)) && (j<N) && (i<N)) 
				{u[I] = 200.0f;}
		}
	}	

	// Generate mesh and intial condition
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = N*j + i;
			x[I] = xmin + h*i;
			y[I] = ymin + h*j;
		}
	}

	// Allocate in GPU
	float *u_d, *u_prev_d;
	
	cudaMalloc( (void**) &u_d, N_max*N_max*sizeof(float));
	cudaMalloc( (void**) &u_prev_d, N_max*N_max*sizeof(float));

	// Bind textures
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, tex_u, u_d, desc, N_max, N_max, sizeof(float)*N_max);
	cudaBindTexture2D(NULL, tex_u_prev, u_prev_d, desc, N_max, N_max, sizeof(float)*N_max);

	// Copy to GPU
	cudaMemcpy(u_d, u, N_max*N_max*sizeof(float), cudaMemcpyHostToDevice);

	// Loop 
	dim3 dimGrid(int((N_max-0.5)/BLOCKSIZE)+1, int((N_max-0.5)/BLOCKSIZE)+1);
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	double start = get_time();
	for (int t=0; t<steps; t++)
	{	// The transfer of u to u_prev needs to be in separate kernel
		// as it's read only
		copy_kernel <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N, BLOCKSIZE, N_max);
		update <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N, h, dt, alpha, BLOCKSIZE, N_max);

	}
	double stop = get_time();
	checkErrors("update");
	
	double elapsed = stop - start;

	// std::cout<<"time = "<<elapsed<<std::endl;
    // std::cout << N << "," << BLOCKSIZE << "," << elapsed << std::endl;
	// Copy result back to host
	cudaMemcpy(u, u_d, N_max*N_max*sizeof(float), cudaMemcpyDeviceToHost);
    /*
	std::ofstream temperature("temperature_texture.txt");
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = N*j + i;
			J = N_max*j + i;
	//		std::cout<<u[J]<<"\t";
			temperature<<x[I]<<"\t"<<y[I]<<"\t"<<u[J]<<std::endl;
		}
		temperature<<"\n";
	//	std::cout<<std::endl;
	}

	temperature.close();
    */
	// Free device
	cudaUnbindTexture(tex_u);
	cudaUnbindTexture(tex_u_prev);
	cudaFree(u_d);
	cudaFree(u_prev_d);
}
