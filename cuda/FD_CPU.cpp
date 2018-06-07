#include <iostream>
#include <vector>
#include <stdlib.h>
#include <ctime>
#include <cmath>


using namespace std;

void copy_array(float numbers[],
                float previous[],
                int N) {
                int N_sq = N*N;
    for(int i=0; i<N_sq; i++) {
        previous[i] = numbers[i];
    }
}

void update(float numbers[],
            float previous[],
            int N, int h, int dt, int alpha) {
            int N_sq = N*N;
    for(int i=N; i<N_sq-N; i++) {
                numbers[i] = previous[i] + (alpha*dt) * (previous[i+N] + previous[i-N] +
                + previous[i+1] + previous[i-1] -4*previous[i]);
        }
    }

int main(int argc, char **argv) {
    if (argc < 2) {
        cout << "Invalid number of arguments. Usage ./FD_CPU [SIZE_OF_GRID]";
        return -1;
    }
    int N = atoi(argv[1]);
    float *numbers  = new float[N*N];
    float *previous = new float[N*N];

    int I;
    float xmin 	= 0.0f;
    float xmax 	= 3.5f;
    float ymin 	= 0.0f;
    float h   	= (xmax-xmin)/(N-1);
    float dt	= 0.00001f;
    float alpha	= 0.645f;
    float time 	= 0.4f;
    float x, y;
    int steps = ceil(time/dt);

    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            I = N*j + i;
            x = xmin + h*i;
            y = ymin + h*j;
            if (i==0 || j==0) {

                numbers[i*N+j] = 200.0f;
            } else {
                numbers[i*N+j] = (x+y);
            }

            // cout << numbers[i][j] << " ";
        }
        // cout << endl;
    }
    clock_t begin = clock();
    for (int t=0; t<steps; t++)
    {
       copy_array(numbers, previous, N);
       update(numbers, previous, N, h, dt, alpha);
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << N << "," << elapsed_secs << std::endl;
    delete numbers;
    delete previous;
    return 0;
}
