// Lets build a cropping algo and try to run that on GPU device and make sure it works fine !!

// kernel : the function that we write ( instead of function we call it kernel ..repeat after me kernel)
// Threads : cuda runs threads ( like cpu has threads ) .. the functions that give actual data to the input 
// Blocks : group of threads 
// grid : group of blocks


// Okay , so are threads placed in 2d or are arranged in 1d ... , that is how we want to code it !! that is not an hardware layer or something .. its how we define that in code  

#include <stdio.h>

// global means function will be compiled using nvcc ( not the gcc )   
__global__ void saxpy(int n , float a, float* x, float* y){
    // need to do a*x + y 
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // blockIdx: this tracks which block we are currently in , the code execution -> 2d / 3d array of blocks ( and this tell which block we are in )
    // blockDim: number of threads per block
    // threadIdx: this tracks which thread we are currently executing .. built-in variable that stores the index of the thread within the block
    int i = blockDim.x * blockIdx.x  + threadIdx.x; //Assuming 1D structure of code 
    if(i<n){
        y[i] = a*x[i] + y[i];
    }
}

// HOST : cpu 
// DEVICE : gpu

int main(){
    int N = 1<<20; // 2^20
    float *x, *y, *d_x , *d_y;
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));


    for(int i=0; i<N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // x -> [1.0 ,1.0, 1.0, 1.0 ...]
    // y -> [2.0 ,2.0, 2.0, 2.0 ...]

    cudaMemcpy(d_x , x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y , y, N*sizeof(float), cudaMemcpyHostToDevice);

    // perform saxpy on 1M elements 
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
    // 256 -> threads per blocks (these are threads laid out in 1D)
    // (N+255)/256 -> no of blocks required to do the processing 

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for(int i=0; i<N; i++){
        maxError = max(maxError, abs(y[i]-4.0f));
    }
    printf("Max error: %f\n", maxError);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
    return 0;
}