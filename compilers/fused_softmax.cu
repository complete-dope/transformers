// Lets learn to create a fused softmax function / operation and play with it !
// https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/


// Shared memory : This is a on chip memory and has access speed of 100x faster than local or global memory and this remains shared between threads that are on the same block !!

// There can be cases where we need to share data between the threads and to avoid race condition we have _syncThreads() function that we need to call once we update our data using thread !! this needs to be done only when we are using __shared__ memory !!


// In normal way , the memory first gets written over to a global and then fetched again making few steps here and there !! but when storing this in shared memory that over head is no more a problem !

// Using Shared memory seems to be very easy / intuitive but the complexity lies in execution part !!  

// indexing in C : when we write name[2] , how does it go to the 2nd character  ? what is does it the pointer to the &name + 2*sizeof(char) , this is what we call as indexing !!!

#include <stdio.h> 
#include <stdlib.h>
#include <time.h> 
#include <math.h> 
#include <float.h>
#include <cuda.h>

__global__ void fused_softmax(float *logits, int N, int M){
    extern __shared__ float s[]; // This is L1 cache
    
    printf("\n%p\n" , s);
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int numCols = M;
    int numRows = N;
    int rowStart = row * numCols;
    // softmax operation -> e^x / sum(e^x)  , x being all the elements !!!
    

    // Maximum partExponential part
    float max_val = -FLT_MAX;
    for(int i =tid;i<M;i+=blockDim.x){ // this loops goes as : 0,  blockDim.x , 2*blockDim.x ..etc
        float val = logits[rowStart + i];
        if(val > max_val){
            max_val = max(max_val , val);
        }
    }
    s[tid] = max_val;
    __syncthreads(); // if we have 6 threads per block then this 's' will be of length 5 !!
    

    // stored the max value in the S , now comes the amazing reduction algo !!
    for(int si = s.length/2 ; si>0 ; si>>=1){
	if(tid<si){
	    float other = s[tid+si];
	    if(other > s[tid]){
		s[tid] = other;    
	    }
	}
	__syncthreads();
    }
    max_val = s[0];
    __syncthreads();


    // Step 2: exp val + sum them 

    float sum = 0.0f;
    for (int i = tid; i<numCols; i+=blockDim.x){
	float exp_val = expf(logits[rowStart+i] - max_val);
	logits[rowStart+i] = exp_val; 
	sum += exp_val;
    }
    s[tid] = sum;
    __syncthreads();

    // Reduction to get SUM 

    for(int si = s/2 ;si>0;si>>=1){
	if(tid <si){
	    float other = s[tid+si];
	    s[tid] += other; 
	}
	__syncthreads();
    }
    float sum_exp = s[0];
    __syncthreads();

    // Step 3: find the softmax  

    // now take overall softmax !! 
    for(int i = tid; i<numCols ;i+=blockDim.x){
	int idx = rowStart + tid;
	logits[idx] = logits[idx] / sum_exp;
    }

}


int main(){
    srand(42); // seed the random value 
    int N = 100;
    int M = 100;


    float* logits = (float*)malloc(N * M * sizeof(float));
    for(int i = 0 ; i< N*M; i++){
	logits[i] = (float)(rand())/ RAND_MAX;
    }
    float* cudaMemlogits;
    float** cudaMemlogitsAddress = &cudaMemlogits;
    cudaMalloc(cudaMemlogitsAddress , (N*M*sizeof(float)));
    
    cudaMemcpy(cudaMemlogits , logits , N*M*sizeof(float) , cudaMemcpyHostToDevice);
    
    int cudaBlocks = N;
    int cudaThreads = M;
    int thirdParam = M * sizeof(float); // third param to pass if you want to pass in dynamic value for shared array 
    fused_softmax<<<cudaBlocks , cudaThreads, thirdParam>>>(logits ,N,M);  

    cudaMemcpy(logits , cudaMemlogits, N*M*sizeof(float) , cudaMemcpyDeviceToHost);
    
    for(int i =0;i<N;i++)
	for(int j = 0; j< M;j++){
	    int idx = i*N+ j;
	    printf("%.2f", logits[idx]);  
	}
    printf("\n");
    }

    cudaFree(cudaMemlogits);
    free(logits);

}











