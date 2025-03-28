// [Compiling , assembling and linking]  https://www.youtube.com/watch?v=N2y6csonII4


#include <stdio.h> 
#include <stdlib.h> 
#include <time.h>

using namespace std;

__global__ void conv2d_kernel(
	float *image_matrix, 
	float *kernel_matrix, 
	float *output_matrix, 
	int stride,
	int image_size_height, 
	int image_size_width, 
	int kernel_size_height, 
	int kernel_size_width, 
	int output_size_height, 
	int output_size_width
){
	
	int output_x = blockDim.x * blockIdx.x + threadIdx.x; // thread Index x : 14  
	int output_y = blockDim.y * blockIdx.y + threadIdx.y; // thread Index y : 13 
	// This translates to this piece of work will be done using the thread at index (14,13)
	
	// Bounds if this thread exists or not ?? ( to prevent recursion !! NICE !! We are so back to leetcoding)
	if(output_x >= output_size_width || output_y >= output_size_height){
		return;
	}

	float sum = 0.0f;
	for(int ky=0; ky<kernel_size_width; ky++){ 
		for(int kx=0; kx<kernel_size_width; kx++){
			int image_x = output_x * stride + kx;
			int image_y = output_y * stride + ky;
			
			if(image_x >= image_size_height || image_y >= image_size_width){
				continue;
			}

			// conv
			sum += kernel_matrix[ky * kernel_size_width + kx] * image_matrix[image_y * image_size_width + image_x];

		}
	}
	output_matrix[output_y * output_size_width + output_x] = sum;

	// This is simple what they did is, ki dekho mere pass a block of threads and mai chahata hu ki hr thread in that block of gpu ek output kernel ki value nikale .. and iske liye maine pehle coordinates nikal liye of the threads and kyunki har thread should correspond to that same coordinate in the output matrix .. and then conv ka logic laga diya !!    
	return;
}

// CPU convolution 
void conv2d_cpu(
	float *image_matrix,
	float *kernel_matrix,
	float *output_matrix,
	int stride,
	int image_size_height,
	int image_size_width,
	int kernel_size_height,
	int kernel_size_width,
	int output_size_height,
	int output_size_width
) {
	for (int oy = 0; oy < output_size_height; oy++) {
		for (int ox = 0; ox < output_size_width; ox++) {
			float sum = 0.0f;
			for (int ky = 0; ky < kernel_size_height; ky++) {
				for (int kx = 0; kx < kernel_size_width; kx++) {
					int image_x = ox * stride + kx;
					int image_y = oy * stride + ky;
				
					if (image_x >= image_size_width || image_y >= image_size_height) {
						continue;
					}
				
					sum += kernel_matrix[ky * kernel_size_width + kx] * 
									image_matrix[image_y * image_size_width + image_x];
				}
			}
			output_matrix[oy * output_size_width + ox] = sum;
		}
	}
	return;
}

void printOutputMatrix(float *output_matrix, int out_size_height, int out_size_width) {
	printf("Output Matrix is %d x %d \n", out_size_height, out_size_width);
	for(int i = 0;i<out_size_height;i++){
		for(int j =0;j<out_size_width;j++){
			int idx = i*out_size_height + j;
			printf("%.2f,", output_matrix[idx]);
		}
		printf("\n");
	}
}

int main(){
	srand(42); // seed the time 

    int image_size_height = 25120;
    int image_size_width = 25120;

    int kernel_size_height = 3;
    int kernel_size_width = 3;

	int stride = 1;

	int out_size_height = (image_size_height - kernel_size_height)/stride + 1;
    int out_size_width = (image_size_width - kernel_size_width)/stride + 1; 


    time_t start_time , end_time;
	
    float* image_matrix = (float*)malloc(image_size_height * image_size_width * sizeof(float)); /// we need this later so that once we get the computed part back from GPU it can work ! 
    float* kernel_matrix = (float*)malloc(kernel_size_height * kernel_size_width * sizeof(float));
	// calloc , allocates memory and allocates it to value of 0.  
    float* output_matrix = (float*)calloc(out_size_height * out_size_width ,  sizeof(float));

	for(int i =0; i< image_size_height; i++){
		for(int j=0; j<image_size_width; j++){
			image_matrix[i * image_size_width + j] = (float)rand()/(float)(RAND_MAX);
		}
	}

	for(int i =0; i< kernel_size_height; i++){
		for(int j=0; j<kernel_size_width; j++){
			kernel_matrix[i * kernel_size_width + j] = (float)rand()/(float)(RAND_MAX);		
		}
	}

	// make the CPU call first 
	time(&start_time); 
	// conv2d_cpu(image_matrix, kernel_matrix, output_matrix, stride, image_size_height, image_size_width, kernel_size_height, kernel_size_width, out_size_height, out_size_width);
	time(&end_time);

    double elapsed_time = difftime(end_time, start_time);
    printf("Elapsed time: %f seconds\n", elapsed_time);


	printf("CPU output \n");
	// printOutputMatrix(output_matrix , out_size_height, out_size_width);

	printf("GPU COMPUTATION STARTED \n");

	float* gpu_image_matrix;
	float* gpu_kernel_matrix;
	float* gpu_output_matrix;
	
	// This all is on CPU !! BRING THIS SHIT TO GPU !!
	cudaMalloc(&gpu_image_matrix, image_size_height * image_size_width * sizeof(float));
	cudaMalloc(&gpu_kernel_matrix, kernel_size_height * kernel_size_width * sizeof(float));
	cudaMalloc(&gpu_output_matrix, out_size_height * out_size_width * sizeof(float));

	//  CANT I DO DIRECT ACCESS TO THIS CUDA ALLOCATED MEMORY ? I HAVE GPU WITH ME DO I NEED A BUS THAT BRING DATA FROM GPU TO CPU AND THEN ACCESS ? WHAT DOES CUDA MEMCPY DO INTERNALLY ??? 
	// --> not different architectures in both cpu and gpu, so we need to copy it to GPU.   

	// Do we really require to copy it ? can this step should be avoided in testing .... All real life applications require this as we dont send random values at that time
	cudaMemcpy(gpu_image_matrix, image_matrix, image_size_height * image_size_width * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_kernel_matrix, kernel_matrix, kernel_size_height * kernel_size_width * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_output_matrix, output_matrix, out_size_height * out_size_width * sizeof(float), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((out_size_width + 15) / 16, (out_size_height + 15) / 16);
	// conv2d_kernel<<<numBlocks, threadsPerBlock>>>(...);

	time(&start_time);
	conv2d_kernel<<<numBlocks , threadsPerBlock>>>(gpu_image_matrix, gpu_kernel_matrix, gpu_output_matrix, stride, image_size_height, image_size_width, kernel_size_height, kernel_size_width, out_size_height, out_size_width);
	// conv2d_kernel<<<dim3(26,26), dim3(16, 16)>>>(gpu_image_matrix, gpu_kernel_matrix, gpu_output_matrix, stride, image_size_height, image_size_width, kernel_size_height, kernel_size_width, out_size_height, out_size_width);

	cudaMemcpy(image_matrix, gpu_image_matrix , image_size_height*image_size_width*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(kernel_matrix, gpu_kernel_matrix , kernel_size_height*kernel_size_width*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(output_matrix, gpu_output_matrix , out_size_height*out_size_width*sizeof(float), cudaMemcpyDeviceToHost);

	time(&end_time);
	elapsed_time = difftime(end_time, start_time);
	printf("Elapsed time: %f seconds\n", elapsed_time);


	printf("Convolution matrix is : \n");

	// printOutputMatrix(output_matrix , out_size_height, out_size_width);

	cudaFree(gpu_image_matrix);
	cudaFree(gpu_output_matrix);
	cudaFree(gpu_kernel_matrix);

	free(image_matrix);
	free(kernel_matrix);
	free(output_matrix);

	return 0;

}


// nvcc -c conv2d.cu -o conv2d.o ##!##This is the assembly file ? 
// nvcc conv2d.o -o conv2d ##!## This is the executable file

