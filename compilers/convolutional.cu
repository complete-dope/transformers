// Convolutional layer on CPU

//  Grid > Block  > Threads collection > Single Thread 

#include <string.h>
#include <time.h> 
#include <stdio.h>
#include <stdlib.h>

// writing the kernel !! 

struct Matrix_props{
	int height;
	int width;	
};

__global__ void conv_kernel(float* kernel, float* image_matrix , float* output, int kernel_size_height, int kernel_size_width, int image_size_height, int image_size_width, int stride, int output_size_height, int output_size_width){
	// This is the conv kernel !! 
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	// each thread has this kernel loaded onto it !! ( but what about there shared data, need a common cache where they all can update the code !!)
	// As the only thing different here in all kernels are the combo params like the thread declared above so we need to use these to bring in the variability in them !! these are all arranged in 1D 
	
	int start_idx = idx * kernel_size_height * kernel_size_width;
	
	float sum = 0.0f;
	for (int k =0; k< kernel_size_height; k++){
		for (int l =0; l< kernel_size_width; l++){
			// int idx_on_image_matrix = 32; // Fucked up logic 
			int val_on_kernel_array = k * kernel_size_height  + l;
			int array_idx =start_idx + val_on_kernel_array;
			sum += kernel[val_on_kernel_array] * array_idx;
		}
	}
	output[idx] = sum;
	
	return;
}


int main(){
    srand(42); // seed the time 

    int image_size_height = 512;
    int image_size_width = 512;

    int kernel_size_height = 2;
    int kernel_size_width = 2;

    int stride = 2;

    time_t start_time , end_time;
	
    float* image_matrix = (float*)malloc(image_size_height * image_size_width * sizeof(float));
    float* kernel_matrix = (float*)malloc(kernel_size_height * kernel_size_width * sizeof(float));


    for(int i =0;i<image_size_height * image_size_width;i++){
    	image_matrix[i] = 1.0f;
    }	    

    for(int i =0;i<kernel_size_height * kernel_size_width; i++){
		kernel_matrix[i] = (float)rand() / RAND_MAX;
    }

//    memset(image_matrix , 1, sizeof(image_matrix));
//    memset(kernel_matrix, rand(), sizeof(kernel_matrix));

    for (int i =0;i<kernel_size_height;i++){
	    for (int j =0;j<kernel_size_width;j++){
		int idx = i*kernel_size_height + j;
		printf("%f,", kernel_matrix[idx]);
	    }
	    printf("\n");
    }

    printf("\n");
    // kernel function output => (image_size - kernel_size)/s + 1
    int out_size_height = (image_size_height - kernel_size_height)/stride + 1;
    int out_size_width = (image_size_width - kernel_size_width)/stride + 1; 
    float* output_matrix = (float*)calloc(out_size_height * out_size_width ,  sizeof(float));

    start_time = time(NULL);

    printf("Expected size for the output matrix is %d x %d \n\n", out_size_height , out_size_width);
    // basic using CPU (noobs do that)
    for(int i = 0; i<out_size_height;i++){
	    for (int j =0;j<out_size_width;j++){
		    float sum = 0.0f;
		    for(int k = 0; k< kernel_size_height; k++){
			    for (int l = 0 ;l<kernel_size_width;l++){
				    int img_x = i * stride + k;
				    int img_y = j * stride + l;  
					
				    // x-> k , y --> l
				    sum += image_matrix[img_x * image_size_height + img_y] * kernel_matrix[k * kernel_size_height + l];
				   // output_matrix[k][l] = kernel_matrix[x][y] * image_matrix[i][j];
			    }
		    }
		    output_matrix[i * out_size_height + j] = sum;
	    }
    }

    end_time = time(NULL);

    // test if the convolution worked as expected or not !! 
	
    for (int i = 0 ; i< out_size_height;i++){
	    for(int j=0;j< out_size_width;j++){
		    printf("%f,", output_matrix[i* out_size_height + j]);
	    }
	    printf("\n");

    }

    printf("Total time taken for this operation is %f sec\n\n" , end_time - start_time);    
    return 0;
}


