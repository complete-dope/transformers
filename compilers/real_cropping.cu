// Here I will solve the most common 3d division / data copying problem !!

#include <stdio.h>

template <typename T>
__global__ void CropCudaKernel(
    const T* input,
    const int* crop_centers_ptr , 
    const int* image_size, 
    const int channels, 
    int crop_size, 
    const int num_crops,
    T* crops_ptr
){
    const int crop_id = blockIdx.x; // index of the block inside the grid ( relative indexing of block /  not the global one) 
    // prev threads took how many integers with them  ? prev_threads => 0 to crop_id = crop_id * 3

    const int center_x = crop_centers_ptr[0+crop_id * 3];
    const int center_y = crop_centers_ptr[1+crop_id * 3];
    const int center_z = crop_centers_ptr[2+crop_id * 3];

    for(int id = threadIdx.x ; id < crop_size * crop_size * crop_size *     channels ; id += blockDim.x){
        // per block single thread usage only 
        int id_temp = id;
        const int c = id_temp % channels;
        
        id_temp /= id_temp % channels
    }
}

