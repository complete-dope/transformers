// Lets learn to create a fused softmax function / operation and play with it !
// https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
// Shared memory : This is a on chip memory adn has access speed of 100x faster than local or global memory and this remains shared between threads that are on the same block !!

// There can be cases where we need to share data between the threads and to avoid race condition we have _syncThreads() function that we need to call once we update our data using thread !! this needs to be done only when we are using __shared__ memory !!


// In normal way , the memory first gets written over to a global and then fetched again making few steps here and there !! but when storing this in shared memory that over head is no more a problem !


