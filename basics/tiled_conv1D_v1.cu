#include <iostream>
#include <cassert>
#include <cstdlib>
#include "../tools/common.cuh"

#define MASK_WIDTH 5
#define RADIUS MASK_WIDTH / 2
// 每个tile需要写的数量
#define O_TILE_WIDTH 256
// 每个tile需要读的数量，需要额外读取更多数据，所以block内的线程数多于每个block需要写的量，一部分线程只负责读数据
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH - 1) 

// mask数据在常量内存里
__constant__ int M[MASK_WIDTH];


__global__ void tiled1DConvKernel(int *N, int *P, int width) {
    // 共享内存
    //__shared__ int Ns[BLOCK_WIDTH];  
    extern __shared__ int Ns[];
    // 当前线程负责写的位置
    int index_o = blockIdx.x * O_TILE_WIDTH + threadIdx.x;
    // 当前线程负责读的位置
    int index_i = index_o - RADIUS;
    int tx = threadIdx.x;
    int output = 0;
    if ((index_i >= 0) && (index_i < width)) {
        Ns[tx] = N[index_i];
    } else {
        Ns[tx] = 0;
    }
    //等待所有线程load数据到共享内存
    __syncthreads();
    //只有前O_TILE_WIDTH个线程负责写数据，其余空闲
    if (tx < O_TILE_WIDTH) {
        output = 0;
        for (int j = 0; j < MASK_WIDTH; j++) {
            output += M[j] * Ns[j + tx];
        }
        P[index_o] = output;
    }
}

void verify_result(int *array, int *result, int *mask, int n) {
    // pad array with 0s
    int r = MASK_WIDTH / 2;
    int n_p = n + r * 2;
    int *h_array = new int[n_p];
    for (int i = 0; i < n_p; i++) {
        if ((i < r) || (i >= (n + r))) {
            h_array[i] = 0;
        } else {
            h_array[i] = array[i - r];
        }
    }
    int temp;
    for (int i = 0; i < n; i++) {
        temp = 0;
        for (int j = 0; j < MASK_WIDTH; j++) {
            temp += h_array[i + j] * mask[j];
        }
        
        assert(temp == result[i]);
    }
}

int main() {
    setGPU();
    // 数组长度
    int width = 1 << 20;

    int *h_array = new int[width];

    int nBytes = width * sizeof(int);
    int mBytes = MASK_WIDTH * sizeof(int);
    // 内存托管
    //ErrorCheck(cudaMallocManaged((void**)&A, nBytes),  __FILE__, __LINE__);
    //ErrorCheck(cudaMallocManaged((void**)&P, nBytes),  __FILE__, __LINE__);

    // 初始化数据
    for (int i = 0; i < width; i++) {
        h_array[i] = rand() % 100;
    }
    // 初始化mask
    int *h_mask = new int[MASK_WIDTH];
    for (int i = 0; i < MASK_WIDTH; i++) {
        h_mask[i] = rand() % 10;
    }

    int *h_result = new int[width];
    int *d_array, *d_result;
    ErrorCheck(cudaMalloc(&d_array, nBytes), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc(&d_result, nBytes), __FILE__, __LINE__);

    // 将A的元素copy到DRAM
    ErrorCheck(cudaMemcpy(d_array, h_array, nBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    // 将数据copy到常量内存
    cudaMemcpyToSymbol(M, h_mask, mBytes);

    // 每个block需要更多线程读数据
    dim3 blockSize(BLOCK_WIDTH, 1);
    // 每个block需要写O_TILE_WIDTH个数据
    dim3 gridSize((width + O_TILE_WIDTH - 1) / O_TILE_WIDTH, 1);
    // 执行kernel
    size_t SHMEM = (BLOCK_WIDTH + RADIUS * 2) * sizeof(int);
    tiled1DConvKernel<<<gridSize, blockSize, SHMEM>>>(d_array, d_result, width);

    // 将执行结果copy到主机
    ErrorCheck(cudaMemcpy(h_result, d_result, nBytes, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    // 等待所有线程执行完毕
    //cudaDeviceSynchronize();
    // 检查结果
    verify_result(h_array, h_result, h_mask, width);
    std::cout << "COMPLETED SUCCESSFULLY\n";
    delete[] h_array;
    delete[] h_result;
    delete[] h_mask;
    cudaFree(d_result);

    return 0;
}