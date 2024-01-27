// This program implements a 1D convolution using CUDA,
// and stores the mask in constant memory, and loads
// reused values into shared memory (scratchpad)
// By: Nick from CoffeeBeforeArch

#include <cassert>
#include <cstdlib>
#include <iostream>
#include "../tools/common.cuh"

#define MASK_LENGTH 7

// 常量内存
__constant__ int mask[MASK_LENGTH];

/*
    Arguments:
        array   = padded array
        result  = result array
        n       = number of elements in array
    每个线程只写一个元素，但是某些线程可能会多读
*/

__global__ void tiled1DConvKernel(int *array, int *result, int n) {
    // 每个线程
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 共享内存
    extern __shared__ int s_array[];

    int r = MASK_LENGTH / 2;

    int d = 2 * r;
    // 每个block需要读多少数据
    int n_padded = blockDim.x + d;

    // 
    int offset = threadIdx.x + blockDim.x;

    int g_offset = blockDim.x * blockIdx.x + offset;

    // 第一次读数据
    s_array[threadIdx.x] = array[tid];

    // 某些线程会多读一些数据
    if (offset < n_padded) {
        s_array[offset] = array[g_offset];
    }
    __syncthreads();

    int temp = 0;

    for (int j = 0; j < MASK_LENGTH; j++) {
        temp += s_array[threadIdx.x + j] * mask[j];
    }

    result[tid] = temp;
}

void verify_result(int *array, int *mask, int *result, int n) {
    int temp;
    for (int i = 0; i < n; i++) {
        temp = 0;
        for (int j = 0; j < MASK_LENGTH; j++) {
            temp += array[i + j] * mask[j];
        }
        assert(temp == result[i]);
    }
}

int main() {
    setGPU();

    int n = 1 << 20;
    int bytes_n = n * sizeof(int);
    size_t bytes_m = MASK_LENGTH * sizeof(int);

    int r = MASK_LENGTH / 2;
    int n_p = n + r * 2;

    size_t bytes_p = n_p * sizeof(int);

    int *h_array = new int[n_p];

    // 初始化
    for (int i = 0; i < n_p; i++) {
        if ((i < r) || (i >= (n + r))) {
            h_array[i] = 0;
        } else {
            h_array[i] = rand() % 100;
        }
    }

    int *h_mask = new int[MASK_LENGTH];
    for (int i = 0; i < MASK_LENGTH; i++) {
        h_mask[i] = rand() % 10;
    }

    int *h_result = new int[n];

    int *d_array, *d_result;
    cudaMalloc(&d_array, bytes_p);
    cudaMalloc(&d_result, bytes_n);

    cudaMemcpy(d_array, h_array, bytes_p, cudaMemcpyHostToDevice);

  
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);

    int THREADS = 256;
    int GRID = (n + THREADS - 1) / THREADS;

    size_t SHMEM = (THREADS + r * 2) * sizeof(int);

    tiled1DConvKernel<<<GRID, THREADS, SHMEM>>>(d_array, d_result, n);

    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    verify_result(h_array, h_mask, h_result, n);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    delete[] h_array;
    delete[] h_result;
    delete[] h_mask;
    cudaFree(d_result);

    return 0;
}