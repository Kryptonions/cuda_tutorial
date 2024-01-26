#include <iostream>
#include "../tools/common.cuh"

#define MASK_WIDTH 5
#define RADIUS MASK_WIDTH / 2
// 每个tile需要写的数量
#define O_TILE_WIDTH 1020   // 每个tile需要写的数量
// 每个tile需要读的数量，需要额外读取更多数据，所以block内的线程数多于每个block需要写的量，一部分线程只负责读数据
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH - 1) 

// mask数据在常量内存里
__constant__ int M[MASK_WIDTH];


__global__ void tiled1DConvKernel(int *N, int *P, int width) {
    // 共享内存
    __shared__ int Ns[BLOCK_WIDTH];  

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
        for (j = 0; j < MASK_WIDTH; j++) {
            output += M[j] * Ns[j + tx];
        }
    }
    //边界检查
    if (index_o < width) {
        P[index_o] = output;
    }
}

int main(void) {
    setGPU();

    // 数组长度
    int width = 1 << 10;

    int *A, *P;
    int nBytes = width * sizeof(int);
    int mBytes = MASK_WIDTH * sizeof(int);

    // 内存托管
    cudaMallocManaged((void**)&A, nBytes);
    cudaMallocManaged((void**)&P, nBytes);

    // 初始化数据
    for (int i = 0; i < width; i++) {
        A[i] = 1;
    }

    // 初始化mask
    int *h_mask = new int[MASK_WIDTH];
    for (int i = 0; i < MASK_WIDTH; i++) {
        h_mask[i] = 1;
    }
    // 将数据copy到常量内存
    cudaMemcpyToSymbol(M, h_mask, mBytes);

    // 每个block需要更多线程读数据
    dim3 blockSize(BLOCK_WIDTH, 1);
    // 每个block需要写O_TILE_WIDTH个数据
    dim3 gridSize((width + O_TILE_WIDTH - 1) / O_TILE_WIDTH, 1);
    // 执行kernel
    tiled1DConvKernel <<< gridSize, blockSize >>>(A, P, width);
    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();

    for (int i = 0; i < 10; i++) {
        printf("id=%d, P=%d", i, P[i]);
    }

    return 0;
}