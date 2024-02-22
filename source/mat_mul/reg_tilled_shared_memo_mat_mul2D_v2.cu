#include <iostream>
#include "../tools/common.cuh"
#include <cuda_profiler_api.h>

using namespace std;

#define THREAD_NUM 16
// 每个thread计算8*8=64个元素，block是16*16=256
#define TM 8
#define TN 8
// 每个thread block负责计算128*128个元素
#define BM THREAD_NUM*8
#define BN THREAD_NUM*8
// tile块的长度
#define BK 8

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

struct Matrix
{
    int width;
    int height;
    float *elements;
};

// 获取矩阵A的(row, col)元素
__device__ float getElement(Matrix *A, int row, int col)
{
    return A->elements[row * A->width + col];
}

// 为矩阵A的(row, col)元素赋值
__device__ void setElement(Matrix *A, int row, int col, float value)
{
	A->elements[row * A->width + col] = value;
}


__global__ void regTilledSharedMemoryMatMulKernel(Matrix *A, Matrix *B, Matrix *C) {
    int K = A->width;                   //A的列数


}
__global__ void regTilledSharedMemoryMatMulKernel(Matrix *A, Matrix *B, Matrix *C, const int M, const int N, const int K) {
    int K = A->width;                   //A的列数

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BM][BK];  // 128 * 8 = 1024，每个thread负责load 1024/256=4，每2个线程load s_a的一行
    __shared__ float s_b[BK][BN];  // 128 * 8 = 1024，每个thread负责load 1024/256=4，每32个线程load s_b的一行

    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;  // tid/2，第几行，因为每2个线程load s_a的一行，每行2*4=8个元素
    int load_a_smem_k = (tid & 1) << 2; // (tid % 2 == 0) ? 0 : 4，第几列
    int load_b_smem_k = tid >> 5;  // tid/32，第几行，因为每32个线程load s_b的一行，每行32*4=128个元素
    int load_b_smem_n = (tid & 31) << 2;// (tid % 32) * 4, 第几列

    int load_a_gmem_m = by * BM + load_a_smem_m;  // A的global memory第几行
    int load_b_gmem_n = bx * BN + load_b_smem_n;  // B的global memory第几列

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        // global memory to shared memory
        int load_a_gmem_k = bk * BK + load_a_smem_k; // A的global memory的第几列
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K); // A的起始地址
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(A->elements[load_a_gmem_addr]);
        int load_b_gmem_k = bk * BK + load_b_smem_k; // B的global memory的第几行
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N); // B的起始地址
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(B->elements[load_b_gmem_addr]);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c->elements[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
        }
    }
}

int main(void)
{
    // 1、设置GPU设备
    setGPU();

    // 2、分配主机内存和设备内存，并初始化
    int width = 1 << 10;
    int height = 1 << 10;


    Matrix *A, *B, *C;
    // 申请托管内存
    cudaMallocManaged((void**)&A, sizeof(Matrix));
    cudaMallocManaged((void**)&B, sizeof(Matrix));
    cudaMallocManaged((void**)&C, sizeof(Matrix));
    int nBytes = width * height * sizeof(float);
    cudaMallocManaged((void**)&A->elements, nBytes);
    cudaMallocManaged((void**)&B->elements, nBytes);
    cudaMallocManaged((void**)&C->elements, nBytes);

    // 初始化数据
    A->height = height;
    A->width = width;
    B->height = height;
    B->width = width;
    C->height = height;
    C->width = width;
    for (int i = 0; i < width * height; ++i)
    {
        A->elements[i] = 1.0;
        B->elements[i] = 2.0;
    }


    // 定义kernel的执行配置
    dim3 blockSize(THREAD_NUM, THREAD_NUM);
    // 每个线程负责V*V个值的计算
    dim3 gridSize((width + blockSize.x * V - 1) / (blockSize.x * V),
        (height + blockSize.y * V - 1) / (blockSize.y * V));
    // 执行kernel
    cout << (width + blockSize.x * V - 1) / (blockSize.x * V) << endl;
    cout << (height + blockSize.y * V - 1) / (blockSize.y * V) << endl;
    regTilledSharedMemoryMatMulKernel <<< gridSize, blockSize >>>(A, B, C);

    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < width * height; ++i)
        maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
    cout << "最大误差: " << maxError << endl;

    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; // add
    cudaProfilerStop();

    return 0;
}