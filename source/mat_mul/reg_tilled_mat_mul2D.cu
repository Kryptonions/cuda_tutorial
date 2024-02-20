#include <iostream>
#include "../tools/common.cuh"

using namespace std;

const int TILE_WIDTH = 16; // 必须加const
//extern __shared__ float sharedA[][];
//extern __shared__ float sharedB[][];
#define V 8  // 每个thread负责计算的元素数量V*V

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

// 矩阵相乘kernel，2-D，每个线程计算一个元素
__global__ void regTilledMatMulKernel(Matrix *A, Matrix *B, Matrix *C)
{
    // block内共享内存，矩阵维度必须是常量，否则编译报错
    //__shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    //__shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];
    float c[V][V] = {0};
    float a[V], b[V];

	float Cvalue = 0.0;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

    int m = A->height;                //A的行数
    int n = A->width;                 //A的列数
    int k = B->width;                 //B的列数
    for (int i = 0; i < n; i++) {
        // global memory to register
        for (int j = 0; j < V; j++) {
            a[j] = getElement(A, row * V + j, i);
        }
        for (int j = 0; j < V; j++>) {
            b[j] = getElement(B, i, col * V + j);
        }
        // 两个向量做乘法
        for (int j = 0; j < V; j++) {
            for (int k = 0; k < V; k++) {
                c[j][k] = a[j] * b[k];
            }
        }
    }
    // copy back
    for (int j = 0; j < V; j++) {
        for (int k = 0; k < V; k++) {
            setElement(C, row * V + j, col * V + k, c[j][k]);
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
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    // 每个线程负责V*V个值的计算
    dim3 gridSize((width + blockSize.x * V - 1) / (blockSize.x * V),
        (height + blockSize.y * V - 1) / (blockSize.y * V));
    // 执行kernel
    regTilledMatMulKernel <<< gridSize, blockSize >>>(A, B, C);

    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < width * height; ++i)
        maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
    cout << "最大误差: " << maxError << endl;

    return 0;
}