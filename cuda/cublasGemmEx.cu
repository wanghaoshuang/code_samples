#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cublas_api.h"
#include <iostream>
#include <math.h>
 
using namespace std;
 
#define M 5
#define N 3
#define BYTE 128
 
int main() 
{   
      // 定义状态变量，矩阵乘法接口的返回值
      cublasStatus_t status;
  
      // 在 CPU内存 中为将要计算的矩阵开辟空间
      float *h_A = (float *)malloc (N * M * sizeof(float));
      char *h_B = (char *)malloc (N * M * sizeof(char));
      
      // 在 CPU内存 中为将要存放运算结果的矩阵开辟空间
      float *h_C = (float *)malloc ( M * M * sizeof(float));
  
      float *f_A = (float *)malloc(N * M * sizeof(float));
      float *f_B = (float *)malloc(N * M * sizeof(float));
      // 为待运算矩阵的元素赋予 0-10 范围内的随机数，实际使用时，A矩阵需要做转置，B矩阵不需要转置，可以手动转置也可以算法内转置
      for (int i = 0; i < N * M; i++) 
      {
          f_A[i] = (float)(rand()%10+1);
          f_B[i] = (float)(rand()%10+1);
      }
 
      //FP32转int8型，有精度损失
      for(int i = 0; i < N * M; i++)
      {
//          h_A[i] = (char)(round(f_A[i] * BYTE)); 
          h_A[i] = f_A[i];
          h_B[i] = (char)(round(f_B[i] * BYTE)); 
      }
      
      // 打印待测试的矩阵
      cout << "矩阵 A :" << endl;
      for (int i = 0; i < N * M; i++){
          cout << (float)h_A[i] << " ";
          if ((i + 1) % N == 0) 
              cout << endl;
      }
      cout << endl;
      cout << "矩阵 B :" << endl;
      for (int i = 0; i < N * M; i++){
         cout << (int)h_B[i] << " ";
          if ((i + 1) % M == 0) 
              cout << endl;
      }
      cout << endl;
      
      /*
      ** GPU 计算矩阵相乘
      */
  
      // 创建并初始化 CUBLAS 库对象
      cublasHandle_t handle;
      status = cublasCreate(&handle);
      
      if (status != CUBLAS_STATUS_SUCCESS)
      {
          if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
              cout << "CUBLAS 对象实例化出错" << endl;
          }
          getchar ();
          return EXIT_FAILURE;
      }
  
      float *d_A;
      char *d_B;
      float *d_C;
      // 在 显存 中为将要计算的矩阵开辟空间
      cudaMalloc (
          (void **)&d_A,    // 指向开辟的空间的指针
          N * M * sizeof(float)    //　需要开辟空间的字节数
      );
      cudaMalloc (
          (void **)&d_B,    
          N * M * sizeof(char)    
      );
  
      // 在 显存 中为将要存放运算结果的矩阵开辟空间
      cudaMalloc (
          (void **)&d_C,
          M * M * sizeof(float)    
      );
  
      // 将矩阵数据传递进 显存 中已经开辟好了的空间
      cublasSetVector (
          N * M,    // 要存入显存的元素个数
          sizeof(float),    // 每个元素大小
          h_A,    // 主机端起始地址
          1,    // 连续元素之间的存储间隔
          d_A,    // GPU 端起始地址
          1    // 连续元素之间的存储间隔
      );
     //注意：当矩阵过大时，使用cudaMemcpy是更好地选择：
     //cudaMemcpy(d_A, h_A, sizeof(char)*N*M, cudaMemcpyHostToDevice);
 
     cublasSetVector (
          N * M, 
          sizeof(char), 
          h_B, 
          1, 
          d_B, 
          1
      );
     //cudaMemcpy(d_B, h_B, sizeof(char)*N*M, cudaMemcpyHostToDevice);
     // 同步函数
//     cudaThreadSynchronize();
     cudaDeviceSynchronize(); 
     // 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
     float a = 1.0; 
     float b = 0;
     // 矩阵相乘。该函数必然将数组解析成列优先数组
//     cublasSgemm (
//         handle,    // blas 库对象 
//         CUBLAS_OP_T,    // 矩阵 A 属性参数
//         CUBLAS_OP_T,    // 矩阵 B 属性参数
//         M,    // A, C 的行数 
//         M,    // B, C 的列数
//         N,    // A 的列数和 B 的行数
//         &a,    // 运算式的 α 值
//         d_A,    // A 在显存中的地址
//         N,    // lda
//         d_B,    // B 在显存中的地址
//         M,    // ldb
//         &b,    // 运算式的 β 值
//         d_C,    // C 在显存中的地址(结果矩阵)
//         M    // ldc
//     );
    cublasGemmEx(handle,               //句柄
                   CUBLAS_OP_T,         //矩阵 A 属性参数
                   CUBLAS_OP_T,         //矩阵 B 属性参数
                   M,                   //A, C 的行数 
                   M,                   //B, C 的列数
                   N,                   //A 的列数和 B 的行数
                   &a,                  //运算式的 α 值
                   d_A,                 //A矩阵
                   CUDA_R_32F,           //A矩阵计算模式，FP32型
                   N,                   //A矩阵的列数
                   d_B,                 //B矩阵
                   CUDA_R_8I,           //B矩阵计算模式，int8型
                   M,                   //B矩阵的行数
                   &b,                  //乘法因子beta
                   d_C,                 //C结果矩阵
                   CUDA_R_32F,          //C矩阵计算模式，FP3232型
                   M,                   //C矩阵的行数
                   CUDA_R_32F,          //计算模式，FP32模式
                   CUBLAS_GEMM_ALGO0    //算法参数
                   ); 
     
     // 同步函数
     //cudaDeviceSynchronize();
     cudaDeviceSynchronize(); 
     // 从 显存 中取出运算结果至 内存中去
     cublasGetVector (
         M * M,    //  要取出元素的个数
         sizeof(float),    // 每个元素大小
         d_C,    // GPU 端起始地址
         1,    // 连续元素之间的存储间隔
         h_C,    // 主机端起始地址
         1    // 连续元素之间的存储间隔
     );
 
     //或使用cudaMemcpy(h_C, d_C, sizeof(int)*M*M, cudaMemcpyDeviceToHost);
     // 打印运算结果
     cout << "计算结果的转置 ( (A*B)的转置 )：" << endl;
 
     for (int i = 0;i < M * M; i++)
     {
          cout << h_C[i] << " ";
          if ((i+1)%M == 0) 
            cout << endl;
     }
     
     //注意，这里需要进行归一化操作，乘出来的矩阵需要除以128*128，以还原原来的大小。在此就省略这一步。
     // 清理掉使用过的内存
     free (h_C);
     free (h_B);
     free (h_A);
     free (f_B);
     free (f_A);
 
     try
     {
         cudaFree (d_A);
         cudaFree (d_B);
         cudaFree (d_C);
     }
     catch(...)
     {
          cout << "cudaFree Error!" << endl;
          // 释放 CUBLAS 库对象
     }
 
     cublasDestroy (handle);
     return 0;
}
