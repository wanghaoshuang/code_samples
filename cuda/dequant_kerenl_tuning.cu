#include <stdio.h>
#include <iostream>
#include <vector>
#include "time.h"

struct Weight {
 int8_t data[4];
};

__global__ void dequant_from_8bits_align(Weight* input, float* output, int nbInput, float* scales, int quant_stride) {
  const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float c_scales[3];
  if (threadIdx.x == 0) {
    c_scales[0] = scales[0];
    c_scales[1] = scales[1];
    c_scales[2] = scales[2];
  }
  __syncthreads();
  for (size_t i = idx; i < nbInput; i += blockDim.x * gridDim.x) {
  //for (size_t i = idx; i < nbInput; i += blockDim.x) {
    Weight n =  input[i]; // 17.6ms

    float out = (static_cast<float>(n.data[0]) * c_scales[i / quant_stride]); // 2ms
    output[i*4] = out; // 24.4ms

    out = (static_cast<float>(n.data[1]) * c_scales[i / quant_stride]); // 2ms
    output[i*4 + 1] = out; // 24.4ms

    out = (static_cast<float>(n.data[2]) * c_scales[i / quant_stride]); // 2ms
    output[i*4 + 2] = out; // 24.4ms

    out = (static_cast<float>(n.data[3]) * c_scales[i / quant_stride]); // 2ms
    output[i*4 + 3] = out; // 24.4ms
  }
}

__global__ void dequant_from_8bits(int8_t* input, float* output, int nbInput, float* scales, int quant_stride) {
  const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < nbInput; i += blockDim.x * gridDim.x) {
    int8_t n =  input[i]; // 17.6ms
    //float out = (static_cast<float>(n) * c_scales[i / quant_stride]); // 2ms
    float out = (static_cast<float>(n) * 1.0); // 2ms
    output[i] = out; // 24.4ms
  }
}



int main(void)
{
  std::vector<int8_t> weight_(3*4096*4096);
  weight_.assign(3*4096*4096, static_cast<int8_t>(1.0));
  int quant_stride_ = 4096*4096;
  std::vector<float> scales_(3);
  scales_.assign(3, 1.0);
  int8_t* p_gpu_weight_;
  float* p_gpu_scales_;
  float* output;
  cudaMalloc(&p_gpu_weight_, sizeof(int8_t) * weight_.size());
  cudaMemcpy(p_gpu_weight_, weight_.data(), weight_.size() * sizeof(int8_t),
             cudaMemcpyHostToDevice);
  cudaMalloc(&p_gpu_scales_, sizeof(float) * scales_.size());
  cudaMemcpy(p_gpu_scales_, scales_.data(), sizeof(float) * scales_.size(),
             cudaMemcpyHostToDevice);

  cudaMalloc(&output, sizeof(float) * weight_.size());



  cudaEvent_t start, stop;
  float time;
  float min_time = 100000.0;
  size_t best_blocks_factor, best_threads, best_algo;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
//  clock_t start_t, finish_t;

for (size_t blocks_factor = 1; blocks_factor<33; blocks_factor*=2) {
for (size_t threads=32; threads<=1024; threads+=32) {
for (size_t algo=0; algo<2; algo++) {
  int blocks = ((weight_.size()/4 + threads - 1) / threads) / blocks_factor;
//  cudaDeviceSynchronize();
  cudaEventRecord(start, 0);
//  start_t = clock();
  for (size_t i=0; i< 180; i++) {
    if (algo==0) {
      dequant_from_8bits<<<blocks, threads>>>(p_gpu_weight_,
                                                              output,
                                                              weight_.size(),
                                                              p_gpu_scales_,
                                                              quant_stride_);
    } else {
      dequant_from_8bits_align<<<blocks, threads>>>(reinterpret_cast<Weight*>(p_gpu_weight_),
                                                              output,
                                                              weight_.size()/4,
                                                              p_gpu_scales_,
                                                              quant_stride_);
    }
  }
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  std::cout <<"blocks_factor=" << blocks_factor << "; threads="<< threads <<"; algo="<<algo<< "; time: " << time << std::endl;
  if (time < min_time) {
    min_time = time;
    best_blocks_factor = blocks_factor;
    best_threads = threads;
    best_algo = algo;
    std::cout <<"\e[1;31m blocks_factor=" << blocks_factor << "; threads="<< threads << "; algo=" << algo<<"; time: " << time << "\e[0m"<< std::endl;
  } else {
    std::cout <<"blocks_factor=" << blocks_factor << "; threads="<< threads <<"; algo="<<algo<< "; time: " << time << std::endl;
  }
} // search
} // search
} // search

std::cout <<"\e[1;31m Best -----blocks_factor=" << best_blocks_factor << "; threads="<< best_threads << "; algo=" << best_algo<<"; time: " << min_time << "\e[0m"<< std::endl;


  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(p_gpu_weight_);
  cudaFree(p_gpu_scales_);
  cudaFree(output);
}
