#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda_fp16.h>
#include <string>
#include "time.h"

struct Weight {
 int8_t data[4];
};

struct HalfWeight {
  half data[2];
};



__global__ void dequant_from_4bits_align_out(int8_t* input, HalfWeight* output, int nbInput, float* scales, int quant_stride) {
  const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < nbInput; i += blockDim.x * gridDim.x) {
      int8_t n =  input[i];
     int8_t low = (n << 4) >> 4;
     int8_t high = n >> 4;
     size_t out_idx = 2 * idx;
     HalfWeight hw;
     hw.data[0] = static_cast<half>(static_cast<float>(low) * scales[out_idx / quant_stride]);
     hw.data[1] = static_cast<half>(static_cast<float>(high) * scales[(out_idx+1)/quant_stride]);
     output[idx] = hw;
  }
}



__global__ void dequant_from_4bits(int8_t* input, half* output, int nbInput, float* scales, int quant_stride) {
  const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < nbInput; i += blockDim.x * gridDim.x) {
      int8_t n =  input[i];
     int8_t low = (n << 4) >> 4;
     int8_t high = n >> 4;
     size_t out_idx = 2 * idx;
     output[out_idx] = static_cast<half>(static_cast<float>(low) * scales[out_idx / quant_stride]);
     output[out_idx+1] = static_cast<half>(static_cast<float>(high) * scales[(out_idx+1)/quant_stride]);
  }
}

__global__ void dequant_from_4bits_align(Weight* input, half* output, int nbInput, float* scales, int quant_stride) {
  const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < nbInput; i += blockDim.x * gridDim.x) {
     Weight n =  input[i];

     int8_t low = (n.data[0] << 4) >> 4;
     int8_t high = n.data[0] >> 4;
     size_t out_idx = 2 * (idx*4);
     output[out_idx] = static_cast<half>(static_cast<float>(low) * scales[out_idx / quant_stride]);
     output[out_idx+1] = static_cast<half>(static_cast<float>(high) * scales[(out_idx+1)/quant_stride]);

     low = (n.data[1] << 4) >> 4;
     high = n.data[1] >> 4;
     out_idx = 2 * (idx*4 + 1);
     output[out_idx] = static_cast<half>(static_cast<float>(low) * scales[out_idx / quant_stride]);
     output[out_idx+1] = static_cast<half>(static_cast<float>(high) * scales[(out_idx+1)/quant_stride]);

     low = (n.data[2] << 4) >> 4;
     high = n.data[2] >> 4;
     out_idx = 2 * (idx*4 + 2);
     output[out_idx] = static_cast<half>(static_cast<float>(low) * scales[out_idx / quant_stride]);
     output[out_idx+1] = static_cast<half>(static_cast<float>(high) * scales[(out_idx+1)/quant_stride]);

     low = (n.data[3] << 4) >> 4;
     high = n.data[3] >> 4;
     out_idx = 2 * (idx*4 + 3);
     output[out_idx] = static_cast<half>(static_cast<float>(low) * scales[out_idx / quant_stride]);
     output[out_idx+1] = static_cast<half>(static_cast<float>(high) * scales[(out_idx+1)/quant_stride]);

  }
}


int main(void)
{


  std::string algos[3];
  algos[0] = "read_8bits_write_16bits";
  algos[1] = "read_32bits_write_16bits";
  algos[2] = "read_8bits_write_32bits";

  std::vector<int8_t> weight_(3*4096*4096/2);
  weight_.assign(3*4096*4096/2, static_cast<int8_t>(1.0));
  int quant_stride_ = 4096*4096;
  std::vector<float> scales_(3);
  scales_.assign(3, 1.0);
  int8_t* p_gpu_weight_;
  float* p_gpu_scales_;
  half* output;
  cudaMalloc(&p_gpu_weight_, sizeof(int8_t) * weight_.size());
  cudaMemcpy(p_gpu_weight_, weight_.data(), weight_.size() * sizeof(int8_t),
             cudaMemcpyHostToDevice);
  cudaMalloc(&p_gpu_scales_, sizeof(float) * scales_.size());
  cudaMemcpy(p_gpu_scales_, scales_.data(), sizeof(float) * scales_.size(),
             cudaMemcpyHostToDevice);

  cudaMalloc(&output, sizeof(half) * weight_.size() * 2);



  cudaEvent_t start, stop;
  float time;
  float min_time = 100000.0;
  size_t best_blocks_factor, best_threads, best_algo;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
//  clock_t start_t, finish_t;

for (size_t blocks_factor = 1; blocks_factor<33; blocks_factor*=2) {
for (size_t threads=32; threads<=1024; threads+=32) {
std::cout << std::endl;
for (size_t algo=0; algo<3; algo++) {
//  cudaDeviceSynchronize();
  cudaEventRecord(start, 0);
//  start_t = clock();
  for (size_t i=0; i< 180; i++) {
    if (algo==0) {
      int blocks = ((weight_.size() + threads - 1) / threads) / blocks_factor;
      dequant_from_4bits<<<blocks, threads>>>(p_gpu_weight_,
                                                              output,
                                                              weight_.size(),
                                                              p_gpu_scales_,
                                                              quant_stride_);
    } else if (algo==1){
      int blocks = ((weight_.size()/4 + threads - 1) / threads) / blocks_factor;
      dequant_from_4bits_align<<<blocks, threads>>>(reinterpret_cast<Weight*>(p_gpu_weight_),
                                                              output,
                                                              weight_.size()/4,
                                                              p_gpu_scales_,
                                                              quant_stride_);
    } else if (algo==2) {
      int blocks = ((weight_.size() + threads - 1) / threads) / blocks_factor;
      dequant_from_4bits_align_out<<<blocks, threads>>>( p_gpu_weight_,
                                                              reinterpret_cast<HalfWeight*>(output),
                                                              weight_.size(),
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
  if (time < min_time) {
    min_time = time;
    best_blocks_factor = blocks_factor;
    best_threads = threads;
    best_algo = algo;
    std::cout <<"\e[1;31m blocks_factor=1/" << blocks_factor << "; threads="<< threads << "; align=" << algos[algo]<<"; cost time: " << time << "\e[0m"<< std::endl;
  } else {
    std::cout <<"blocks_factor=1/" << blocks_factor << "; threads="<< threads <<"; align="<<algos[algo]<< "; cost time: " << time << std::endl;
  }
} // search
} // search
} // search

std::cout <<"\e[1;31m Best -----blocks_factor=1/" << best_blocks_factor << "; threads="<< best_threads << "; align=" << algos[best_algo]<<"; cost time: " << min_time << "\e[0m"<< std::endl;


  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(p_gpu_weight_);
  cudaFree(p_gpu_scales_);
  cudaFree(output);
}
