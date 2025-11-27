// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.



#include <nvbench/nvbench.cuh>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include "kernel.cuh"
#include "utils.h"


#define CUDA_CHECK(x) swiftware::hpp::cuda_check((x), __FILE__, __LINE__)


void report_summary(nvbench::state& state)
{
  state.get_summary("nv/cold/time/gpu/min").remove_value("hide");
  state.get_summary("nv/cold/time/gpu/max").remove_value("hide");
  state.get_summary("nv/cold/time/gpu/mean").remove_value("hide");
  //state.get_summary("nv/cold/time/gpu/mean").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/mean").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/min").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/max").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/stdev/relative").set_string("hide", "");
  state.get_summary("nv/cold/sm_clock_rate/mean").remove_value("hide");
  state.get_summary("nv/cold/sm_clock_rate/scaling/percent").remove_value("hide");

}

void nvbench_mulAdd_v1(nvbench::state& state)
{
  const size_t n = static_cast<size_t>(state.get_int64("n"));
  std::vector<float> A(n), B(n), Out(n), Ref(n);
  swiftware::hpp::generate_random_float_arrays(A.data(), B.data(), Out.data(), Ref.data(), n);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  swiftware::hpp::allocate_and_copy_to_device(A.data(), B.data(), Out.data(), &d_a, &d_b, &d_out, n);

  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
    CUDA_CHECK(cudaMemcpyAsync(d_out, Out.data(), n * sizeof(float), cudaMemcpyHostToDevice, launch.get_stream()));
    // start timer
    timer.start();
    swiftware::hpp::mulAddKernel_v1<float><<<grid, block, 0, launch.get_stream()>>>(d_a, d_b, d_out, n);
    // stop timer
    timer.stop();
  });

  //CUDA_CHECK(cudaStreamSynchronize(0)); // or pass the stream you used
  std::vector<float> out_host(n);
  CUDA_CHECK(cudaMemcpy(out_host.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));


  // compare to Ref with small epsilon
  const float eps = 1e-5f;
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(out_host[i] - Ref[i]) > eps) {
      std::fprintf(stderr, "Mismatch at %zu: got %f expected %f\n", i, out_host[i], Ref[i]);
      std::abort();
    }
  }

  swiftware::hpp::free_buffers(d_a, d_b, d_out);

  report_summary(state);
}

void nvbench_mulAdd_v2(nvbench::state& state)
{
  const size_t n = static_cast<size_t>(state.get_int64("n"));
  std::vector<float> A(n), B(n), Out(n), Ref(n);
  swiftware::hpp::generate_random_float_arrays(A.data(), B.data(), Out.data(), Ref.data(), n);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  swiftware::hpp::allocate_and_copy_to_device(A.data(), B.data(), Out.data(), &d_a, &d_b, &d_out, n);

  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
    CUDA_CHECK(cudaMemcpyAsync(d_out, Out.data(), n * sizeof(float), cudaMemcpyHostToDevice, launch.get_stream()));
    // start timer
    timer.start();
    swiftware::hpp::mulAddKernel_v2<float><<<grid, block, 0, launch.get_stream()>>>(d_a, d_b, d_out, n);
    // stop timer
    timer.stop();
  });

  //CUDA_CHECK(cudaStreamSynchronize(0)); // or pass the stream you used
  std::vector<float> out_host(n);
  CUDA_CHECK(cudaMemcpy(out_host.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));


  // compare to Ref with small epsilon
  const float eps = 1e-5f;
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(out_host[i] - Ref[i]) > eps) {
      std::fprintf(stderr, "Mismatch at %zu: got %f expected %f\n", i, out_host[i], Ref[i]);
      std::abort();
    }
  }

  swiftware::hpp::free_buffers(d_a, d_b, d_out);

  report_summary(state);
}

// TODO: add nvbench_mulAdd_v3
void nvbench_mulAdd_v3(nvbench::state& state)
{
  // TODO: add your benchmarking code here
}

//NVBENCH_BENCH(nvbench_mulAdd_v1).set_name("mulAdd_v1").add_int64_axis("n", {1<<10, 1<<15, 1<<20, 1<<25});
//NVBENCH_BENCH(nvbench_mulAdd_v2).set_name("mulAdd_v2").add_int64_axis("n", {1<<10, 1<<15, 1<<20, 1<<25});

#include <stdio.h>
#include <cuda_runtime.h>
// ... include headers for half and WMMA if placing all code in one file

#define M 16
#define K 16
#define WMMA_N_TILE 16

void run_comparison_N(int N) {
    // 1. Calculate Grid Dimensions
    int total_elements_C = M * N;
    int num_n_tiles = (N + WMMA_N_TILE - 1) / WMMA_N_TILE; // Total 16x16 tiles across the N dimension
    
    // --- Naive CUDA Core Execution ---
    // Launch one thread per element C[row][col]
    dim3 threads_naive(32, 1, 1); // 32 threads per block, simple for 1D mapping
    dim3 blocks_naive((N + 31) / 32, M, 1); // A block row for each M row, blocks to cover N
    
    // ... Initialization, allocation, HtoD transfer (omitted for brevity)
    
    // Naive Kernel Call
    Naive_CUDA_Core_GEMM<<<blocks_naive, threads_naive>>>(A_d_f, B_d_f, C_d_f_naive, N);
    
    // ... Timing and synchronization

    // --- WMMA Tensor Core Execution ---
    // Launch one warp (32 threads) per 16x16 tile in the N dimension.
    // The total number of warps needed is num_n_tiles.
    
    // Set threads to 32 (one warp) and use the grid to launch the tiles.
    dim3 threads_wmma(32, 1, 1); 
    dim3 blocks_wmma(num_n_tiles, 1, 1); // One block per 16x16 tile in the N dimension

    // WMMA Kernel Call
    WMMA_Tensor_Core_GEMM<<<blocks_wmma, threads_wmma>>>(A_d_h, B_d_h, C_d_f_wmma, N);
    
    // ... Timing, synchronization, result reporting
    
    // Note: The WMMA kernel logic relies on threads_wmma being exactly 32 for one warp.
}

int main() {
    int N_value = 4096; // Example N value
    run_comparison_N(N_value);
    return 0;
}
