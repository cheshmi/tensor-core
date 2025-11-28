// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef TUT02_KERNEL_CUH
#define TUT02_KERNEL_CUH

#include <cuda_fp16.h>
#include <mma.h>
namespace wmma = nvcuda::wmma;

namespace swiftware::hpp
{

// Kernel utilizing standard CUDA Cores (float)
__global__ void Naive_CUDA_Core_GEMM(const float* A, const float* B, float* C, int N) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) { // M=16 is fixed
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) { // K=16 is fixed
            // A is M x K (16x16), B is K x N (16xN)
            sum += A[row * K + k] * B[k * N + col]; 
        }
        C[row * N + col] = sum;
    }
}



// WMMA fragment dimensions (M x K x N = 16 x 16 x 16)
constexpr int WMMA_M = 16;
constexpr int WMMA_N_TILE = 16;
constexpr int WMMA_K = 16;

// Kernel utilizing WMMA for D = A * B + C
__global__ void WMMA_Tensor_Core_GEMM(const half* A, const half* B, float* C_out, int N) {

    // 1. Fragment Declarations
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N_TILE, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N_TILE, WMMA_K, half, wmma::col_major> b_frag; 
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N_TILE, WMMA_K, float> acc_frag;
    
    // 2. Tiling Setup
    // Grid maps to the 16x16 tiles of C.
    // blockIdx.y maps to the M dimension (row tile index). Since M=16, blockIdx.y should be 0.
    // blockIdx.x maps to the N dimension (column tile index).
    
    int tile_idx_n = blockIdx.x;
    
    // The warp index (0-63) inside the thread block is used to compute a tile.
    // For simplicity and efficiency, we launch one warp per tile across the N dimension.
    
    // Check if we are within the bounds of N
    if (tile_idx_n * WMMA_N_TILE < N) {

        // The starting point (top-left element) of the 16x16 tile this block computes.
        int tile_m = 0; // Fixed at 0 since M=16
        int tile_n = tile_idx_n * WMMA_N_TILE; // N dimension steps by 16

        // 3. Initialize Accumulator
        wmma::fill_fragment(acc_frag, 0.0f);

        // 4. Load A and Loop through the K dimension
        // Since K=16, the entire calculation only involves a single mma_sync operation.
        
        // Load A (16x16)
        const half* A_start = A; 
        wmma::load_matrix_sync(a_frag, A_start, K); // K is leading dimension for A

        // Load B (16x16 fragment) - The B fragment changes with the column tile (tile_n)
        // B is K x N. The column index is tile_n.
        const half* B_start = B + tile_n; 
        wmma::load_matrix_sync(b_frag, B_start, N); // N is leading dimension for B (full N)

        // 5. Matrix Multiplication and Accumulation
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        // 6. Store Result C (16x16 fragment)
        // C_out is M x N. The column index is tile_n.
        float* C_tile_start = C_out + tile_n;
        wmma::store_matrix_sync(C_tile_start, acc_frag, N, wmma::row_major);
    }
}


}
#endif //TUT02_KERNEL_CUH