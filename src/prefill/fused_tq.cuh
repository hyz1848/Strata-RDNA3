#pragma once
// Trellis (Gyro) sub-block decode for the fused prompt kernels: the same states and int8 codebook the decode path
// uses (third_party/ggml-tq/tq.cuh), reduced to one 32-weight sub-block per call.
#include <cstdint>
#include <cuda_fp16.h>

#include "tq_lut_i8.cuh"

#ifndef TQ_STATE_BITS
#define TQ_STATE_BITS 15
#endif
#define TQ_STATE_MASK_ ((1u << TQ_STATE_BITS) - 1u)

#define FUSED_TQ_TQ2_T 144
#define FUSED_TQ_TQK6  145
#define FUSED_TQ_TQK7  146
#define FUSED_TQ_TQP5  147
#define FUSED_TQ_TQP6  148
#define FUSED_TQ_TQP7  149

__host__ __device__ constexpr bool fused_tq_is_pair(int t) {
    return t == FUSED_TQ_TQP5 || t == FUSED_TQ_TQP6 || t == FUSED_TQ_TQP7;
}
__host__ __device__ constexpr int fused_tq_K(int t) {
    return t == FUSED_TQ_TQ2_T ? 8 : t == FUSED_TQ_TQK6 || t == FUSED_TQ_TQP6 ? 6
         : t == FUSED_TQ_TQK7 || t == FUSED_TQ_TQP7 ? 7 : 5;
}

// steps 4l..4l+3 of a 128-weight block (TQ2_T / TQK6 / TQK7)
template <int K> __device__ __forceinline__ void fused_tq_states4(const uint8_t* qs, int l, uint32_t s[4]) {
    const uint16_t* qs16 = (const uint16_t*) qs;
    if constexpr (K == 8) {
        const uint32_t w01 = qs16[2 * l], w23 = qs16[2 * l + 1];
        const uint32_t prev = qs[(4 * l + 31) & 31];
        const uint32_t b0 = w01 & 0xFF, b1 = w01 >> 8, b2 = w23 & 0xFF, b3 = w23 >> 8;
        s[0] = (prev << 8) | b0; s[1] = (b0 << 8) | b1; s[2] = (b1 << 8) | b2; s[3] = (b2 << 8) | b3;
    } else {
        constexpr int nwords = 2 * K;
        const int off3 = (28 - 4 * l) * K, wb = off3 >> 4, r = off3 & 15;
        const int wi1 = wb + 1 >= nwords ? wb + 1 - nwords : wb + 1;
        const int wi2 = wb + 2 >= nwords ? wb + 2 - nwords : wb + 2;
        const int wi3 = wb + 3 >= nwords ? wb + 3 - nwords : wb + 3;
        const uint64_t v = (uint64_t) qs16[wb] | ((uint64_t) qs16[wi1] << 16) | ((uint64_t) qs16[wi2] << 32) |
                           ((uint64_t) qs16[wi3] << 48);
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i] = (uint32_t) (v >> (r + (3 - i) * K)) & 0xFFFFu;
    }
}

// pair trellis: steps 8l..8l+7 (Gyro-L)
template <int K> __device__ __forceinline__ void fused_tq_states8(const uint8_t* qs, int l, uint32_t s[8]) {
    constexpr int nwords = 4 * K;
    const uint16_t* qs16 = (const uint16_t*) qs;
    const int S = (56 - 8 * l) * K, wb = S >> 4, r = S & 15;
    const int wi1 = wb + 1 >= nwords ? wb + 1 - nwords : wb + 1;
    const int wi2 = wb + 2 >= nwords ? wb + 2 - nwords : wb + 2;
    const int wi3 = wb + 3 >= nwords ? wb + 3 - nwords : wb + 3;
    uint64_t v = (uint64_t) qs16[wb] | ((uint64_t) qs16[wi1] << 16) | ((uint64_t) qs16[wi2] << 32) |
                 ((uint64_t) qs16[wi3] << 48);
    v >>= r;
    if constexpr (8 + 7 * K + TQ_STATE_BITS > 64) {
        const int wi4 = wb + 4 >= nwords ? wb + 4 - nwords : wb + 4;
        v |= r ? (uint64_t) qs16[wi4] << (64 - r) : 0;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) s[j] = (uint32_t) (v >> ((7 - j) * K)) & TQ_STATE_MASK_;
}

// Sub-block ib (0..3) of one 128-weight block as 32 int8 weights (8 packed words, weight order): two decode lanes
// (16 weights each).  Returns the scale: the block's fp16 d times the codebook's one global int8 scale.
template <int T> __device__ __forceinline__ float fused_tq_sub_q8(const uint16_t* d, const uint8_t* qs, int ib, uint32_t q[8]) {
    constexpr int K = fused_tq_K(T);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int l = 2 * ib + h;
        if constexpr (fused_tq_is_pair(T)) {
            uint32_t s[8];
            fused_tq_states8<K>(qs, l, s);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t p0 = tq_lut_i8[((s[2 * i] & TQ_STATE_MASK_) * 0x9e3779b1u) >> 21];
                const uint32_t p1 = tq_lut_i8[((s[2 * i + 1] & TQ_STATE_MASK_) * 0x9e3779b1u) >> 21];
                q[4 * h + i] = p0 | (p1 << 16);
            }
        } else {
            uint32_t s[4];
            fused_tq_states4<K>(qs, l, s);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t xh = (s[i] & TQ_STATE_MASK_) * 0x9e3779b1u;
                q[4 * h + i] = (uint32_t) tq_lut_i8[xh >> 21] | ((uint32_t) tq_lut_i8[(xh >> 10) & 2047u] << 16);
            }
        }
    }
    return __half2float(__ushort_as_half(*d)) * TQ_LUT_I8_SCALE;
}
