#ifndef CUBJMM_PLUS_BJMM_GPU_H
#define CUBJMM_PLUS_BJMM_GPU_H

#include <bitset>
#include <cstdint>
#include "bjmm_constant.cuh"

bitset<cfg::n> BJMM(uint64_t **orig_hs);
int initialize_gpu(int thread_num);
bool uninitialize_gpu(int thread_num);

#endif // CUBJMM_PLUS_BJMM_GPU_H
