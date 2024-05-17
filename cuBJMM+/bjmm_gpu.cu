#include <array>
#include <bit>
#include <bitset>
#include <cstdint>
#include <vector>
#include <omp.h>
#include "bjmm_constant.cuh"
#include "bjmm_gpu.cuh"
#pragma GCC target("avx2")

using namespace std;

constexpr int pow2(int x) { return 1 << x; }

constexpr uint64_t combination(uint64_t num, uint64_t r)
{
    uint64_t c = 1;
    uint64_t b = 1;
    for (uint64_t i = 0; i < r; i++)
    {
        c *= (num - i);
        b *= (i + 1);
    }
    return c / b;
}

constexpr int maxThreadNum = 1000;
constexpr int hsWidth = ((cfg::n - cfg::k) + 63) / 64;
constexpr int kl2 = cfg::kl + 2;
constexpr int mid = kl2 / 2 + cfg::nkl;
constexpr int leftZeroIdx = cfg::nkl;
constexpr int rightZeroIdx = mid;
constexpr int hsMid = mid - cfg::nkl;
constexpr int hsLeftZeroIdx = leftZeroIdx - cfg::nkl;
constexpr int hsRightZeroIdx = rightZeroIdx - cfg::nkl;
constexpr int hsLeftStartIdx = 0;
constexpr int hsLeftEndIdx = hsMid;
constexpr int numLeftCols = hsLeftEndIdx - hsLeftStartIdx;
constexpr int hsRightStartIdx = hsMid;
constexpr int hsRightEndIdx = kl2;
constexpr int numRightCols = hsRightEndIdx - hsRightStartIdx;
constexpr int sizeL1 = numLeftCols * (numLeftCols - 1) / 2;
constexpr int bucketL1 = pow2(cfg::l2);
constexpr int bucketL = pow2(cfg::l1);
constexpr int numL1 = sizeL1 / bucketL1;
constexpr int pL1 = cfg::p / 4;
constexpr int pL = cfg::p / 2;
constexpr int numRightCombination = combination(numRightCols, pL1);
constexpr int numParallelThread = numRightCombination * numL1;

constexpr int div_up(int a, int b)
{
    return (a % b) == 0 ? (a / b) : (a / b + 1);
}

constexpr array<int, numRightCombination * pL1> make_combination(int start, int end)
{
    int size = 0;
    array<int, numRightCombination * pL1> combination = {};
    for (int comb_0 = start; comb_0 < end; comb_0++)
    {
        for (int comb_1 = comb_0 + 1; comb_1 < end; comb_1++)
        {
            combination[size++] = comb_0;
            combination[size++] = comb_1;
        }
    }
    return combination;
}

uint64_t *gd_hs[maxThreadNum];
uint32_t *gd_h1[maxThreadNum];
uint16_t *gd_h2[maxThreadNum];
int *gd_L[maxThreadNum];
int *gd_L1[maxThreadNum];
int *gd_counterL1[maxThreadNum];
int *gd_e[maxThreadNum];
int *gd_solutionFound[maxThreadNum];
int *gd_rightCombination;
bool g_bUseCudaStream;
vector<int> g_devNoList;
vector<cudaStream_t> g_stmList;

__global__ void merge_L_kernel(
    int *right_combination, uint32_t *h1, uint16_t *h2,
    int *L1, int *counterL1, int *L)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numParallelThread)
    {
        return;
    }
    int idx = int(tid / numL1);
    int m = int(tid % numL1);

    int *right_comb_ptr = right_combination + idx * pL1;
    int right_0 = *(right_comb_ptr + 0);
    int right_1 = *(right_comb_ptr + 1);
    uint16_t x2 = h2[right_0] ^ h2[right_1];

    if (m < min(numL1 - 1, counterL1[x2]))
    {
        int index_L1 = x2 * numL1 * pL1 + m * pL1;
        int left_0 = L1[index_L1 + 0];
        int left_1 = L1[index_L1 + 1];
        uint32_t x1 = h1[left_0] ^ h1[left_1] ^ h1[right_0] ^ h1[right_1];
        int index_L = x1 * pL;
        L[index_L + 0] = left_0;
        L[index_L + 1] = left_1;
        L[index_L + 2] = right_0;
        L[index_L + 3] = right_1;
    }
}

__global__ void match_LR_kernel(
    int *right_combination, uint32_t s1, uint16_t s2,
    uint64_t *hs, uint32_t *h1, uint16_t *h2,
    int *L1, int *counterL1, int *L, int *e, int *solution_found)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numParallelThread)
    {
        return;
    }
    int idx = int(tid / numL1);
    int m = int(tid % numL1);

    int *right_comb_ptr = right_combination + idx * pL1;
    int right_0 = *(right_comb_ptr + 0);
    int right_1 = *(right_comb_ptr + 1);
    uint16_t x2 = h2[right_0] ^ h2[right_1] ^ s2;

    if (m < min(numL1 - 1, counterL1[x2]))
    {
        int index_L1 = x2 * numL1 * pL1 + m * pL1;
        int left_0 = L1[index_L1 + 0];
        int left_1 = L1[index_L1 + 1];
        uint32_t x1 = h1[left_0] ^ h1[left_1] ^ h1[right_0] ^ h1[right_1] ^ s1;
        int index_L = x1 * pL;
        int eR[4];
        eR[0] = left_0;
        eR[1] = left_1;
        eR[2] = right_0;
        eR[3] = right_1;

        uint64_t x[hsWidth] = {};
        for (int c = 0; c < pL; c++)
        {
            int index_hs_L = L[index_L + c] * hsWidth;
            int index_hs_R = eR[c] * hsWidth;
            for (int b = 0; b < hsWidth; b++)
            {
                x[b] = x[b] ^ hs[index_hs_L + b] ^ hs[index_hs_R + b];
            }
        }

        uint64_t *s = hs + kl2 * hsWidth;
        int total_diffs = 0;
        for (int b = 0; b < hsWidth; b++)
        {
            x[b] = x[b] ^ s[b];
            total_diffs += __popcll(x[b]);
        }
        if (total_diffs <= cfg::w)
        {
            int duplicate_count = 0;
            for (int cl = 0, cr = 0; cl < pL && cr < pL;)
            {
                if (L[index_L + cl] < eR[cr])
                {
                    cl++;
                }
                else if (L[index_L + cl] == eR[cr])
                {
                    cl++;
                    cr++;
                    duplicate_count++;
                }
                else
                {
                    cr++;
                }
            }
            bool leftZeroCol = false;
            bool rightZeroCol = false;
            for (int i = 0; i < pL; i++)
            {
                leftZeroCol = leftZeroCol != (L[index_L + i] == hsLeftZeroIdx);
                leftZeroCol = leftZeroCol != (eR[i] == hsLeftZeroIdx);
                rightZeroCol = rightZeroCol != (L[index_L + i] == hsRightZeroIdx);
                rightZeroCol = rightZeroCol != (eR[i] == hsRightZeroIdx);
            }
            if (total_diffs <= cfg::w - (cfg::p - 2 * duplicate_count - leftZeroCol - rightZeroCol) && atomicCAS(solution_found, 0, 1) == 0)
            {
                e[0] = L[index_L + 0];
                e[1] = L[index_L + 1];
                e[2] = L[index_L + 2];
                e[3] = L[index_L + 3];
                e[4] = eR[0];
                e[5] = eR[1];
                e[6] = eR[2];
                e[7] = eR[3];
            }
        }
    }
}

bool allocate_memory_gpu(int thread_num, vector<int> &dev_list)
{
    for (int idx = 0; idx < thread_num; idx++)
    {
        g_devNoList.push_back(dev_list[idx % dev_list.size()]);
    }
    if (thread_num <= dev_list.size())
    {
        g_bUseCudaStream = false;
    }
    else
    {
        g_bUseCudaStream = true;
        for (int idx = 0; idx < thread_num; idx++)
        {
            cudaStream_t stm;
            g_stmList.push_back(stm);
        }
    }

    for (int idx = 0; idx < thread_num; idx++)
    {
        cudaSetDevice(g_devNoList[idx]);
        if (g_bUseCudaStream)
        {
            cudaStreamCreate(&g_stmList[idx]);
        }
        cudaMalloc((void **)&gd_hs[idx], sizeof(uint64_t) * (kl2 + 1) * hsWidth);
        cudaMalloc((void **)&gd_h1[idx], sizeof(uint32_t) * kl2);
        cudaMalloc((void **)&gd_h2[idx], sizeof(uint16_t) * kl2);
        cudaMalloc((void **)&gd_L[idx], sizeof(int) * bucketL * pL);
        cudaMalloc((void **)&gd_L1[idx], sizeof(int) * bucketL1 * numL1 * pL1);
        cudaMalloc((void **)&gd_counterL1[idx], sizeof(int) * bucketL1);
        cudaMalloc((void **)&gd_e[idx], sizeof(int) * cfg::p);
        cudaMalloc((void **)&gd_solutionFound[idx], sizeof(int));
        cudaMemset(gd_solutionFound[idx], 0, sizeof(int));
    }
    constexpr array<int, numRightCombination * pL1> right_combination = make_combination(hsRightStartIdx, hsRightEndIdx);
    cudaMalloc((void **)&gd_rightCombination, sizeof(int) * numRightCombination * pL1);
    cudaMemcpy(gd_rightCombination, right_combination.begin(), sizeof(int) * numRightCombination * pL1, cudaMemcpyHostToDevice);

    return true;
}

int initialize_gpu(int thread_num)
{
    int dev_count = 0;
    vector<int> dev_list;

    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0)
    {
        return 0;
    }

    cudaDeviceProp dev_prop = {0};
    for (int i = 0; i < dev_count; i++)
    {
        cudaGetDeviceProperties(&dev_prop, i);
        if (dev_prop.major >= 6)
        {
            dev_list.push_back(i);
        }
    }

    if (dev_list.size() <= 0)
    {
        return 0;
    }

    if (!allocate_memory_gpu(thread_num, dev_list))
    {
        return 0;
    }

    return dev_list.size();
}

bool deallocate_memory_gpu(int thread_num)
{
    for (int idx = 0; idx < thread_num; idx++)
    {
        cudaSetDevice(g_devNoList[idx]);
        if (g_bUseCudaStream)
        {
            cudaStreamDestroy(g_stmList[idx]);
        }
        cudaFree(gd_hs[idx]);
        cudaFree(gd_h1[idx]);
        cudaFree(gd_h2[idx]);
        cudaFree(gd_L[idx]);
        cudaFree(gd_L1[idx]);
        cudaFree(gd_counterL1[idx]);
        cudaFree(gd_e[idx]);
        cudaFree(gd_solutionFound[idx]);
    }
    cudaFree(gd_rightCombination);
    return true;
}

bool uninitialize_gpu(int thread_num)
{
    deallocate_memory_gpu(thread_num);
    cudaDeviceReset();
    return true;
}

bool precompute_gpu(int &tid, cudaStream_t &stm)
{
    tid = omp_get_thread_num();
    stm = (g_bUseCudaStream) ? g_stmList[tid] : 0;
    cudaSetDevice(g_devNoList[tid]);
    return true;
}

bool BJMM_initialize_gpu(uint64_t *hs_ptr, uint32_t *h1_ptr, uint16_t *h2_ptr)
{
    int tid;
    cudaStream_t stm;

    if (!precompute_gpu(tid, stm))
    {
        return false;
    }

    cudaMemcpyAsync(gd_hs[tid], hs_ptr, sizeof(uint64_t) * (kl2 + 1) * hsWidth, cudaMemcpyHostToDevice, stm);
    cudaMemcpyAsync(gd_h1[tid], h1_ptr, sizeof(uint32_t) * kl2, cudaMemcpyHostToDevice, stm);
    cudaMemcpyAsync(gd_h2[tid], h2_ptr, sizeof(uint16_t) * kl2, cudaMemcpyHostToDevice, stm);

    return true;
}

bool merge_L_gpu(int *L1, int *counterL1)
{
    int tid;
    cudaStream_t stm;

    if (!precompute_gpu(tid, stm))
    {
        return false;
    }

    cudaMemcpyAsync(gd_L1[tid], L1, sizeof(int) * pL1 * numL1 * bucketL1, cudaMemcpyHostToDevice, stm);
    cudaMemcpyAsync(gd_counterL1[tid], counterL1, sizeof(int) * bucketL1, cudaMemcpyHostToDevice, stm);
    constexpr int nThreadX = 256;
    constexpr dim3 nThread(nThreadX, 1, 1);
    constexpr dim3 nGrid(div_up(numParallelThread, nThreadX), 1, 1);
    merge_L_kernel<<<nGrid, nThread, 0, stm>>>(
        gd_rightCombination, gd_h1[tid], gd_h2[tid],
        gd_L1[tid], gd_counterL1[tid], gd_L[tid]);

    return true;
}

bool match_LR_gpu(uint32_t s1, uint16_t s2, int *e_ptr, bool *solution_found_ptr)
{
    int tid;
    cudaStream_t stm;

    if (!precompute_gpu(tid, stm))
    {
        return false;
    }

    constexpr int nThreadX = 128;
    constexpr dim3 nThread(nThreadX, 1, 1);
    constexpr dim3 nGrid(div_up(numParallelThread, nThreadX), 1, 1);
    match_LR_kernel<<<nGrid, nThread, 0, stm>>>(
        gd_rightCombination, s1, s2, gd_hs[tid], gd_h1[tid], gd_h2[tid],
        gd_L1[tid], gd_counterL1[tid], gd_L[tid], gd_e[tid],
        gd_solutionFound[tid]);

    cudaMemcpyAsync(solution_found_ptr, gd_solutionFound[tid], sizeof(int), cudaMemcpyDeviceToHost, stm);
    cudaStreamSynchronize(stm);
    if (*solution_found_ptr)
    {
        cudaMemcpyAsync(e_ptr, gd_e[tid], sizeof(int) * cfg::p, cudaMemcpyDeviceToHost, stm);
        cudaStreamSynchronize(stm);
    }

    return true;
}

bitset<cfg::n> BJMM(uint64_t **orig_hs)
{
    bool solution_found = false;
    array<int, cfg::p> e;
    bitset<cfg::n> earray;

    array<uint64_t, (kl2 + 1) * hsWidth> hs;
    for (int i = 0, append_cols_count = 0; i < kl2 + 1; i++)
    {
        if (i == hsLeftZeroIdx || i == hsRightZeroIdx)
        {
            append_cols_count++;
            for (int b = 0; b < hsWidth; b++)
            {
                hs[i * hsWidth + b] = 0ULL;
            }
            continue;
        }
        for (int b = 0; b < hsWidth; b++)
        {
            hs[i * hsWidth + b] = orig_hs[i - append_cols_count][b];
        }
    }

    constexpr uint64_t mask1 = (uint64_t(UINT32_MAX) >> (UINT32_WIDTH - cfg::l1)) << cfg::l2;
    constexpr uint64_t mask2 = uint64_t(UINT16_MAX) >> (UINT16_WIDTH - cfg::l2);

    array<uint32_t, kl2> h1;
    for (int i = 0; i < kl2; i++)
    {
        h1[i] = (hs[i * hsWidth] & mask1) >> cfg::l2;
    }
    array<uint16_t, kl2> h2;
    for (int i = 0; i < kl2; i++)
    {
        h2[i] = hs[i * hsWidth] & mask2;
    }

    BJMM_initialize_gpu(hs.data(), h1.data(), h2.data());

    uint32_t s1 = (orig_hs[cfg::kl][0] & mask1) >> cfg::l2;
    uint16_t s2 = orig_hs[cfg::kl][0] & mask2;
    int L1[bucketL1 * numL1 * pL1];
    int counterL1[bucketL1] = {};
    uint16_t x2 = 0;
    for (int left_0 = 0; left_0 < hsMid; left_0++)
    {
        x2 = h2[left_0];
        for (int left_1 = left_0 + 1; left_1 < hsMid; left_1++)
        {
            x2 = x2 ^ h2[left_1];
            int index_L1 = x2 * numL1 * pL1 + min(numL1 - 1, counterL1[x2]) * pL1;
            L1[index_L1 + 0] = left_0;
            L1[index_L1 + 1] = left_1;
            counterL1[x2]++;
            x2 = x2 ^ h2[left_1];
        }
    }

    merge_L_gpu(L1, counterL1);

    match_LR_gpu(s1, s2, e.data(), &solution_found);

    if (solution_found)
    {
        for (int cl = 0, cr = pL; cl < pL && cr < cfg::p;)
        {
            if (e[cl] < e[cr])
            {
                cl++;
            }
            else if (e[cl] == e[cr])
            {
                e[cl++] = -1;
                e[cr++] = -1;
            }
            else
            {
                cr++;
            }
        }
        for (int i = 0; i < cfg::p; i++)
        {
            if (e[i] == -1 || e[i] == hsLeftZeroIdx || e[i] == hsRightZeroIdx)
            {
                e[i] = -1;
                continue;
            }
            int append_cols_count = (e[i] < hsLeftZeroIdx ? 0 : (e[i] < hsRightZeroIdx ? 1 : 2));
            e[i] = e[i] - append_cols_count;
        }
        for (int i = 0; i < cfg::p; i++)
        {
            if (e[i] != -1)
            {
                earray[e[i] + cfg::nkl] = 1;
            }
        }

        array<uint64_t, hsWidth> diff = {};
        for (int i = 0; i < cfg::p; i++)
        {
            if (e[i] != -1)
            {
                for (int b = 0; b < hsWidth; b++)
                {
                    diff[b] = diff[b] ^ orig_hs[e[i]][b];
                }
            }
        }
        for (int b = 0; b < hsWidth; b++)
        {
            diff[b] = diff[b] ^ orig_hs[cfg::kl][b];
        }

        int j = cfg::n - cfg::k - 1;
        for (int b = 0; b < hsWidth; b++)
        {
            for (int i = 0; i < UINT64_WIDTH; i++)
            {
                if (((1ULL << i) & diff[b]) != 0ULL)
                {
                    earray[j] = 1;
                }
                j--;
                if (j < 0)
                {
                    break;
                }
            }
        }
    }

    return earray;
}
