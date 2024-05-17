#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <omp.h>
#include "custom_matrix.h"
#include "bjmm_constant.cuh"
#include "bjmm_gpu.cuh"
#pragma GCC target("avx2")

using namespace std;

int main(int argc, char *argv[])
{
    int thread_num = 1;
    if (argc > 1)
    {
        thread_num = atoi(argv[1]);
        if (thread_num > omp_get_max_threads())
        {
            thread_num = omp_get_max_threads();
        }
        else if (thread_num <= 0)
        {
            thread_num = 1;
        }
    }
    bitset<cfg::n - cfg::k> s;
    bitset<cfg::n> e_permuted[thread_num];
    mzd_t *mzd_h[thread_num], *mzd_h_T[thread_num], *mzd_h_fliped[thread_num];
    mzp_t *permutation[thread_num];
    customMatrixData *matrix_data[thread_num];
    mzd_t *mzd_h_init = mzd_init(cfg::n - cfg::k, cfg::n + 1);
    mzp_t *permutation_init = mzp_init(cfg::n);
    for (int idx = 0; idx < thread_num; idx++)
    {
        mzd_h[idx] = matrix_init(cfg::n - cfg::k, cfg::n + 1);
        mzd_h_T[idx] = mzd_init(mzd_h[idx]->ncols, mzd_h[idx]->nrows);
        permutation[idx] = mzp_init(cfg::n);
        matrix_data[idx] = init_matrix_data(mzd_h[idx]->ncols);
        mzd_h_fliped[idx] = matrix_init(cfg::n - cfg::k, cfg::n + 1);
    }
    ifstream in(cfg::input_path);
    cin.rdbuf(in.rdbuf());
    for (int i = 0; i < 9 + cfg::k; i++)
    {
        string input_string;
        int bit;
        getline(cin, input_string);
        if (i == 5)
        {
            assert(cfg::w == stoi(input_string));
        }
        else if (i >= 7 && i < 7 + cfg::k)
        {
            for (int j = 0; j < cfg::n - cfg::k; j++)
            {
                bit = int(input_string[j]) - 48;
                mzd_write_bit(mzd_h_init, j, cfg::n - cfg::k + (i - 7), bit);
            }
        }
        else if (i == 8 + cfg::k)
        {
            s = bitset<cfg::n - cfg::k>(input_string);
            for (int j = 0; j < cfg::n - cfg::k; j++)
            {
                bit = int(input_string[j]) - 48;
                mzd_write_bit(mzd_h_init, j, cfg::n, bit);
            }
        }
    }
    for (int i = 0; i < cfg::n - cfg::k; i++)
    {
        mzd_write_bit(mzd_h_init, i, i, 1);
    }
    random_device rnd1, rnd2;
    uint64_t seedl = rnd1();
    uint64_t seedr = rnd2();
    uint64_t seed;
    seed = seedl << 32;
    seed += seedr;
    random_seed(seed);
    if (!initialize_gpu(thread_num))
    {
        return 1;
    }

    int chk = 0;
    int target_tid = 0;
#pragma omp parallel num_threads(thread_num)
    {
        while (!chk)
        {
            int tid = omp_get_thread_num();
            mzd_copy(mzd_h[tid], mzd_h_init);
            mzp_copy(permutation[tid], permutation_init);
            constexpr size_t m4ri_k = matrix_opt_k(cfg::n - cfg::k, MATRIX_AVX_PADDING(cfg::n));
            matrix_echelonize_partial_plusfix_opt<cfg::n, cfg::k, cfg::l>(mzd_h[tid], mzd_h_T[tid], m4ri_k, cfg::nkl, matrix_data[tid], permutation[tid]);

            for (int i = 0; i < cfg::n - cfg::k; i++)
            {
                mzd_copy_row(mzd_h_fliped[tid], i, mzd_h[tid], cfg::n - cfg::k - 1 - i);
            }
            mzd_transpose(mzd_h_T[tid], mzd_h_fliped[tid]);
            e_permuted[tid] = BJMM(&mzd_h_T[tid]->rows[cfg::nkl]);
            if (e_permuted[tid].any())
            {
                target_tid = tid;
                chk = 1;
#pragma omp flush(chk)
            }
        }
    }

    bitset<cfg::n> e;
    bitset<cfg::n - cfg::k> He;
    for (int i = 0; i < cfg::n; i++)
    {
        e[cfg::n - 1 - permutation[target_tid]->values[i]] = e_permuted[target_tid][i];
    }
    for (int i = 0; i < cfg::n; i++)
    {
        if (e[i] == 1)
        {
            bitset<cfg::n - cfg::k> h_col;
            for (int j = 0; j < cfg::n - cfg::k; j++)
            {
                h_col[cfg::n - cfg::k - 1 - j] = mzd_read_bit(mzd_h_init, j, cfg::n - 1 - i);
            }
            He = He ^ h_col;
        }
    }
    uninitialize_gpu(thread_num);
    mzd_free(mzd_h_init);
    mzp_free(permutation_init);
    for (int idx = 0; idx < thread_num; idx++)
    {
        mzd_free(mzd_h[idx]);
        mzd_free(mzd_h_T[idx]);
        mzp_free(permutation[idx]);
        free_matrix_data(matrix_data[idx]);
        mzd_free(mzd_h_fliped[idx]);
    }
    cout << "e (solution): " << e << endl;
    assert(He == s);
    return 0;
}
