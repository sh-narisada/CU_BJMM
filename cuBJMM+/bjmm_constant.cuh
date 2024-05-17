#ifndef CUBJMM_PLUS_BJMM_CONSTANT_H
#define CUBJMM_PLUS_BJMM_CONSTANT_H

#include <string>

using namespace std;

namespace cfg
{
    constexpr int n = 431;
    constexpr int k = 345;
    constexpr int w = 10;
    constexpr int p = 8;
    constexpr int l1 = 16;
    constexpr int l2 = 12;
    constexpr int l = l1 + l2;
    constexpr int nkl = n - k - l;
    constexpr int kl = k + l;
    const string input_path = "./Challenges/Goppa/Goppa_" + to_string(n);
};

#endif // CUBJMM_PLUS_BJMM_CONSTANT_H
