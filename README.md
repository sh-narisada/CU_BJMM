## $\texttt{cuBJMM}^+$: a GPU implementation of the improved Becker--Joux--May--Meurer Algorithm
* This repository provides supplemental materials for the paper ["Solving McEliece-1409 in One Day --- Cryptanalysis with the Improved BJMM Algorithm"](https://eprint.iacr.org/2024/393), including:
    * $\texttt{cuBJMM}^+$: a GPU implementation of the improved BJMM algorithm, modified from the [original implementation](https://www.jstage.jst.go.jp/article/transfun/E106.A/3/E106.A_2022CIP0023/_pdf/) developed by Narisada, Fukushima and Kiyomoto.
    * Bit security estimator: a bit security estimator of the revisited BJMM algorithm for [CryptographicEstimators](https://github.com/Crypto-TII/CryptographicEstimators), which is developed by  Esser, Verbel, Zweydinger and Bellini.
    * ISD optimizer: optimizers used in our paper based on [the optimizer](https://github.com/Memphisd/Revisiting-NN-ISD), which is developed by Esser.

### How to Run
* $\texttt{cuBJMM}^+$: 
    * Clone this repository.
    * Install all dependencies listed as follows.
    * ```cd cuBJMM+```
    * ```make```
    * ```./bjmm.out```

* Bit security estimator: 
    * Clone [CryptographicEstimators](https://github.com/Crypto-TII/CryptographicEstimators).
    * Add ```__init__.py``` and ```bjmm_rev.py``` to ```cryptographic_estimators/SDEstimator/SDAlgorithms``` directory in $\texttt{CryptographicEstimators}$.
    * Please run $\texttt{CryptographicEstimators}$ with our codes.
    * Tested on commit `7362f58`.

* ISD optimizer:
    * Clone https://github.com/Memphisd/Revisiting-NN-ISD.
    * Add ```bjmm_d2_tmto.py, dumer_ss.py, mmt_tmto_ss.py, mmt_tmto.py``` to the directory in the repository.
    * Please run ```optimize_all.ipynb``` in https://github.com/Memphisd/Revisiting-NN-ISD with our codes.
    * Tested on commit `8c29e5e`.

### Licenses
* $\texttt{cuBJMM}^+$: 
    * This software uses [cryptanalysislib](https://github.com/FloydZ/cryptanalysislib) and licensed open-source software as follows:
        * [m4ri](https://github.com/malb/m4ri)
            *  GPL-2.0 license
    * This software is licensed under GPL-2.0.
    * The license follows m4ri.

* Bit security estimator:
    * This software uses licensed open-source software as follows:
        * [$\texttt{CryptographicEstimators}$](https://github.com/Crypto-TII/CryptographicEstimators)
            *  GPL-3.0 license
    * This software is licensed under GPL-3.0.
    * The license follows $\texttt{CryptographicEstimators}$.
 
* ISD optimizer:
    * The license follows https://github.com/Memphisd/Revisiting-NN-ISD.

### Dependencies
* $\texttt{cuBJMM}^+$: 
    * autoconf
    * cryptanalysislib (auto-install with make)
    * cuda
    * gcc
    * libpng-dev
    * libtool
    * make


### Contact
* sh-narisada [a.t.] kddi.com
