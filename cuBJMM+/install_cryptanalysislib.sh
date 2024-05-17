#!/bin/bash

git clone --recurse-submodules -j4 https://github.com/FloydZ/cryptanalysislib

cd cryptanalysislib/
git checkout 2f6445e
git submodule update --init --recursive

cd deps/m4ri
git checkout 042cab8
autoreconf --install
./configure --enable-openmp
make -j8
cp ../patches/patch.diff ./patch.diff
git apply patch.diff

cd ../../

cp ../cryptanalysislib_patch.diff ./patch.diff
git apply patch.diff

cd ../
