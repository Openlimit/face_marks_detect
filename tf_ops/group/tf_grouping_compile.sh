#!/usr/bin/env bash
CUDA_PATH=/usr/local/cuda
TF_LIB=/home/meidai/anaconda3/lib/python3.6/site-packages/tensorflow
TF_PATH=/home/meidai/anaconda3/lib/python3.6/site-packages/tensorflow/include
$CUDA_PATH/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -L$TF_LIB -ltensorflow_framework -I $TF_PATH -I $TF_PATH/external/nsync/public/ -I $CUDA_PATH/include -lcudart -L $CUDA_PATH/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0