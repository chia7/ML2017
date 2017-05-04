#!/bin/bash
wget https://www.csie.ntu.edu.tw/~b04902097/model_1.h5
wget https://www.csie.ntu.edu.tw/~b04902097/model_2.h5
wget https://www.csie.ntu.edu.tw/~b04902097/model_3.h5
wget https://www.csie.ntu.edu.tw/~b04902097/model_4.h5
CUDA_VISIBLE_DEVICES= python3.5 hw3_test.py $1 $2