#!/bin/bash
nvcc task1.cu reduce_v1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o task1_v1
