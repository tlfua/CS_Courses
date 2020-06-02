#!/bin/bash
nvcc task2_test.cu scan_test.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o task2_test
