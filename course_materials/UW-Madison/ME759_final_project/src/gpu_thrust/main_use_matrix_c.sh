module load cuda
nvcc main_use_matrix.cu use_matrix.cu matrix.cu vector.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o main_use_matrix
