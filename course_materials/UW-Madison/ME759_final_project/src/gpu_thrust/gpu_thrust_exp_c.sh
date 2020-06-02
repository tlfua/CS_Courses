module load cuda
nvcc gpu_thrust_exp.cu matrix.cu vector.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o gpu_thrust_exp