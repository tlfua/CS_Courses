#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:00:20
#SBATCH -J main_use_matrix
#SBATCH -o main_use_matrix.out
#SBATCH -N 1 -c 20

srun main_use_matrix
