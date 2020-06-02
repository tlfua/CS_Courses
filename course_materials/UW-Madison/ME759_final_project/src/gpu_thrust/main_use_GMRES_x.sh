#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:10:00
#SBATCH -J main_use_GMRES
#SBATCH -o main_use_GMRES.out
#SBATCH --gres=gpu:1 -c 1

srun main_use_GMRES