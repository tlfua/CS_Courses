import sys
import os
import csv
import time

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print(
            "usage: python3 generate_shes.py [start_pow] [end_pow]")
        sys.exit(0)
    elif len(sys.argv) != 3:
        print("Error: there should be 3 arguments")
        sys.exit(0)

    # task_name = sys.argv[1]
    start_pow = int(sys.argv[1])
    end_pow = int(sys.argv[2])

    for pow in range(start_pow, end_pow + 1):

        # 1. Generate draw bash
        bash_name = "task1_" + str(pow) + ".sh"

        # bash content
        bash_line = "#!/usr/bin/env bash\n"

        # specification_line = "#SBATCH -p ppc --gres=gpu:v100:1 -t 0-00:30:00\n"
        specification_line = "#SBATCH -p ppc --gres=gpu:v100:1\n"
        job_name_line = "#SBATCH -J task1_" + str(pow) + '\n'
        out_line = "#SBATCH -o task1_" + str(pow) + ".out\n"

        purge_line = "module purge\n"
        load_line = "module load cuda/10.1 clang/7.0.0\n"
        compile_line = "nvcc task1.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -ccbin $CC -o task1\n"
        run_line = "srun task1 " + str(2 ** pow) + " 5\n"

        with open(bash_name, 'w') as out_bash:
            out_bash.write(bash_line)
            out_bash.write(specification_line)
            out_bash.write(job_name_line)
            out_bash.write(out_line)
            out_bash.write(purge_line)
            out_bash.write(load_line)
            out_bash.write(compile_line)
            out_bash.write(run_line)

        # 2. Execute draw bash
        os.system("sbatch " + bash_name)
