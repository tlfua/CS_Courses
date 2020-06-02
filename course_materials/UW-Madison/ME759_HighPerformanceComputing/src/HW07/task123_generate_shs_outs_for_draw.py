import sys
import os
import csv
import time

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print(
            "usage: python3 generate_shes.py [task_name] [start_pow] [end_pow]")
        sys.exit(0)
    elif len(sys.argv) != 4:
        print("Error: there should be 4 arguments")
        sys.exit(0)

    task_name = sys.argv[1]
    start_pow = int(sys.argv[2])
    end_pow = int(sys.argv[3])

    for pow in range(start_pow, end_pow + 1):

        # 1. Generate draw bash

        # dir_name = task_name + "_draw/"
        bash_name = task_name + "_" + str(pow) + ".sh"
        # file_name = dir_name + file_name

        bash_line = "#!/usr/bin/env bash\n"
        wacc_line = "#SBATCH -p wacc\n"
        x_duration_line = "#SBATCH -t 0-00:00:20\n"
        job_name_line = "#SBATCH -J "
        out_line = "#SBATCH -o "
        gpu_cpu_line = "#SBATCH --gres=gpu:1 -c 1\n"
        cmd_line = "srun "

        job_name_line += task_name + "_" + str(pow) + '\n'
        out_name = task_name + "_" + str(pow) + ".out"

        out_line += out_name + "\n"
        cmd_line += task_name + ' ' + str(2 ** pow) + '\n'

        with open(bash_name, 'w') as out_bash:
            out_bash.write(bash_line)
            out_bash.write(wacc_line)
            # out_bash.write(x_duration_line)
            out_bash.write(job_name_line)
            out_bash.write(out_line)
            out_bash.write(gpu_cpu_line)
            out_bash.write(cmd_line)

        # 2. Execute draw bash
        os.system("sbatch " + bash_name)
