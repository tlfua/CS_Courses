import sys
import os
import csv
import time

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print(
            "usage: python3 generate_shes.py [task_name] [n] [t_start] [t_end]")
        sys.exit(0)
    elif len(sys.argv) != 5:
        print("Error: there should be 5 arguments")
        sys.exit(0)

    task_name = sys.argv[1]
    n = int(sys.argv[2])
    t_start = int(sys.argv[3])
    t_end = int(sys.argv[4])

    for t in range(t_start, t_end + 1):

        # 1. Generate draw bash
        bash_name = task_name + "_" + str(t) + ".sh"

        bash_line = "#!/usr/bin/env bash\n"
        wacc_line = "#SBATCH -p wacc\n"
        x_duration_line = "#SBATCH -t 0-00:00:20\n"
        job_name_line = "#SBATCH -J "
        out_line = "#SBATCH -o "
        # gpu_cpu_line = "#SBATCH --gres=gpu:1 -c 1\n"
        cpu_line = "#SBATCH -N 1 -c 20\n"
        cmd_line = "srun "

        job_name_line += task_name + "_" + str(t) + '\n'
        out_name = task_name + "_" + str(t) + ".out"

        out_line += out_name + "\n"
        cmd_line += task_name + ' ' + str(n) + ' ' + str(t) + '\n'

        with open(bash_name, 'w') as out_bash:
            out_bash.write(bash_line)
            out_bash.write(wacc_line)
            out_bash.write(x_duration_line)
            out_bash.write(job_name_line)
            out_bash.write(out_line)
            # out_bash.write(gpu_cpu_line)
            out_bash.write(cpu_line)
            out_bash.write(cmd_line)

        # 2. Execute draw bash
        os.system("sbatch " + bash_name)
