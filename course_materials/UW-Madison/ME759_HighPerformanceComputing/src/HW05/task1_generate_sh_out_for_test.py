import sys
import os
import csv
import time

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print(
            "usage: python3 generate_shes.py [task_name] [n] [threads_per_block]")
        sys.exit(0)
    elif len(sys.argv) != 4:
        print("Error: there should be 4 arguments")
        sys.exit(0)

    task_name = sys.argv[1]
    n = int(sys.argv[2])
    threads_per_block = int(sys.argv[3])
    # txt_name = sys.argv[4]
    # sleep_secs = int(sys.argv[5])

    # out_names = []


    # 1. Generate test bash
    test_name = task_name + "_" + str(n) + "_" + str(threads_per_block)
    bash_name = test_name + ".sh"

    bash_line = "#!/usr/bin/env bash\n"
    wacc_line = "#SBATCH -p wacc\n"
    x_duration_line = "#SBATCH -t 0-00:00:20\n"
    job_name_line = "#SBATCH -J "
    out_line = "#SBATCH -o "
    gpu_cpu_line = "#SBATCH --gres=gpu:1 -c 1\n"
    cmd_line = "srun "

    job_name_line += test_name + '\n'

    out_name = test_name + ".out"
    out_line += out_name + "\n"

    cmd_line += task_name + ' ' + \
            str(n) + ' ' + str(threads_per_block) + '\n'

    with open(bash_name, 'w') as out_bash:
        out_bash.write(bash_line)
        out_bash.write(wacc_line)
        out_bash.write(job_name_line)
        out_bash.write(out_line)
        out_bash.write(gpu_cpu_line)
        out_bash.write(cmd_line)

    # 2. Execute test bash
    os.system("sbatch " + bash_name)

