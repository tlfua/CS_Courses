import sys
import os
import csv
import time

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print ("usage: python3 gen_shs_outs.py [branch]_exp [power_start] [power_end]")
        print ("argv[0]: \"gen_shs_outs.py\"")
        print ("argv[1]: the executable name which is the branch name concatenated by \"_exp\"")
        print ("argv[2]: the starting log scale size. In the case of Table 2, it is 5")
        print ("argv[3]: the ending log scale size. In the case of Table 2, it is 8\n")
        print ("As the script name \"gen_shs_outs\" suggests, it iterates the interval [power_start, power_end].")
        print ("In each iteration, it generates an execution bash and submits it to Euler to get a .out file.")
        print ("Rest assured that the execution configuration already hard-coded in the script according to each branch.")
        sys.exit(0)
    elif len(sys.argv) != 4:
        print("Error: there should be 4 arguments")
        sys.exit(0)

    task_name = sys.argv[1]
    n_start = int(sys.argv[2])
    n_end = int(sys.argv[3])

    for n in range(n_start, n_end + 1):

        # 1. Generate draw bash
        bash_name = task_name + "_" + str(n) + ".sh"

        bash_line = "#!/usr/bin/env bash\n"
        wacc_line = "#SBATCH -p wacc\n"
        x_duration_line = "#SBATCH -t 0-00:10:00\n"
        job_name_line = "#SBATCH -J "
        out_line = "#SBATCH -o "
        gpu_cpu_line = "#SBATCH --gres=gpu:1 -c 1\n"
        #cpu_line = "#SBATCH -c 1\n"
        cmd_line = "srun "

        job_name_line += task_name + "_" + str(n) + '\n'
        out_name = task_name + "_" + str(n) + ".out"

        out_line += out_name + "\n"
        cmd_line += task_name + ' ' + str(pow(2, n)) + '\n'

        with open(bash_name, 'w') as out_bash:
            out_bash.write(bash_line)
            out_bash.write(wacc_line)
            out_bash.write(x_duration_line)
            out_bash.write(job_name_line)
            out_bash.write(out_line)
            out_bash.write(gpu_cpu_line)
            #out_bash.write(cpu_line)
            out_bash.write(cmd_line)

        # 2. Execute draw bash
        os.system("sbatch " + bash_name)
