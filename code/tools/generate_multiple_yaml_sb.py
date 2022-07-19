"""
    Sample Run:
    python generate_multiple_yaml_sb.py -i experiments/run_11.yaml -t 10

    Generates multiple runs files and the scripts
"""
import os, sys
sys.path.append(os.getcwd())

import argparse
import numpy as np

np.set_printoptions   (precision= 4, suppress= True)

def execute(command, print_flag= False):
    if print_flag:
        print(command)
    os.system(command)

def get_sb_file_content(yaml_file_path, yaml_basename, time_in_hour, index=0):
    content  = "#!/bin/bash --login\n"
    content += "########## SBATCH Lines for Resource Request ##########\n"
    content += "#SBATCH --time=" + str(time_in_hour) + ":00:00             # limit of wall clock time - how long the job will run (same as -t)\n"
    content += "#SBATCH --nodes=1                   # number of different nodes - could be an exact number or a range of nodes (same as -N)\n"
    content += "#SBATCH --ntasks=1                  # number of tasks - how many tasks (nodes) that you require (same as -n)\n"
    content += "#SBATCH --gres=gpu:a100:1\n"
    content += "#SBATCH --cpus-per-task=4           # number of CPUs (or cores) per task (same as -c)\n"
    content += "#SBATCH --mem-per-cpu=4G            # memory required per allocated CPU (or core) - amount of memory (in bytes)\n"
    if index == 0:
        content += "#SBATCH --job-name " + yaml_basename + "            # you can give your job a name for easier identification (same as -J)\n"
    else:
        content += "#SBATCH --job-name " + yaml_basename + "_" + str(index) + "            # you can give your job a name for easier identification (same as -J)\n"
    content += "#SBATCH -o ./slurm_log/%j.log\n"
    content += "########## Command Lines to Run ##########\n"
    content += "cd /mnt/home/kumarab6/project/DEVIANT/code ### change to the directory where your code is located\n"
    content += "conda activate GUPNet_a ### Activate virtual environment\n"
    content += "srun /mnt/home/kumarab6/anaconda3/envs/GUPNet_a/bin/python -u tools/train_val.py --config=" + yaml_file_path + " ### Run python code\n"
    content += "scontrol show job $SLURM_JOB_ID ### write job information to output file\n"

    return content

#===============================================================================
# Argument Parsing
#===============================================================================
ap      = argparse.ArgumentParser()
ap.add_argument('-i', '--input',  default= 'experiments/run_11.yaml', help= 'path of the yaml file')
ap.add_argument('-t', '--time' ,  type= int, default= '10', help= "running time in hours")
ap.add_argument('-c', '--copies', type= int, default= 3   , help= 'number of copies of experiments')
args    = ap.parse_args()

total_copies     = args.copies
time_in_hour     = args.time
yaml_folder      = os.path.dirname(args.input)
yaml_basename    = os.path.basename(args.input).split(".")[0]
yaml_sb_basename = yaml_basename

yaml_paths_list  = [args.input]
sb_paths_list    = [yaml_sb_basename + ".sb"]

# First know the paths of yamls and sb
for i in range(1, total_copies):
    yaml_paths_list   .append(os.path.join(yaml_folder, yaml_basename + "_" + str(i) + ".yaml"))
    sb_paths_list.append(yaml_sb_basename + "_" + str(i) + ".sb")

# Copy the yamls
for i in range(1, total_copies):
    command = "cp " + yaml_paths_list[0] + " " + yaml_paths_list[i]
    execute(command, print_flag= True)

for i in range(total_copies):
    # Create the sb file content
    content = get_sb_file_content(yaml_file_path= yaml_paths_list[i], yaml_basename= yaml_basename, time_in_hour= time_in_hour, index=i)

    # Write content to a file
    with open(sb_paths_list[i], "w") as text_file:
        text_file.write(content)