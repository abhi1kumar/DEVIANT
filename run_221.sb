#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########
#SBATCH --time=10:00:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                   # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=1                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4           # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem-per-cpu=4G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name run_221          # you can give your job a name for easier identification (same as -J)
#SBATCH -o ./slurm_log/%j.log
########## Command Lines to Run ##########
cd /mnt/home/kumarab6/project/DEVIANT/code ### change to the directory where your code is located
conda activate DEVIANT ### Activate virtual environment
srun /mnt/home/kumarab6/anaconda3/envs/DEVIANT/bin/python -u tools/train_val.py --config=experiments/run_221.yaml ### Run python code
scontrol show job $SLURM_JOB_ID ### write job information to output file
