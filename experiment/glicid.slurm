#!/bin/bash
#SBATCH --cluster=nautilus
#SBATCH --nodes=1
#SBATCH --job-name=texturebranchf1dataset25e-3         # Job name
#SBATCH --qos=quick	                 # qos
#SBATCH -p gpu                      # partition
#SBATCH --output=texturebranchf1dataset25e-3.out       # Output file to store results
#SBATCH --ntasks=1                  # Number of tasks (1 task in this case)
#SBATCH --gres=gpu:1                # Request 1 GPU for this job
#SBATCH --time=03:00:00             # Set the maximum job runtime
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
echo "Starting training..."
source ~/.bashrc
micromamba activate 3dcp

cd ./Efficientformer/experiment
python main.py 
