#!/bin/bash
#SBATCH --cluster=nautilus
#SBATCH --nodes=1
#SBATCH --job-name=stylegan3resumefluid         # Job name
#SBATCH --qos=long	                 # qos
#SBATCH -p gpu                      # partition
#SBATCH --output=stylegan3resumefluid.out       # Output file to store results
#SBATCH --ntasks=1                  # Number of tasks (1 task in this case)
#SBATCH --gres=gpu:1                # Request 1 GPU for this job
#SBATCH --time=8-00:00:00             # Set the maximum job runtime 
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
echo "Starting training..."
source ~/.bashrc
micromamba activate 3dcp

cd ./stylegan3-main
python train.py --outdir=training-runs --cfg=stylegan3-r --data=./concret_windows/fluid.zip --gpus=1 --batch=64 --batch-gpu=16 --gamma=2 --mirror=1 --kimg=3000 --snap=25 --workers=1 --cond=0 --resume=./fluidresume2822.pkl
