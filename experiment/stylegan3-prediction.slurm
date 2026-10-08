#!/bin/bash
#SBATCH --cluster=nautilus
#SBATCH --nodes=1
#SBATCH --job-name=stylegan3-prediction         # Job name
#SBATCH --qos=debug	                 # qos
#SBATCH -p gpu                      # partition
#SBATCH --output=stylegan3-prediction.out       # Output file to store results
#SBATCH --ntasks=1                  # Number of tasks (1 task in this case)
#SBATCH --gres=gpu:1                # Request 1 GPU for this job
#SBATCH --time=00:20:00             # Set the maximum job runtime 
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
echo "Starting training..."
source ~/.bashrc
micromamba activate 3dcp

cd ./stylegan3-main
python gen_images.py --outdir=out2/fluid --trunc=1 --seeds=1-5000  --network=./model_final/fluidresume2721.pkl
python gen_images.py --outdir=out2/good --trunc=1 --seeds=1-5000  --network=./model_final/goodresume2822.pkl
python gen_images.py --outdir=out2/dry --trunc=1 --seeds=1-5000  --network=./model_final/dryresume1814.pkl
python gen_images.py --outdir=out2/tearing --trunc=1 --seeds=1-5000  --network=./model_final/tearingresume504.pkl
python gen_images.py --outdir=out2/ecrase --trunc=1 --seeds=1-5000 --class=4 --network=./network-snapshot-001501.pkl

