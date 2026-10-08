#!/bin/bash
#SBATCH --cluster=nautilus
#SBATCH --nodes=1
#SBATCH --job-name=modelparam         # Job name
#SBATCH --qos=quick                 # qos
#SBATCH -p gpu                      # partition
#SBATCH --output=modelparam.out       # Output file to store results
#SBATCH --ntasks=1                  # Number of tasks (1 task in this case)
#SBATCH --gres=gpu:1                # Request 1 GPU for this job
#SBATCH --time=03:00:00             # Set the maximum job runtime
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G 
source ~/.bashrc
micromamba activate 3dcp
echo "efficientformer..."

cd ~/Efficientformer/models
python efficientformer.py



echo "vgg..."  
cd ~/vgg/models
python custom_vgg19.py

echo "inception..."
cd ~/inception/models
python inceptionresnetv2.py

echo "texture..."
   
cd ~/texture/models
python texture_model.py

echo "midfusion..."
cd ~/Efficientformer/models
python midfusionmodel.py


