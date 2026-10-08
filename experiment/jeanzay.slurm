#!/bin/bash 
#SBATCH --job-name=inceptiond2f5
#SBATCH --error=inceptiond2f5
#SBATCH --partition=gpu_p13 
#SBATCH --constraint=v100
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=10 
#SBATCH --time=02:00:00 
#SBATCH --qos=qos_gpu-dev
#SBATCH --account=rog@v100 
#SBATCH --hint=nomultithread



module purge                            # nettoyer les modules herites par defaut

module load pytorch-gpu/py3/2.3.0       # charger les modules

set -x                                  # activer l’echo des commandes
cd ./Baseline_modele/experiment
srun python main.py                   # executer son script
