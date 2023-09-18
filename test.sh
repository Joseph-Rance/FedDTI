#!/bin/bash
#SBATCH --job-name=fairness_attacks
#SBATCH -c 4
#SBATCH --gres=gpu:1
cd /nfs-share/jr897/FedDTI
source ../miniconda3/bin/activate workspace
bash run_server.sh
bash run_client.sh