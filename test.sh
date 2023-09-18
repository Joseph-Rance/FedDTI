#!/bin/bash
#SBATCH --job-name=fairness_attacks
#SBATCH -c 4
#SBATCH --gres=gpu:1

#python create_data.py ../DeepDTA/data
#cp ../DeepDTA/data/processed/kiba_train.pt data/processed/kiba_train.pt
#cp ../DeepDTA/data/processed/kiba_test.pt data/processed/kiba_test.pt

cd /nfs-share/jr897/FedDTI
source ../miniconda3/bin/activate workspace
bash run_server.sh
bash run_client.sh