#!/bin/bash
#SBATCH --job-name=fairness_attacks
#SBATCH -c 4
#SBATCH --gres=gpu:1

: '
cd /nfs-share/jr897/FedDTI
source ../miniconda3/bin/activate workspace
python create_data.py ../DeepDTA/data
mkdir data/processed
cp ../DeepDTA/data/processed/kiba_train.pt data/processed/kiba_train.pt
cp ../DeepDTA/data/processed/kiba_test.pt data/processed/kiba_test.pt
mkdir outs
srun -c 16 --gres=gpu:4 --pty bash
bash test.sh
'

cd /nfs-share/jr897/FedDTI
rm client_output
source ../miniconda3/bin/activate workspace
bash run_server.sh
sleep 3
for CLIENT in 0 1 2 3 4 5 6 7
do
    bash run_client.sh $CLIENT
done
wait