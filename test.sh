: '
python create_data.py ../DeepDTA/data
cp ../DeepDTA/data/processed/kiba_train.pt data/processed/kiba_train.pt
cp ../DeepDTA/data/processed/kiba_test.pt data/processed/kiba_test.pt
srun -c 4 --gres=gpu:1 --pty bash
bash test.sh
: '

#!/bin/bash
#SBATCH --job-name=fairness_attacks
#SBATCH -c 4
#SBATCH --gres=gpu:1

cd /nfs-share/jr897/FedDTI
source ../miniconda3/bin/activate workspace
bash run_server.sh
sleep 10
for ROUND in 0 1 2 3 4 5 6 7
do
    bash run_client.sh
done