#!/bin/bash
nohup python training_fl.py --server localhost:8080 --seed $RANDOM  --folder data --early-stop 50 --normalisation ln --num-clients 8 --partition $1 &>> client_output &