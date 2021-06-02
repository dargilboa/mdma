#!/bin/bash

# d = 43
python3 UCI_density_estimation.py --dataset "miniboone" --lr .1 --m 5 --L 4 --n 200 --batch_size 1000 --patience 2000 --n_epochs 1000 --use_tb 1 -ce 1000 -save_checkpoints 1 --max_iters 10000 --eval_validation 0
# d = 21
python3 UCI_density_estimation.py --dataset "hepmass" --lr .5 --m 10 --L 6 --n 500 --batch_size 500 --patience 2000 --n_epochs 1000 --use_tb 1 -ce 1000 -save_checkpoints 1 --max_iters 10000 --eval_validation 0
# d = 8
python3 UCI_density_estimation.py --dataset "gas" --lr .5 --m 10 --L 6 --n 500 --batch_size 500 --patience 2000 --n_epochs 1000 --use_tb 1 -ce 1000 -save_checkpoints 1 --max_iters 10000  --eval_validation 0
# d = 63
python3 UCI_density_estimation.py --dataset "bsds300" --lr .5 --m 10 --L 6 --n 500 --batch_size 500 --patience 2000 --n_epochs 1000 --use_tb 1 -ce 1000 -save_checkpoints 1 --max_iters 10000  --eval_validation 0
