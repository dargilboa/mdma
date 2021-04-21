#!/bin/bash

# python3 UCI_density_estimation.py --dataset "power" --lr .05 --m 5 --L 6 --n 1000 --batch_size 2000 --patience 200 --epochs 2
# python3 UCI_density_estimation.py --dataset "power" --lr .05 --m 5 --L 6 --n 1000 --batch_size 2000 --patience 500 --epochs 2 --l2 0.0001
# python3 UCI_density_estimation.py --dataset "power" --lr .05 --m 5 --L 6 --n 500 --batch_size 4000 --patience 500 --epochs 2
# python3 UCI_density_estimation.py --dataset "power" --lr .1 --m 10 --L 6 --n 500 --batch_size 2000 --patience 400 --epochs 2

# python3 UCI_density_estimation.py --dataset "power" --lr .05 --m 5 --L 6 --n 3000 --batch_size 1000 --patience 1000 --epochs 2
python3 UCI_density_estimation.py --dataset "power" --lr .1 --m 5 --l 6 --n 100 --r 100 --batch_size 2000 --patience 200 --l2=0.0 --epochs 1000 --use_tb 1 -ce 200 -save_checkpoints 1
