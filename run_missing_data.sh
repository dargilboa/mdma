#!/bin/bash
CUDA_VISIBLE_DEVICES="$2"
python3 UCI_density_estimation.py --dataset "gas" --lr .01 --m 3 --L 2 --n 4000 --batch_size 500 --tb_dir "/data/tb" --data_dir "/data/data" --n_epochs 1000 --use_tb 1 --save_checkpoints 0 --missing_data_pct "$1"
python3 UCI_density_estimation.py --dataset "gas" --lr .01 --m 3 --L 2 --n 4000 --batch_size 500 --tb_dir "/data/tb" --data_dir "/data/data" --n_epochs 1000 --use_tb 1 --save_checkpoints 0 --missing_data_pct "$1"
python3 UCI_density_estimation.py --dataset "gas" --lr .01 --m 3 --L 2 --n 4000 --batch_size 500 --tb_dir "/data/tb" --data_dir "/data/data" --n_epochs 1000 --use_tb 1 --save_checkpoints 0 --missing_data_pct "$1"
