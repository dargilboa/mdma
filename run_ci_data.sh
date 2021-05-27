#!/bin/bash

# Less than 1mn for each on my computer

# If the test has valid level, then p-values should be U(0,1) over multiple runs
python3 ci_test.py --dataset "hard_proc_0" --lr .1 --m 3 --L 2 --n 1000 --batch_size 500 --patience 2000 --n_epochs 10 --use_MERA 0 --verbose 0 -sc 0

# If the test has power, then p-values should be small
python3 ci_test.py --dataset "hard_proc_A" --lr .1 --m 3 --L 2 --n 1000 --batch_size 500 --patience 2000 --n_epochs 10 --use_MERA 0 --verbose 0 -sc 0