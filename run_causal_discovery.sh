#!/bin/bash

# # If the CI test has valid level, then p-values should be U(0,1) over multiple runs
# python3 experiments/causal_discovery/ci_test_debug.py --dataset "hard_proc_0" --lr .1 --m 3 --L 2 --n 1000 --batch_size 500 --patience 2000 --n_epochs 10 --verbose 0 -sc 0

# # If the CI test has power, then p-values should be small
# python3 experiments/causal_discovery/ci_test_debug.py --dataset "hard_proc_A" --lr .1 --m 3 --L 2 --n 1000 --batch_size 500 --patience 2000 --n_epochs 10 --verbose 0 -sc 0

# Reproduce the figure of the paper
python3 causal_discovery.py --dataset "sachs" --lr .1 --m 3 --L 2 --n 1000 --batch_size 500 --patience 100 --n_epochs 50 --verbose 1 -sc 0
