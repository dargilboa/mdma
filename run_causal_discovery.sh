#!/bin/bash

python3 causal_discovery.py --dataset "sachs" --lr .1 --m 3 --L 2 --n 1000 --batch_size 500 --patience 2000 --n_epochs 10 --verbose 0 -sc 0
