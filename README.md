# MDMA
Pytorch implementation of the Marginalizable Density Model Approximator

## Requirements
* **``python>=3.6``** 
* **``numpy>=1.20.2``** 
* **``pytorch>=1.0.0``**

Optional for visualization and plotting: ``matplotlib`` and ``tensorboardX``

## Structure
* MDMA/models.py:   Implementation of MDMA class
* MDMA/fit.py:      Fitting MDMA model
* MDMA/utils.py:    Various auxiliary functions
* experiments:      Code for reproducing the experiments in the paper

## Usage
Below, example commands are given for running experiments.

#### Download datasets
Run the following command to download the UCI datasets:
```
./experiments/download_datasets.sh
```

#### Run 2D toy density estimation
This example runs density estimation on the `8 Gaussians` dataset using 1 flow of BNAF with 2 layers and 100 hidden units (`50 * 2` since the data dimensionality is 2).
```
python toy2d.py --dataset 8gaussians \    # which dataset to use
                --experiment density2d \  # which experiment to run
                --flows 1 \               # BNAF flows to concatenate
                --layers 2 \              # layers for each flow of BNAF
                --hidden_dim 50 \         # hidden units per dimension for each hidden layer
                --save                    # save the model after training
                --savefig                 # save the density plot on disk
```

![Imgur](https://i.imgur.com/DWVGsyn.jpg)

#### Toy 3D density estimation

Density estimation on a toy dataset of two spirals, showing the ability of MDMA to compute marginal and conditional distributions.

Run toy density estimation:

```
python3 experiments/toy_density_estimation.py
```

![Data](./experiments/images/s1.pdf)

#### Density estimation with missing values

Runs MDMA on the UCI POWER dataset:

```
python3 experiments/UCI_density_estimation.py --dataset gas \
                                              --m 4000 \
                                              --r 5 \
                                              --l 4 \
                                              --batch_size 500 \
                                              --n_epochs 1000 \
                                              --lr 0.01 \
                                              --missing_data_pct 0.5 # proportion of missing values
```

Runs estimation with BNAF on the same dataset with MICE imputation:

```
python3 experiments/BNAF/density_estimation.py --dataset gas \
                                               --hidden_dim 320 \
                                               --missing_data_pct 0.5 \
                                               --missing_data_strategy mice
```

#### Mutual information estimation


#### Causal discovery 
Requires 
* **``R>=0.5.23``** 

as well as the python packages
* **``cdt>=0.5.23``** 
* **``rpy2>=3.4.4``** 
* **``pytorch>=1.0.0``**

Runs the causal discovery experiment, recovering a causal graph from data by testing for conditional independence using MDMA:

```
python3 experiments/causal_discovery/causal_discovery.py --dataset "sachs" \
                                                         --lr .1 \
                                                         --r 3 \
                                                         --l 2 \
                                                         --m 1000 \
                                                         --batch_size 500 \
                                                         --patience 100 \
                                                         --n_epochs 50 \
                                                         --verbose 1 \
                                                         --save_checkpoints 0

```

#### UCI density estimation

Density estimation on UCI dataset. Possible values for dataset are power, gas, hepmass, miniboone.

Runs MDMA on the UCI POWER dataset:

```
python3 experiments/UCI_density_estimation.py --dataset power \
                                              --m 1000 \
                                              --r 3 \
                                              --l 2 \
                                              --batch_size 500 \
                                              --n_epochs 1000 \
                                              --lr 0.01 
```

For other datasets, we found the best performance with the following values of the parameters:

