# MDMA
Pytorch implementation of the Marginalizable Density Model Approximator — A density estimator that provides closed-form marginal and conditional densities. 

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

#### Toy 3D density estimation

Density estimation on a toy dataset of two spirals, showing the ability of MDMA to compute marginal and conditional distributions.

Fit two spirals density using MDMA, and plot marginals and conditionals:

```
python3 experiments/toy_density_estimation.py
```

For a two spiral dataset, the samples and marginal histograms of the data take the following form:
![Data](experiments/images/s1.jpg)
Samples from the trained MDMA model and the learned marginal densities evaluated on a grid are indistinguishable:
![Samples and marginals](experiments/images/s2.jpg)
MDMA also provides closed-form expression for all conditional densities:
![Conditionals](experiments/images/s4.jpg?s=100)

#### UCI density estimation

Density estimation on UCI dataset. Possible values for dataset are power, gas, hepmass, miniboone.

Runs MDMA on the UCI POWER dataset:

```
python3 experiments/UCI_density_estimation.py --dataset power \
                                              --m 1000 \           # Width of tensor network
                                              --r 3 \              # Width of univariate CDF networks
                                              --l 2 \              # Depth of univariate CDF networks
                                              --batch_size 500 \
                                              --n_epochs 1000 \
                                              --lr 0.01 
```

#### Density estimation with missing values

Fit UCI POWER dataset using MDMA:

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

Density estimation using BNAF on the same dataset after performing MICE imputation:

```
python3 experiments/BNAF/density_estimation.py --dataset gas \
                                               --hidden_dim 320 \
                                               --missing_data_pct 0.5 \
                                               --missing_data_strategy mice
```

#### Mutual information estimation

Generate data from a multivariate gaussian, fit the joint density using MDMA and estimate the mutual information between subsets of variables:

```
python3 experiments/MI_estimation.py
```

#### Causal discovery 
Requires 
* **``R>=4.0.5``** 

as well as the python packages
* **``cdt>=0.5.23``** 
* **``rpy2>=3.4.4``** 
* **``pytorch>=1.0.0``**

Run the causal discovery experiment, recovering a causal graph from data by testing for conditional independence using MDMA:

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