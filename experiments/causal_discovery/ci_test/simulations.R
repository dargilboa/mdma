source('parCopCITest.R')
source('gcm.R')
source('npn.R')
library(tidyverse)
library(furrr)
library(ppcor)
library(evd)
library(ald)

### Proof of concept simulations: the hypothesis test actually works --------------------

sim_study_1 <- function(n_obs, d_dim, n_sim, data_gen_proc, 
                        method='B-Spline', bspline_df = 3, poly_deg = 3) {
  
  fun <- function(n, d, i) {
    # Generate data
    dat <- data_gen_proc(n, d)
    X <- dat$X
    Y <- dat$Y
    Z <- dat$Z
    
    # Perform partial copula test (cor and hsic)
    tryCatch({
      test <- test_CI(X=X, Y=Y, Z=Z,
                      quantile_reg = method,
                      bspline_df = bspline_df, 
                      poly_deg = poly_deg,
                      q = c(1, 2, 3, 4, 5), 
                      tau_min = 0.01,
                      tau_max = 0.99, 
                      delta = 0.01)
      
      pvalq1 <- test$p_value[1]
      pvalq2 <- test$p_value[2]
      pvalq3 <- test$p_value[3]
      pvalq4 <- test$p_value[4]
      pvalq5 <- test$p_value[5]
      
      c(pvalq1, pvalq2, pvalq3, pvalq4, pvalq5)}, 
      error = function(e) {
        rep(NA, 5)
      })
  }
  
  results <- expand_grid(n=n_obs, d=d_dim, i=1:n_sim) %>%
    mutate(p_value = future_pmap(.l = list(n, d, i), .f = fun)) %>% 
    unnest(cols=c(p_value)) %>% 
    mutate(test=factor(rep(c('pvalq1', 'pvalq2', 'pvalq3', 'pvalq4', 'pvalq5'),
                           length(n_obs) * length(d_dim) * n_sim)))
  
  results
}

### The data generating process H_1 and A_1

hard_proc_0 <- function(snr=1, rand=F) {
  if (!rand) {
    beta1  <- snr * rnorm(50)
    beta2  <- snr * rnorm(50)
    alpha1 <- snr * rnorm(50)
    alpha2 <- snr * rnorm(50)
  }
  
  fun <- function(n, d) {
    if (rand) {
      beta1  <- snr * rnorm(50)
      beta2  <- snr * rnorm(50)
      alpha1 <- snr * rnorm(50)
      alpha2 <- snr * rnorm(50)
    }
    
    Z <- matrix(runif(n*d, -1, 1), nrow=n, ncol=d)
    
    X <- (cbind(Z, Z^2) %*% beta1[1:(2*d)]) + 
      exp(-abs(cbind(Z, Z^2) %*% alpha1[1:(2*d)])) * rALD(n, p=0.8)
    
    Y <- (cbind(Z, Z^2) %*% beta2[1:(2*d)]) + 
      exp(-abs(cbind(Z, Z^2) %*% alpha2[1:(2*d)])) * rgumbel(n)
    list(X=X, Y=Y, Z=Z)
  }
  
  return(fun)
}

hard_proc_A <- function(snr=1, rand=F) {
  if (!rand) {
    beta1  <- snr * sample(c(-1, 1), 50, T)
    beta2  <- snr * sample(c(-1, 1), 50, T)
    alpha1 <- snr * sample(c(-1, 1), 50, T)
    alpha2 <- snr * sample(c(-1, 1), 50, T)
  }
  
  fun <- function(n, d) {
    if (rand) {
      beta1  <- snr * sample(c(-1, 1), 50, T)
      beta2  <- snr * sample(c(-1, 1), 50, T)
      alpha1 <- snr * sample(c(-1, 1), 50, T)
      alpha2 <- snr * sample(c(-1, 1), 50, T)
    }
    
    Z <- matrix(runif(n*d, -1, 1), nrow=n, ncol=d)
    
    X <- (cbind(Z, Z^2) %*% beta1[1:(2*d)]) + 
      exp(-abs(cbind(Z, Z^2) %*% alpha1[1:(2*d)])) * rALD(n, p=0.8)
    
    Y <- (cbind(Z, Z^2, X, X^2) %*% beta2[1:(2*(d+1))]) + 
      exp(-abs(cbind(Z, Z^2, X, X^2) %*% alpha2[1:(2*(d+1))])) * rgumbel(n)
    
    list(X=X, Y=Y, Z=Z)
  }
  
  return(fun)
}

### The simulations
set.seed(1)
plan(multisession, workers = 60)

n0 <- seq(100, 2000, by=100)
nA <- seq(10, 500, by=10)
n_sim_0 <- 2000
n_sim_A <- 500
d_dim <- c(1, 5, 10)

POC_0 <- sim_study_1(n_obs = n0,
                     d_dim = d_dim,
                     n_sim = n_sim_0,
                     method = 'B-Spline',
                     bspline_df = 5,
                     data_gen_proc = hard_proc_0(rand=T, snr=1))

POC_A <- sim_study_1(n_obs = nA,
                     d_dim = d_dim,
                     n_sim = n_sim_A,
                     method = 'B-Spline',
                     bspline_df = 5,
                     data_gen_proc = hard_proc_A(rand=T, snr=1))

### Comparison study: PC Cor, PC HSIC, GCM (HSIC), Partial Spearman ---------------------

sim_study_2 <- function(n_obs, d_dim, n_sim, data_gen_proc, 
                        method='B-Spline', 
                        method_gcm_2 = 'B-Spline',
                        bspline_df = 3, 
                        poly_deg = 3, 
                        poly_deg_gcm2 = 3) {
  
  fun <- function(n, d, i) {
    
    # Generate data
    dat <- data_gen_proc(n, d)
    X <- dat$X
    Y <- dat$Y
    Z <- dat$Z
    
    tryCatch({
      # Partial copula conditional independence test
      test <- test_CI(X=X, Y=Y, Z=Z,
                      quantile_reg = method,
                      bspline_df = bspline_df, 
                      poly_deg = poly_deg,
                      q = c(1, 2, 3, 4, 5), 
                      tau_min = 0.01,
                      tau_max = 0.99, 
                      delta = 0.01)
      
      pvalq1 <- test$p_value[1]
      pvalq2 <- test$p_value[2]
      pvalq3 <- test$p_value[3]
      pvalq4 <- test$p_value[4]
      pvalq5 <- test$p_value[5]
      
      # GCM conditional independence test
      tmp <- gcm_custom(X=X, Y=Y, Z=Z, method=method_gcm_2, 
                        poly_deg=poly_deg, bspline_df=bspline_df)
      pvalgcm <- tmp$p_value_cor
      
      # GCM with squared X and Y
      tmp <- gcm_custom(X=X^2, Y=Y^2, Z=Z, method='custom', custom_poly_deg = poly_deg_gcm2)
      pvalgcm_squared <- tmp$p_value_cor
      
      # NPN conditional independence test
      
      pvalnpn <- npn_test(X=X, Y=Y, Z=Z)
    
      # Return the p-values
      c(pvalq1,
        pvalq2, 
        pvalq3, 
        pvalq4, 
        pvalq5,
        pvalgcm, 
        pvalgcm_squared,
        pvalnpn)
    }, 
    error = function(e) {
      rep(NA, 8)
    })
  }
  
  results <- expand_grid(n=n_obs, d=d_dim, i=1:n_sim) %>%
    mutate(p_value = future_pmap(.l = list(n, d, i), .f = fun)) %>% 
    unnest(cols=c(p_value)) %>% 
    mutate(test=factor(rep(c('pvalq1', 'pvalq2', 'pvalq3', 'pvalq4', 'pvalq5',
                             'gcm', 'gcm squared', 'npn'),
                           length(n_obs) * length(d_dim) * n_sim)))
  
  results
}

# fixing the dimension
d_dim <- 3

### Linear Gaussian H_2 and A_2

lin_gauss_0 <- function(snr=1, rand=F) {
  if (!rand) {
    beta1  <- snr * rnorm(50)
    beta2  <- snr * rnorm(50)
  }
  
  fun <- function(n, d) {
    if (rand) {
      beta1  <- snr * rnorm(50)
      beta2  <- snr * rnorm(50)
    }
    
    Z <- matrix(runif(n*d, min = -1, max = 1), ncol=d, nrow=n)
    X <- (Z %*% beta1[1:d]) + rnorm(n)
    Y <- (Z %*% beta2[1:d]) + rnorm(n)
    list(X=X, Y=Y, Z=Z)
  }
  return(fun)
}

lin_gauss_A <- function(snr=1, rand=F) {
  if (!rand) {
    beta1  <- snr * sample(c(-1, 1), 50, T)
    beta2  <- snr * sample(c(-1, 1), 50, T)
  }
  
  fun <- function(n, d) {
    if (rand) {
      beta1  <- snr * sample(c(-1, 1), 50, T)
      beta2  <- snr * sample(c(-1, 1), 50, T)
    }
    
    Z <- matrix(runif(n*d, min = -1, max = 1), ncol=d, nrow=n)
    X <- (Z %*% beta1[1:d]) + rnorm(n)
    Y <- (cbind(Z, X) %*% beta2[1:(d+1)]) + rnorm(n)
    list(X=X, Y=Y, Z=Z)
  }
  return(fun)
}

exp_lin_gauss_0 <- sim_study_2(n_obs = n0, 
                               d_dim = d_dim, 
                               n_sim = n_sim_0,
                               method = 'linear',
                               method_gcm_2 = 'polynomial', 
                               poly_deg_gcm2 = 2,
                               data_gen_proc = lin_gauss_0(rand=T, snr=1))

exp_lin_gauss_A <- sim_study_2(n_obs = nA, 
                               d_dim = d_dim, 
                               n_sim = n_sim_A,
                               method = 'linear',
                               method_gcm_2 = 'polynomial', 
                               poly_deg_gcm2 = 2,
                               data_gen_proc = lin_gauss_A(rand=T, snr=1))

### Non-linear Gaussian H_3 and A_3

non_lin_gauss_0 <- function(snr=1, rand=F) {
  if (!rand) {
    beta1  <- snr * rnorm(50)
    beta2  <- snr * rnorm(50)
  }
  
  fun <- function(n, d) {
    if (rand) {
      beta1  <- snr * rnorm(50)
      beta2  <- snr * rnorm(50)
    }
    
    Z <- matrix(runif(n*d, min = -1, max = 1), ncol=d, nrow=n)
    X <- (cbind(Z, Z^2) %*% beta1[1:(2*d)]) + rnorm(n)
    Y <- (cbind(Z, Z^2) %*% beta2[1:(2*d)]) + rnorm(n)
    list(X=X, Y=Y, Z=Z)
  }
  return(fun)
}

non_lin_gauss_A <- function(snr=1, rand=F) {
  if (!rand) {
    beta1  <- snr * sample(c(-1, 1), 50, T)
    beta2  <- snr * sample(c(-1, 1), 50, T)
  }
  
  fun <- function(n, d) {
    if (rand) {
      beta1  <- snr * sample(c(-1, 1), 50, T)
      beta2  <- snr * sample(c(-1, 1), 50, T)
    }
    
    Z <- matrix(runif(n*d, min = -1, max = 1), ncol=d, nrow=n)
    X <- (cbind(Z, Z^2) %*% beta1[1:(2*d)]) + rnorm(n)
    Y <- (cbind(Z, Z^2, X, X^2) %*% beta2[1:(2*(d+1))]) + rnorm(n)
    list(X=X, Y=Y, Z=Z)
  }
  return(fun)
}

exp_non_lin_gauss_0 <- sim_study_2(n_obs = n0, 
                                   d_dim = d_dim, 
                                   n_sim = n_sim_0,
                                   method = 'polynomial', 
                                   poly_deg = 2, 
                                   method_gcm_2 = 'polynomial', 
                                   poly_deg_gcm2 = 4,
                                   data_gen_proc = non_lin_gauss_0(rand=T, snr=1))

exp_non_lin_gauss_A <- sim_study_2(n_obs = nA, 
                                   d_dim = d_dim, 
                                   n_sim = n_sim_A,
                                   method = 'polynomial', 
                                   poly_deg = 2, 
                                   method_gcm_2 = 'polynomial', 
                                   poly_deg_gcm2 = 4,
                                   data_gen_proc = non_lin_gauss_A(rand=T, snr=1))

### Heterogeneous Gaussian H_4 and A_4

hetero_gauss_0 <- function(snr=1, rand=F) {
  if (!rand) {
    beta1  <- snr * rnorm(50)
    beta2  <- snr * rnorm(50)
  }
  
  fun <- function(n, d) {
    if (rand) {
      beta1  <- snr * rnorm(50)
      beta2  <- snr * rnorm(50)
    }
    
    Z <- matrix(runif(n*d, min = -2, max = 2), ncol=d, nrow=n)
    X <- (cbind(Z, Z^2) %*% beta1[1:(2*d)]) * rnorm(n)
    Y <- (cbind(Z, Z^2) %*% beta2[1:(2*d)]) * rnorm(n)
    list(X=X, Y=Y, Z=Z)
  }
  return(fun)
}

hetero_gauss_A <- function(snr=1, rand=F) {
  if (!rand) {
    beta1  <- snr * sample(c(-1, 1), 50, T)
    beta2  <- snr * sample(c(-1, 1), 50, T)
  }
  
  fun <- function(n, d) {
    if (rand) {
      beta1  <- snr * sample(c(-1, 1), 50, T)
      beta2  <- snr * sample(c(-1, 1), 50, T)
    }
    
    Z <- matrix(runif(n*d, min = -2, max = 2), ncol=d, nrow=n)
    X <- (cbind(Z, Z^2) %*% beta1[1:(2*d)]) * rnorm(n)
    Y <- (cbind(Z, Z^2, X, X^2) %*% beta2[1:(2*(d+1))]) * rnorm(n)
    list(X=X, Y=Y, Z=Z)
  }
  return(fun)
}

exp_hetero_gauss_0 <- sim_study_2(n_obs = n0, 
                                  d_dim = d_dim, 
                                  n_sim = n_sim_0,
                                  method = 'polynomial', 
                                  poly_deg = 2, 
                                  method_gcm_2 = 'polynomial', 
                                  poly_deg_gcm2 = 4,
                                  data_gen_proc = hetero_gauss_0(rand=T, snr=1))

exp_hetero_gauss_A <- sim_study_2(n_obs = nA, 
                                  d_dim = d_dim, 
                                  n_sim = n_sim_A,
                                  method = 'polynomial', 
                                  poly_deg = 2, 
                                  method_gcm_2 = 'polynomial', 
                                  poly_deg_gcm2 = 4,
                                  data_gen_proc = hetero_gauss_A(rand=T, snr=5))

# Saving the results

save(POC_0, POC_A,
     exp_lin_gauss_0, exp_lin_gauss_A,
     exp_non_lin_gauss_0, exp_non_lin_gauss_A,
     exp_hetero_gauss_0, exp_hetero_gauss_A,
     file='simulations.RData')
