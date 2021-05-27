library(evd)
library(ald)

### The data generating process H_1 and A_1

hard_proc_0 <- function(snr = 1, rand = F) {
  if (!rand) {
    beta1 <- snr * rnorm(50)
    beta2 <- snr * rnorm(50)
    alpha1 <- snr * rnorm(50)
    alpha2 <- snr * rnorm(50)
  }

  fun <- function(n, d) {
    if (rand) {
      beta1 <- snr * rnorm(50)
      beta2 <- snr * rnorm(50)
      alpha1 <- snr * rnorm(50)
      alpha2 <- snr * rnorm(50)
    }

    Z <- matrix(runif(n * d, -1, 1), nrow = n, ncol = d)

    X <- (cbind(Z, Z^2) %*% beta1[1:(2 * d)]) +
      exp(-abs(cbind(Z, Z^2) %*% alpha1[1:(2 * d)])) * rALD(n, p = 0.8)

    Y <- (cbind(Z, Z^2) %*% beta2[1:(2 * d)]) +
      exp(-abs(cbind(Z, Z^2) %*% alpha2[1:(2 * d)])) * rgumbel(n)
    list(X = X, Y = Y, Z = Z)
  }

  return(fun)
}

hard_proc_A <- function(snr = 1, rand = F) {
  if (!rand) {
    beta1 <- snr * sample(c(-1, 1), 50, T)
    beta2 <- snr * sample(c(-1, 1), 50, T)
    alpha1 <- snr * sample(c(-1, 1), 50, T)
    alpha2 <- snr * sample(c(-1, 1), 50, T)
  }

  fun <- function(n, d) {
    if (rand) {
      beta1 <- snr * sample(c(-1, 1), 50, T)
      beta2 <- snr * sample(c(-1, 1), 50, T)
      alpha1 <- snr * sample(c(-1, 1), 50, T)
      alpha2 <- snr * sample(c(-1, 1), 50, T)
    }

    Z <- matrix(runif(n * d, -1, 1), nrow = n, ncol = d)

    X <- (cbind(Z, Z^2) %*% beta1[1:(2 * d)]) +
      exp(-abs(cbind(Z, Z^2) %*% alpha1[1:(2 * d)])) * rALD(n, p = 0.8)

    Y <- (cbind(Z, Z^2, X, X^2) %*% beta2[1:(2 * (d + 1))]) +
      exp(-abs(cbind(Z, Z^2, X, X^2) %*% alpha2[1:(2 * (d + 1))])) * rgumbel(n)

    list(X = X, Y = Y, Z = Z)
  }

  return(fun)
}

### Linear Gaussian H_2 and A_2

lin_gauss_0 <- function(snr = 1, rand = F) {
  if (!rand) {
    beta1 <- snr * rnorm(50)
    beta2 <- snr * rnorm(50)
  }

  fun <- function(n, d) {
    if (rand) {
      beta1 <- snr * rnorm(50)
      beta2 <- snr * rnorm(50)
    }

    Z <- matrix(runif(n * d, min = -1, max = 1), ncol = d, nrow = n)
    X <- (Z %*% beta1[1:d]) + rnorm(n)
    Y <- (Z %*% beta2[1:d]) + rnorm(n)
    list(X = X, Y = Y, Z = Z)
  }
  return(fun)
}

lin_gauss_A <- function(snr = 1, rand = F) {
  if (!rand) {
    beta1 <- snr * sample(c(-1, 1), 50, T)
    beta2 <- snr * sample(c(-1, 1), 50, T)
  }

  fun <- function(n, d) {
    if (rand) {
      beta1 <- snr * sample(c(-1, 1), 50, T)
      beta2 <- snr * sample(c(-1, 1), 50, T)
    }

    Z <- matrix(runif(n * d, min = -1, max = 1), ncol = d, nrow = n)
    X <- (Z %*% beta1[1:d]) + rnorm(n)
    Y <- (cbind(Z, X) %*% beta2[1:(d + 1)]) + rnorm(n)
    list(X = X, Y = Y, Z = Z)
  }
  return(fun)
}

### Non-linear Gaussian H_3 and A_3

non_lin_gauss_0 <- function(snr = 1, rand = F) {
  if (!rand) {
    beta1 <- snr * rnorm(50)
    beta2 <- snr * rnorm(50)
  }

  fun <- function(n, d) {
    if (rand) {
      beta1 <- snr * rnorm(50)
      beta2 <- snr * rnorm(50)
    }

    Z <- matrix(runif(n * d, min = -1, max = 1), ncol = d, nrow = n)
    X <- (cbind(Z, Z^2) %*% beta1[1:(2 * d)]) + rnorm(n)
    Y <- (cbind(Z, Z^2) %*% beta2[1:(2 * d)]) + rnorm(n)
    list(X = X, Y = Y, Z = Z)
  }
  return(fun)
}

non_lin_gauss_A <- function(snr = 1, rand = F) {
  if (!rand) {
    beta1 <- snr * sample(c(-1, 1), 50, T)
    beta2 <- snr * sample(c(-1, 1), 50, T)
  }

  fun <- function(n, d) {
    if (rand) {
      beta1 <- snr * sample(c(-1, 1), 50, T)
      beta2 <- snr * sample(c(-1, 1), 50, T)
    }

    Z <- matrix(runif(n * d, min = -1, max = 1), ncol = d, nrow = n)
    X <- (cbind(Z, Z^2) %*% beta1[1:(2 * d)]) + rnorm(n)
    Y <- (cbind(Z, Z^2, X, X^2) %*% beta2[1:(2 * (d + 1))]) + rnorm(n)
    list(X = X, Y = Y, Z = Z)
  }
  return(fun)
}
### Heterogeneous Gaussian H_4 and A_4

hetero_gauss_0 <- function(snr = 1, rand = F) {
  if (!rand) {
    beta1 <- snr * rnorm(50)
    beta2 <- snr * rnorm(50)
  }

  fun <- function(n, d) {
    if (rand) {
      beta1 <- snr * rnorm(50)
      beta2 <- snr * rnorm(50)
    }

    Z <- matrix(runif(n * d, min = -2, max = 2), ncol = d, nrow = n)
    X <- (cbind(Z, Z^2) %*% beta1[1:(2 * d)]) * rnorm(n)
    Y <- (cbind(Z, Z^2) %*% beta2[1:(2 * d)]) * rnorm(n)
    list(X = X, Y = Y, Z = Z)
  }
  return(fun)
}

hetero_gauss_A <- function(snr = 1, rand = F) {
  if (!rand) {
    beta1 <- snr * sample(c(-1, 1), 50, T)
    beta2 <- snr * sample(c(-1, 1), 50, T)
  }

  fun <- function(n, d) {
    if (rand) {
      beta1 <- snr * sample(c(-1, 1), 50, T)
      beta2 <- snr * sample(c(-1, 1), 50, T)
    }

    Z <- matrix(runif(n * d, min = -2, max = 2), ncol = d, nrow = n)
    X <- (cbind(Z, Z^2) %*% beta1[1:(2 * d)]) * rnorm(n)
    Y <- (cbind(Z, Z^2, X, X^2) %*% beta2[1:(2 * (d + 1))]) * rnorm(n)
    list(X = X, Y = Y, Z = Z)
  }
  return(fun)
}