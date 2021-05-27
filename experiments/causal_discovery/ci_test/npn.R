### The NPN correlation of Harris and Drton

npn_test <- function(X, Y, Z) {
  d <- ncol(Z)
  n <- length(X)
  
  # Estimate latent covariance matrix in the Gaussian copula
  M <- cor(cbind(X, Y, Z), method = 'spearman')
  M <- 2 * sin(pi/6 * M)
  
  # Invert to find concentration matrix
  M_inv <- solve(M)
  
  # Get the partial correlation of X and Y given Z
  p_cor <- -M_inv[1, 2] / sqrt(M_inv[1, 1] * M_inv[2, 2])
  
  # Fisher z-transform
  z <- sqrt(n - d - 3) * atanh(p_cor)
  
  # Compute p-value
  p_value <- 2 * pnorm(abs(z), lower.tail = F)
  p_value
}
