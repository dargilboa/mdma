# Simple implementation of the Generalized Covariance Measure
# for linear models and polynomial models

gcm_custom <- function(X, Y, Z, method='linear', poly_deg=2, bspline_df=2, custom_poly_deg=2) {
  Z <- as.matrix(Z)
  n <- nrow(Z)
  d <- ncol(Z)
  df <- data.frame(x=X, y=Y, z=Z)
  z_names <- names(df)[3:(d+2)]
  
  # Used for regression on the squares of X and Y in simulation study
  if (method=='custom') {
    reg_x = lm(X ~ poly(Z, degree=custom_poly_deg, raw=T))
    reg_y = lm(Y ~ poly(Z, degree=custom_poly_deg, raw=T))
  }
  
  if (method=='linear') {
    form_x <- as.formula(paste('x', paste(z_names, collapse='+'), sep='~'))
    form_y <- as.formula(paste('y', paste(z_names, collapse='+'), sep='~'))
  }
  
  if (method=='polynomial') {
    form_x <- as.formula(paste('x', paste('poly(', z_names, ', degree=', poly_deg, ')', collapse='+'), sep='~'))
    form_y <- as.formula(paste('y', paste('poly(', z_names, ', degree=', poly_deg, ')', collapse='+'), sep='~'))
  }
  
  if (method=='B-Spline') {
    form_x <- as.formula(paste('x', paste('bs(', z_names, ', degree=', bspline_df, ')', collapse='+'), sep='~'))
    form_y <- as.formula(paste('y', paste('bs(', z_names, ', degree=', bspline_df, ')', collapse='+'), sep='~'))
  }
  
  if (method != 'custom') {
    reg_x <- lm(data=df, formula = form_x)
    reg_y <- lm(data=df, formula = form_y)
  }
  resid_x <- reg_x$residuals
  resid_y <- reg_y$residuals
  
  R <- resid_x * resid_y
  tau_N <- sqrt(n) * mean(R)
  tau_D <- sqrt(mean(R^2) - mean(R)^2)
  test_stat <- tau_N / tau_D
  
  p_value_cor <- 2 * pnorm(abs(test_stat), lower.tail = F)

  list(resid_x=resid_x, resid_y=resid_y, 
       p_value_cor=p_value_cor)
}
