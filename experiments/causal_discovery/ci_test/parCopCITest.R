### Dependencies ------------------------------------------------------------------------
library(quantreg)
library(splines)

### Main functions ----------------------------------------------------------------------

test_CI <- function(X, Y, Z = NULL, m = NULL, alpha = 0.05,
                    tau_min = 0.01, tau_max = 0.99, delta = 1e-2, q = c(1, 2, 3),
                    quantile_reg = "linear", bspline_df = 3, poly_deg = 3, return_all = F) {
  # Creating a list with all relevant information
  ci_obj <- create_ci_obj(
    X = X, Y = Y, Z = Z, m = m, alpha = alpha,
    tau_min = tau_min, tau_max = tau_max, delta = delta, q = q,
    quantile_reg = quantile_reg,
    bspline_df = bspline_df, poly_deg, return_all = return_all
  )

  # If there is no conditioning variable Z, then we transform
  # with the ecdf of X and Y to return U1 and U2. Else we perform
  # quantile regression, do predictions and transform with Fhat
  if (is.null(Z)) {
    ci_obj <- compute_u_without_z(ci_obj)
  } else {
    ci_obj <- quant_reg(ci_obj)
    ci_obj <- compute_u_with_z(ci_obj)
  }

  # Testing independence between U1 and U2
  ci_obj <- test_independence(ci_obj)

  # Returning either all information about the test or essentials
  if (return_all) {
    return(ci_obj)
  } else {
    return(list(
      statistic = ci_obj$statistic,
      p_value = ci_obj$p_value,
      q_vals = ci_obj$q
    ))
  }
}


create_ci_obj <- function(X, Y, Z, m, alpha, tau_min, tau_max, delta, q,
                          quantile_reg, bspline_df, poly_deg, return_all) {
  n <- length(X)
  d <- ncol(Z)
  if (is.null(m)) {
    m <- ceiling(sqrt(n))
  }

  X <- standardize(X)
  Y <- standardize(Y)
  if (!is.null(Z)) {
    Z <- standardize(Z)
  }

  list(
    X = X, Y = Y, Z = Z,
    n = n, m = m, d = d, alpha = alpha,
    tau_min = tau_min, tau_max = tau_max, delta = delta, q = q,
    quantile_reg = quantile_reg,
    bspline_df = bspline_df,
    poly_deg = poly_deg,
    return_all = return_all
  )
}


### Quantile regression models ----------------------------------------------------------

quant_reg <- function(ci_obj) {
  m <- ci_obj$m
  tau_min <- ci_obj$tau_min
  tau_max <- ci_obj$tau_max
  tau_seq <- create_tau_seq(m, tau_min = tau_min, tau_max = tau_max)

  dat <- data.frame(x = ci_obj$X, y = ci_obj$Y, z = ci_obj$Z)
  z_names <- names(dat)[-c(1, 2)]

  if (ci_obj$quantile_reg == "linear") {
    form_x <- as.formula(paste("x", paste(z_names, collapse = "+"), sep = "~"))
    form_y <- as.formula(paste("y", paste(z_names, collapse = "+"), sep = "~"))
  }
  if (ci_obj$quantile_reg == "polynomial") {
    form_x <- as.formula(paste("x", paste("poly(", z_names, ", degree=", ci_obj$poly_deg, ")", collapse = "+"), sep = "~"))
    form_y <- as.formula(paste("y", paste("poly(", z_names, ", degree=", ci_obj$poly_deg, ")", collapse = "+"), sep = "~"))
  }
  if (ci_obj$quantile_reg == "B-Spline") {
    form_x <- as.formula(paste("x", paste("bs(", z_names, ", degree=", ci_obj$bspline_df, ")", collapse = "+"), sep = "~"))
    form_y <- as.formula(paste("y", paste("bs(", z_names, ", degree=", ci_obj$bspline_df, ")", collapse = "+"), sep = "~"))
  }

  reg_x <- rq(data = dat, formula = form_x, tau = tau_seq)
  reg_y <- rq(data = dat, formula = form_y, tau = tau_seq)

  ci_obj$quant_pred_x <- predict(reg_x)
  ci_obj$quant_pred_y <- predict(reg_y)

  return(ci_obj)
}


### Computing U1 and U2 -----------------------------------------------------------------

compute_u_without_z <- function(ci_obj) {
  ci_obj$U1 <- ecdf(ci_obj$X)(ci_obj$X)
  ci_obj$U2 <- ecdf(ci_obj$Y)(ci_obj$Y)
  return(ci_obj)
}


compute_u_with_z <- function(ci_obj) {
  tmp_x <- cbind(ci_obj$X, ci_obj$quant_pred_x)
  tmp_y <- cbind(ci_obj$Y, ci_obj$quant_pred_y)

  transform_to_u_x <- function(x) {
    xx <- x[1]
    preds <- x[-1]
    m <- ci_obj$m
    x_coord <- sort(c(0, preds, 1))
    y_coord <- c(0, seq(ci_obj$tau_min, ci_obj$tau_max, length = m), 1)
    Fhat <- approxfun(x = x_coord, y = y_coord, ties = "mean")
    Fhat(xx)
  }

  transform_to_u_y <- function(x) {
    xx <- x[1]
    preds <- x[-1]
    m <- ci_obj$m
    x_coord <- sort(c(0, preds, 1))
    y_coord <- c(0, seq(ci_obj$tau_min, ci_obj$tau_max, length = m), 1)
    Fhat <- approxfun(x = x_coord, y = y_coord, ties = "mean")
    Fhat(xx)
  }

  ci_obj$U1 <- apply(tmp_x, 1, transform_to_u_x)
  ci_obj$U2 <- apply(tmp_y, 1, transform_to_u_y)

  return(ci_obj)
}


transform_to_u <- function(x) {
  xx <- x[1]
  preds <- x[-1]
  m <- length(preds) + 1

  x_coord <- sort(c(0, preds, 1))
  y_coord <- seq(from = 0, to = 1, length = m + 1)
  Fhat <- approxfun(x = x_coord, y = y_coord, ties = "mean")
  Fhat(xx)
}


### Performing independence test between U1 and U2 --------------------------------------

test_independence <- function(ci_obj) {
  U1 <- ci_obj$U1
  U2 <- ci_obj$U2
  qseq <- ci_obj$q
  pval <- statistic <- numeric(length(qseq))
  k <- 1

  for (q in qseq) {
    rho_fun <- create_rho_func(
      q = q, tau_min = ci_obj$tau_min,
      tau_max = ci_obj$tau_max, delta = ci_obj$delta
    )
    rho_hat <- rho_fun(U1, U2)
    statistic[k] <- ci_obj$n * t(rho_hat) %*% rho_hat
    pval[k] <- pchisq(q = statistic[k], df = q^2, lower.tail = FALSE)
    k <- k + 1
  }

  ci_obj$p_value <- pval
  ci_obj$statistic <- statistic

  return(ci_obj)
}


### Help functions ----------------------------------------------------------------------

standardize <- function(x) {
  if (is.matrix(x)) {
    for (i in 1:ncol(x)) {
      x[, i] <- ecdf(x[, i])(x[, i])
    }
  } else {
    x <- ecdf(x)(x)
  }
  return(x)
}


create_tau_seq <- function(m, tau_min, tau_max) {
  taus <- seq(from = tau_min, to = tau_max, length = m)
  return(taus)
}


create_phi_func <- function(mu, lambda, delta) {
  # Normalization constant K
  K <- 1 / (lambda - mu - delta)
  # Trimming function sigma
  sigma <- function(u) {
    K * (u >= mu + delta & u <= lambda - delta) +
      K * (u - mu) / delta * (u >= mu & u < mu + delta) +
      K * (lambda - u) / delta * (u > lambda - delta & u <= lambda)
  }

  # Making the phi function centered and scaled
  m <- integrate(function(u) u * sigma(u), lower = 0, upper = 1)$value
  c <- 1 / sqrt(
    integrate(function(u) (u - m)^2 * sigma(u)^2, lower = 0, upper = 1)$value
  )

  # Defining phi
  phi <- function(u) {
    c * (u - m) * sigma(u)
  }

  return(phi)
}


create_rho_func <- function(q, tau_min, tau_max, delta) {
  phi_funs <- list()
  tmp <- seq(tau_min, tau_max, length = q + 1)
  for (i in 1:q) {
    phi_funs[[i]] <- create_phi_func(mu = tmp[i], lambda = tmp[i + 1], delta = delta)
  }

  rho_fun <- function(U1, U2) {
    res <- matrix(NA, nrow = q, ncol = q)
    for (k in 1:q) {
      for (l in 1:q) {
        res[k, l] <- mean(phi_funs[[k]](U1) * phi_funs[[l]](U2))
      }
    }
    as.vector(res)
  }
  rho_fun
}