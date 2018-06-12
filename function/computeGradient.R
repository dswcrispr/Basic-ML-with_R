# Computing gradient for optimization

computeGradient = function(X, y, lambda) {
  function(theta) {
    gradient = c(rep(0, dim(X)[2]))
    m = length(y)
    X_theta = X %*% theta
    h = 1 / (1 + exp(-X_theta))
    gradient = (t(X) %*% (h - y)) / m
    gradient_w_regular = gradient[-1] + ((lambda / m) * theta[-1])
    gradient = c(gradient[1], gradient_w_regular)
    gradient
  }
}