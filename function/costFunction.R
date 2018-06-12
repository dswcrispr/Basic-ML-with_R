costFunction = function(X, y, lambda) {
  function(theta) {
    J = 0
    m = length(y)
    X_theta = X %*% theta # m by 1
    h = 1 / (1 + exp(-1 * X_theta)) # m by 1
    reg_term = (t(theta[-1]) %*% theta[-1]) * (lambda / 2) # [-1] syntex make vector without 1st element
    J = (t(y) %*% log(h) + t(1 - y) %*% (log(1 - h)) - reg_term) / (-m)
    J
    
  }
}
