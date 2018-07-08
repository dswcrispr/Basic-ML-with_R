featureNormalize = function(X) {
  mu = colMeans(X) # 1 by n, n is a number of features
  std = apply(X, 2, sd) # use apply function to get sd, '2' for columns
  mu_matrix = rep(mu, rep.int(nrow(X), ncol(X)))
  std_matrix = rep(std, rep.int(nrow(X), ncol(X)))
  X_norm = (X - mu_matrix)/std_matrix
  
  list(X_norm = X_norm, mu = mu, std = std)
}
  
  
  
  