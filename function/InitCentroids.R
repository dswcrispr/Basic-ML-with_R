InitCentroids = function(X, K) {
  
  m = dim(X)[1]
  
  # pick K numbers in m randomly
  k = sample(1:m, K)
  initcentroids = X[k, ]
  initcentroids
}