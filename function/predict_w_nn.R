predict_w_nn = function(theta1, theta2, X) {
  
  if (is.vector(X)) {
    X = t(X) # if input X was a vector, we should transform 
             # X to 1 by n matrix
  }
  
  m = dim(X)[1]
  X = cbind(c(rep(1, m)), X) # add bias unit to X
  z_2 = theta1 %*% t(X) # 25 by 5000
  a_2 = 1 / (1 + exp(-z_2)) # 25 by 5000
  a_2 = rbind(c(rep(1, dim(a_2)[2])), a_2) # add bias unit to a_2, 26 by 5000
  z_3 = theta2 %*% a_2 # 10 by 5000
  a_3 = 1 / (1 + exp(-z_3)) # 10 by 5000
  
  pred =  c(apply(a_3, 2, which.max)) # 5000 by 1
  list(pred = pred)
  
}