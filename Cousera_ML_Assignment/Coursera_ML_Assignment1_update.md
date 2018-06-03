Linear regression
================

1.Linear regression with one variable
=====================================

### 1.1 Plotting the Data

``` r
data = read.table("C:/Users/user/Documents/Basic-ML-with_R/data/ex1data1.txt", sep = ',')
X_1 = data[, 1]
y = data[, 2]
m = length(y) # number of training examples

plotData = function (x, y) {
  plot(
    x, y, col = "red", pch = 4, cex = 1.1, lwd = 2,
    xlab = 'profit in $10,000s',
    ylab = 'population of City in 10,000s'
  )
}
plotData(X_1, y) # using plotData function
```

![](Coursera_ML_Assignment1_update_files/figure-markdown_github-ascii_identifiers/plotting%20data-1.png)

### 1.2 Gradient descent

#### 1.2.1 Settings

``` r
X = cbind(rep(1, m), X_1)
X = as.matrix(X) # from data.frame to matrix

theta = c(0, 0) # initializing fitting parameters

iterations = 1500
alpha = 0.01
```

#### 1.2.2 Computing the cost J(theta)

``` r
computeCost = function(X, y, theta) {
  J = 0 
  m = length(y)
    
  h_x = X %*% theta
  res = h_x - y
  J = (t(res) %*% res) / (2 * m)
  J
}

# print result to screen
sprintf('Cost J: %.3f', computeCost(X, y, theta))
```

    ## [1] "Cost J: 32.073"

#### 1.2.3 Setting Gradient descent algorithm

``` r
gradientDescent = function(X, y, theta, alpha, num_iters) {
  m = length(y)
  J_history = rep(0, num_iters + 1) # making room for saving J
  theta_history = matrix(0, num_iters + 1, length(theta)) # making room for saving theta
  theta_history[1, ] = t(theta) # saving initial vlaue to matrix
  J_history[1] = computeCost(X, y, theta) # saving initial vlaue to matrix
  
  for (i in 2 : (num_iters + 1)) {
    
    theta_prev = theta # create a copy of theta for simultaneous update
    
    # simultaneous update theta using theta_prev, using vectorized method
    
    deriv_J = (t(X) %*% ((X%*% theta_prev) - y)) / m
    theta = theta - (alpha * deriv_J)
    
    # updating J_history and theta_history
    
    J_history[i] = computeCost(X, y, theta)
    theta_history[i, ] = t(theta)
  }
  
  list(theta = theta, J_history = J_history, theta_history = theta_history)
}
```

#### 1.2.4 Run Gradient descent algorithm

``` r
gd = gradientDescent(X, y, theta, alpha, iterations)

# Saving results from list variables into global env variables
theta = gd$theta
J_history = gd$J_history
theta_history = gd$theta_history
rm(gd) # remove gd

# print theta to screen
sprintf('Theta found by gradient descent: %.3f %.3f', theta[1], theta[2])
```

    ## [1] "Theta found by gradient descent: -3.630 1.166"

``` r
# check whether gradient descent worked correctly 
n_iter = c(0:iterations)
plot(n_iter, J_history, xlab = 'n of iterations', ylab = 'Cost J',
     ylim = c(4, 7), cex = 0.1, col = "blue")
```

![](Coursera_ML_Assignment1_update_files/figure-markdown_github-ascii_identifiers/checking-1.png)

``` r
# plot the linear fit
# keep previous plot visible
plotData(X_1, y) # using plotData function
lines(X[, 2], X %*% theta, col = "blue")
legend("bottomright", c('Training data', 'Linear regression'), pch=c(4,NA),col=c("red","blue"), lty=c(NA,1) )
```

![](Coursera_ML_Assignment1_update_files/figure-markdown_github-ascii_identifiers/plotting-1.png)

``` r
# Predict values for population sizes of 35,000 and 70,000
predict1 = c(1, 3.5) %*% theta
sprintf('For population = 35,000, we predict a profit of %f',predict1*10000)
```

    ## [1] "For population = 35,000, we predict a profit of 4519.767868"

``` r
predict2 <- c(1, 7) %*% theta
sprintf('For population = 70,000, we predict a profit of %f',predict2*10000)
```

    ## [1] "For population = 70,000, we predict a profit of 45342.450129"

### 1.3 Visualization

#### 1.3.1 Visualizing J(theta\_0, theta\_1), Contour

``` r
# Grid over which we will calculate J
theta0_vals = seq(-10, 10, length.out = 100)
theta1_vals = seq(-1, 4, length.out = 100)

# initialize J_vals to a matrix of 0's
J_vals = matrix(0, length(theta0_vals), length(theta1_vals))

# Fill out J_vals
for (i in 1 : length(theta0_vals)) {
  for (j in 1 : length(theta1_vals)) {
    J_vals[i, j] = computeCost(X, y, c(theta0_vals[i], theta1_vals[j]))
  }
}


# plot J-vals as 20 countours spaced logarithmically between 0.01 and 100
# logarithmic contours are denser near the center

logspace = function(d1, d2, n) {
  return(exp(log(10) * seq(d1, d2, length.out = n)))
}

contour(theta0_vals, theta1_vals, J_vals, levels = logspace(-2, 3, 20),
        xlab = expression(theta_0),
        ylab = expression(theta_1),
        drawlabels = FALSE)

points(theta[1], theta[2], pch = 4, cex = 2, col = "red", lwd =2)
points(theta_history[, 1], theta_history[, 2], col = "red", cex = 0.2, lwd = 1, pch =19)
lines(theta_history[, 1], theta_history[, 2], col = "red")
```

![](Coursera_ML_Assignment1_update_files/figure-markdown_github-ascii_identifiers/contour-1.png)

2.Linear regression with multiple variables
===========================================

### 2.1 Feature Normalization

``` r
data = read.table("C:/Users/user/Documents/Basic-ML-with_R/data/ex1data2.txt", sep = ',') # data loading

X = data[, 1:2]
y = data[, 3]
m = length(y) # number of training examples

featureNormalize = function(X) {
  mu = colMeans(X) # 1 by n, n is a number of features
  std = apply(X, 2, sd) # use apply function to get sd, '2' for columns
  mu_matrix = rep(mu, rep.int(nrow(X), ncol(X)))
  std_matrix = rep(std, rep.int(nrow(X), ncol(X)))
  X_norm = (X - mu_matrix)/std_matrix
  
  list(X_norm = X_norm, mu = mu, std = std)
}
  
X_n = featureNormalize(X)$X_norm # saving normalized x in X_n
mu = featureNormalize(X)$mu
std = featureNormalize(X)$std

X_n = cbind(rep(1, m), X_n) # appending intercept term
```

### 2.2 Gradient Descent

``` r
X_n = as.matrix(X_n) # from data.frame to matrix

# initializing fitting parameter
theta = c(rep(0, dim(X_n)[2]))


# some gradient descent settings
iterations = 400
alpha = 0.1

# Computing Cost J
computeCost(X_n, y, theta)
```

    ##             [,1]
    ## [1,] 65591548106

``` r
# run Gradient Descent process
gd = gradientDescent(X_n, y, theta, alpha, iterations)

# form list variables into global env variables
theta = gd$theta
J_history = gd$J_history
theta_history = gd$theta_history
rm(gd) # remove gd

# print theta to screen
sprintf('Thetas found by gradient descent: %.3f %.3f', theta[1], theta[2], theta[3])
```

    ## [1] "Thetas found by gradient descent: 340412.660 110631.049"

### 2.3 Visualization

``` r
# Cost J decreasing with different alphas 

alpha_0.01 = gradientDescent(X_n, y, c(rep(0, dim(X_n)[2])), 0.01, iterations)$J_history
alpha_0.03 = gradientDescent(X_n, y, c(rep(0, dim(X_n)[2])), 0.03, iterations)$J_history
alpha_0.1 = gradientDescent(X_n, y, c(rep(0, dim(X_n)[2])), 0.1, iterations)$J_history
alpha_0.3 = gradientDescent(X_n, y, c(rep(0, dim(X_n)[2])), 0.3, iterations)$J_history

df = data.frame(n_iter = c(0:100),
     alpha_0.01 = alpha_0.01[1:101]/(10^10), alpha_0.03 = alpha_0.03[1:101]/(10^10),
     alpha_0.1 = alpha_0.1[1:101]/(10^10), alpha_0.3 = alpha_0.3[1:101]/(10^10))

library(ggplot2)
library(reshape2) # to use melt function
```

    ## Warning: package 'reshape2' was built under R version 3.4.4

``` r
df_long = melt(df, id = "n_iter")
ggplot(data = df_long, aes(x = n_iter, y = value, colour = variable)) +
  geom_line(size = 1) + ylim(0, 7) + xlim(0, 100) + ylab("Cost J(10^10)") +
  xlab("number of iteration") + ggtitle("Decreasing cost J with different alpha")
```

![](Coursera_ML_Assignment1_update_files/figure-markdown_github-ascii_identifiers/visualization-1.png)
