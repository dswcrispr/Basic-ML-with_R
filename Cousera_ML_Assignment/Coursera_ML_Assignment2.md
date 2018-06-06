Logistic regression
================

1.Logistic regression without regularization term
=================================================

### 1.1 Visualizing the data

``` r
data = read.table("C:/Users/user/Documents/Basic-ML-with_R/data/ex2data1.txt", sep = ',')
data_plot = data

# to make ggplot represent admission result
data_plot$V3 = ifelse(data$V3 == 1, 'Admitted', 'Not admitted')

# Plotting scatter plot
library(ggplot2)
p = ggplot(data_plot, aes(V1, V2)) + geom_point(aes(color = V3))+
    ylab("Exam 2 score") + xlab("Exam 1 score") +
    scale_x_continuous(breaks = seq(30, 100, 10)) +
    scale_y_continuous(breaks = seq(30, 100, 10)) +
    theme_bw() + ggtitle("Scatter plot of training data") +
    theme(legend.title = element_blank(), 
    panel.grid.major.x =element_blank(),
    panel.grid.minor.y = element_blank(),  
    panel.grid.minor.x = element_blank(), 
    panel.grid.major.y = element_blank())   
p
```

![](Coursera_ML_Assignment2_files/figure-markdown_github-ascii_identifiers/visualization-1.png)

### 1.2 Compute Cost J & Gradient

#### 1.2.1 Settings

``` r
y = data[, 3]
m = length(y)
X = cbind(rep(1, m), data[, 1:2])
X = as.matrix(X) # convert X to matrix from data.frame

# initialize parameters
initial_theta = c(rep(0, dim(X)[2]))
```

#### 1.2.2 Computing the cost J(theta)

``` r
costFunction = function(X, y) {
  function(theta) {
    J = 0
    m = length(y)
    X_theta = X %*% theta # m by 1
    h = 1 / (1 + exp(-X_theta)) # m by 1
    J = (t(y) %*% log(h) + t(1 - y) %*% (log(1 -h))) / (-m)
    J
  }
}

# Computing Cost J with initial_theta
sprintf('Cost J: %.3f', costFunction(X, y)(initial_theta))
```

    ## [1] "Cost J: 0.693"

#### 1.2.3 Computing Gradient for optimization

``` r
computeGradient = function(X, y) {
  function(theta) {
    gradient = c(rep(0, dim(X)[2]))
    m = length(y)
    X_theta = X %*% theta
    h = 1 / (1 + exp(-X_theta))
    gradient = (t(X) %*% (h - y)) / m
    gradient
  }
}

gradient = computeGradient(X, y)(initial_theta)

# Computing gradient with initial_theta
sprintf('gradient: %.3f, %.3f, %.3f: ', gradient[1], gradient[2], gradient[3])
```

    ## [1] "gradient: -0.100, -12.009, -11.263: "

### 1.3 Optimization

Learning parameters using advanced optimization algorithm In this exercise, use a built-in function (optim) to find the optimal parameters theta.

``` r
# Run optim to obtain the optimal theta
# This function will return theta and the cost
optimResult = optim(par = initial_theta, fn =  costFunction(X, y),
                 gr = computeGradient(X, y), 
                 method = "BFGS", control = list(maxit = 400))
# maxit is maximum iteration

theta = optimResult$par
cost = optimResult$value

# Printing result
sprintf('cost at theta found by optim: %.3f', cost)
```

    ## [1] "cost at theta found by optim: 0.203"

``` r
sprintf('Optimized theta: %.3f %.3f %.3f:', theta[1], theta[2], theta[3])
```

    ## [1] "Optimized theta: -25.089 0.206 0.201:"

### 1.4 Plotting decision boundary

``` r
p = p + geom_abline(slope = - theta[2]/theta[3],
                    intercept = -theta[1]/theta[3], color = "blue")
p
```

![](Coursera_ML_Assignment2_files/figure-markdown_github-ascii_identifiers/decision%20boundary-1.png)

### 1.5 Evaluation

After learning the parameters, you'll like to use it to predict the outcomes on unseen data.

``` r
# Predict probability for a student with score 45 on exam 1
# and score 85 on exam 2

predict_sample = t(c(1, 45, 85))
probability = 1 / (1 + exp(-predict_sample %*% theta))

sprintf('For a student with scores 45 and 85, we predict an admission probability of %.3f', probability)
```

    ## [1] "For a student with scores 45 and 85, we predict an admission probability of 0.776"

``` r
# Compute accuracy on our training set
# Make predict function first
Predict = function(X, theta) {
  m = dim(X)[1]
  p = rep(0, m)
  
  p[X %*% theta >= 0 ] = 1
  p
}  

# make predict vector  
predict = Predict(X, theta)

accuracy = mean(predict == y) * 100
sprintf('Train Accuracy: %.3f:', accuracy)
```

    ## [1] "Train Accuracy: 89.000:"
