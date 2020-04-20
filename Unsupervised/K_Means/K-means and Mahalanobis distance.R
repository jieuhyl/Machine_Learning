# K-means and Mahalanobis distance

# Sample data 
n <- 100
k <- 5
x <- matrix( rnorm(k*n), nr=n, nc=k )
var(x)

# Rescale the data
C <- chol( var(x) )
y <- x %*% solve(C)
var(y) # The identity matrix

kmeans(y, 4)
