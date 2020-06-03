rm(list = ls())
setwd("C:/Users/Jie.Hu/Desktop/Segmentation/0522")
#set.seed(1337)

library(dplyr)
library(tidyverse)
library(cluster)
#install.packages("factoextra")
library(factoextra)
library(Rtsne)
library(ggplot2) 


# read data
df <- read.csv('Design Home_Clustering_Vars.csv', header=T, stringsAsFactors=FALSE)
df_seg <- df[,-c(1,2)]

# check missing value
sapply(df_seg, function(x) sum(is.na(x)))


#normalize
#normalize <- function(x) {
#  return ((x - min(x)) / (max(x) - min(x)))
#}

#df_seg[,-1] <- as.data.frame(lapply(df_seg[,-1], normalize))



# k-means==============================================================================
# use elbow method to find the optimal mumber of clusters
wcss<- vector()

for (i in 1:20)
  wcss[i] <- kmeans(df_seg, i)$tot.withinss/kmeans(df_seg, i)$totss

plot(1:20, wcss, type='b', 
     main='Clusters of Clients',
     xlab='Number of Clusters',
     ylab='Percentage of Within Cluster Sum of Squares',
     pch=20, cex=2)

       
wcss<- vector()

for (i in 1:15)
  wcss[i] <- kmeans(df_seg, i)$betweenss/kmeans(df_seg, i)$totss

plot(1:15, wcss, type='b', 
     #main='Clusters of Clients',
     xlab='Number of Clusters',
     ylab='Percentage of Explained Variance',
     pch=20, cex=2)
#============================================================
# Elbow Method
set.seed(123)
fviz_nbclust(df_seg, kmeans, method = "wss")+
  labs(subtitle = "Elbow method")

# function to compute total within-cluster sum of square
wss <- function(k) {
  kmeans(df_seg, k, iter = 100, nstart = 10)$tot.withinss
}

# Compute and plot wss for k = 1 to k = 15
k.values <- 1:15

# extract wss for 2-15 clusters
wss_values <- map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")


#============================================================
# Silhouette Method
set.seed(123)
fviz_nbclust(df_seg, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")

# function to compute average silhouette for k clusters
avg_sil <- function(k) {
  km.res <- kmeans(df_seg, k, iter = 100, nstart = 10)
  ss <- silhouette(km.res$cluster, dist(df))
  mean(ss[, 3])
}

# Compute and plot wss for k = 2 to k = 15
k.values <- 2:15

# extract avg silhouette for 2-15 clusters
avg_sil_values <- map_dbl(k.values, avg_sil)

plot(k.values, avg_sil_values,
     type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of clusters K",
     ylab = "Average Silhouettes")

       
# Visualize kmeans clustering
km5 <- kmeans(df_seg, 5, nstart = 10)

fviz_cluster(km5, df_seg, ellipse.type = "norm")+
  theme_minimal()

# Visualize silhouhette information
sil <- silhouette(km5$cluster, dist(df_seg))
fviz_silhouette(sil)
       
       
#============================================================
# apply to the data
set.seed(1234)
km5 <- kmeans(df_seg, 5, iter = 100, nstart = 10)
df$Cluster5 = km5$cluster

table(df$Cluster5)



# Euclidean distance
eu_dist <- daisy(df_seg,
                 metric = "euclidean")

# t-SNE
set.seed(123)
tsne_obj <- Rtsne(eu_dist, is_distance = TRUE)


# graph
tsne_data <- as.data.frame(tsne_obj$Y)
colnames(tsne_data) <- c("X", "Y")
tsne_data <- cbind(ID = df$ï..record..Record.number, tsne_data)
tsne_data$Cluster <- factor(df$Segment.Cluster.memberships.from.kmeans.5)

ggplot(aes(x = X, y = Y), data = tsne_data) +
  geom_point(aes(color=Cluster, shape = Cluster), size = 2) +
  scale_shape_manual(values = 1:5) +
  ggtitle("Clustering Graph_5") + 
  xlab("Dimension_1") + 
  ylab("Dimension_2") +
  theme_bw() 


# PCA
prin_comp <- prcomp(df_seg, scale. = T)
# check the loadings
head(prin_comp$rotation[, 1:5])

#compute standard deviation of each principal component
std_dev <- prin_comp$sdev

#compute variance
pr_var <- std_dev^2

#check variance of first 5 components greater than 1
pr_var[1:10]

#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
prop_varex[1:20]

cumsum(prop_varex[1:7])

View(prin_comp$x)

# graph
pca_data <- as.data.frame(prin_comp$x)
pca_data <- pca_data[1:2]
colnames(pca_data) <- c("X", "Y")
pca_data <- cbind(ID = df$ï..record..Record.number, pca_data)
pca_data$Cluster <- factor(df$Segment.Cluster.memberships.from.kmeans.5)

ggplot(aes(x = X, y = Y), data = pca_data) +
  geom_point(aes(color=Cluster, shape = Cluster), size = 2) +
  scale_shape_manual(values = 1:5) +
  ggtitle("Clustering Graph_5") + 
  xlab("Dimension_1") + 
  ylab("Dimension_2") +
  theme_bw()        
