rm(list = ls())
setwd("C:/Users/Jie.Hu/Desktop/Segmentation/0622")
#set.seed(1337)

#install.packages("tidyverse")
library(tidyverse)
library(dplyr)
library(cluster)
library(factoextra)
library(Rtsne)
library(ggplot2) 


# read data
df <- read.csv('seg_0622.csv', header=T, stringsAsFactors=FALSE)
df_seg <- df[, -c(1)]

# check missing value
sapply(df_seg, function(x) sum(is.na(x)))


#normalize
#normalize <- function(x) {
#  return ((x - min(x)) / (max(x) - min(x)))
#}
#
#df_seg[,-1] <- as.data.frame(lapply(df_seg[,-1], normalize))




# Dissimilarity matrix
d <- dist(df_seg, method = "euclidean")

# Method 1
# Hierarchical clustering using Complete Linkage
# m <- c( "average", "single", "complete", "ward.D2")
hc1 <- hclust(d, method = "ward.D2" )
# Plot the obtained dendrogram
plot(hc1, cex = 0.6, hang = -1)

# Cut tree into 4 groups
sub_grp <- cutree(hc1, k = 4)

# Number of members in each cluster
table(sub_grp)

plot(hc1, cex = 0.6)
rect.hclust(hc1, k = 4, border = 2:5)

# visualize
fviz_cluster(list(data = df_seg, cluster = sub_grp))


# Using the dendrogram to find the optimal number of clusters
dendrogram = hclust(d = dist(df_seg, method = 'euclidean'), method = 'ward.D2')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distances')

# Fitting Hierarchical Clustering to the dataset
y_hc = cutree(dendrogram , 3)

# Visualising the clusters
clusplot(df_seg,
         y_hc,
         lines = 0,
         shade = FALSE,
         color = TRUE,
         labels= 1,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')


# Method 2
hc2 <- agnes(df_seg, method = "ward")
hc2$ac
# Plot the obtained dendrogram
pltree(hc2, cex = 0.6, hang = -1, main = "Dendrogram of agnes")

# evaluation ============================================================
# Elbow Method
fviz_nbclust(df_seg, FUN = hcut, method = "wss")

# Silhouette Method
fviz_nbclust(df, FUN = hcut, method = "silhouette")

# Gap Statistic Method
gap_stat <- clusGap(df_seg, FUN = kmeans, nstart = 25, K.max = 10, B = 50)
fviz_gap_stat(gap_stat)
