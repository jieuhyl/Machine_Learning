rm(list = ls())
setwd("C:/Users/Jie.Hu/Desktop/Segmentation/0331")
#set.seed(1337)

library(dplyr)
library(cluster)
library(Rtsne)
library(ggplot2) 

# read data
df <- read.csv('us04388online-mall-final.csv', header=T, stringsAsFactors=FALSE)

# watch 1
df <- df[,c(1,587:604)]
df = na.omit(df)

# watch 2
df <- df[,c(1,745:762)]
df = na.omit(df)

# watch 3
df <- df[,c(1,905:922)]
df = na.omit(df)

# watch 4
df <- df[,c(1,1065:1082)]
df = na.omit(df)

# check missing value
sapply(df, function(x) sum(is.na(x)))

# add weights
df$Weight <- 1

# Gower distance
gower_dist <- daisy(df[, -c(2:19)], metric = "gower")

summary(gower_dist)

gower_mat <- as.matrix(gower_dist)


# Calculate silhouette width for many k using PAM
sil_width <- c(NA)

for(i in 2:10){
  
  pam_fit <- pam(gower_dist,
                 diss = TRUE,
                 k = i)
  
  sil_width[i] <- pam_fit$silinfo$avg.width
  
}

# Plot sihouette width (higher is better)
plot(1:10, sil_width,
     xlab = "Number of clusters",
     ylab = "Silhouette Width")
lines(1:10, sil_width)


# Partitioning Around Medoids (PAM)
pam_fit3 <- pam(gower_dist, diss = TRUE, k = 3)
pam_fit4 <- pam(gower_dist, diss = TRUE, k = 4)
pam_fit5 <- pam(gower_dist, diss = TRUE, k = 5)
#pam_fit6 <- pam(gower_dist, diss = TRUE, k = 6)

df$Cluster3 = pam_fit3$clustering
df$Cluster4 = pam_fit4$clustering
df$Cluster5 = pam_fit5$clustering
#df$Cluster6 = pam_fit6$clustering

table(df$Cluster3)
table(df$Cluster4)
table(df$Cluster5)
#table(df$Cluster6)

#df_seg$Weight <- 1
sapply(split(df, df$Cluster3), function(x) apply(x[, 2:19], 2, weighted.mean, x$Weight, na.rm =T))
sapply(split(df, df$Cluster4), function(x) apply(x[, 2:19], 2, weighted.mean, x$Weight, na.rm =T))
sapply(split(df, df$Cluster5), function(x) apply(x[, 2:19], 2, weighted.mean, x$Weight, na.rm =T))
#sapply(split(df, df$Cluster6), function(x) apply(x[, 2:19], 2, weighted.mean, x$Weight, na.rm =T))


write.csv(df, file="seg_output_watch_4.csv", row.names = FALSE)


# Euclidean distance
#eu_dist <- daisy(df[, 2:19],
#                 metric = "euclidean")

# t-SNE
set.seed(123)
tsne_obj <- Rtsne(gower_dist, is_distance = TRUE)

# graph
tsne_data <- as.data.frame(tsne_obj$Y)
colnames(tsne_data) <- c("X", "Y")
tsne_data <- cbind(ID = df$record, tsne_data)
tsne_data$Cluster <- factor(df$Cluster6)

ggplot(aes(x = X, y = Y), data = tsne_data) +
  geom_point(aes(color=Cluster, shape = Cluster), size = 2) +
  ggtitle("Clustering Graph") + 
  xlab("Dimension_1") + 
  ylab("Dimension_2") +
  theme_bw() 

#write.csv(tsne_data, file="seg_xy_cluster3.csv", row.names = FALSE)




