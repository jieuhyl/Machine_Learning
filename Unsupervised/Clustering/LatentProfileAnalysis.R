# A different name for latent profile analysis is "gaussian (finite) mixture model"
# Its Bayesian version is popular in the computer science literature as "latent Dirichlet
# Confusingly, sometimes latent class analysis is used as a broader term for mixture models.


rm(list = ls())
setwd("C:/Users/Jie.Hu/Desktop/Segmentation/0706")
#set.seed(1337)


#install.packages("tidyLPA")
library(dplyr)
library(tidyLPA)
library(cluster)
library(Rtsne)
library(ggplot2) 


# read data
df <- read.csv('seg_0701_v2.csv', header=T, stringsAsFactors=FALSE)
df <- subset(df, QCONCEPT_INTEREST %in% c(1,2,3))
df_seg <- df[, -c(1,12)]

# check missing value
sapply(df_seg, function(x) sum(is.na(x)))


#normalize
#normalize <- function(x) {
#  return ((x - min(x)) / (max(x) - min(x)))
#}
#
#df_seg[,-1] <- as.data.frame(lapply(df_seg[,-1], normalize))


# LPA
lpa <- df_seg %>%
  #select(broad_interest, enjoyment, self_efficacy) %>%
  single_imputation() %>%
  scale %>%
  estimate_profiles(1:3, 
                    variances = c("equal", "varying"),
                    covariances = c("zero", "varying")) %>%
  compare_solutions(statistics = c("AIC", "BIC"))
lpa


lpa <- df_seg %>%
  #select(broad_interest, enjoyment, self_efficacy) %>%
  single_imputation() %>%
  scale %>%
  estimate_profiles(3, variances = 'equal', covariances = 'zero')

lpa
plot_profiles(lpa)

# get the outputs
View(get_data(lpa))
df_seg_output <- get_data(lpa)
get_fit(lpa)
get_estimates(lpa)



# Euclidean distance
set.seed(123)
eu_dist <- daisy(df_seg_output[, 3:12],
                 metric = "euclidean")

# t-SNE
tsne_obj <- Rtsne(eu_dist, is_distance = TRUE)


# graph
tsne_data <- as.data.frame(tsne_obj$Y)
colnames(tsne_data) <- c("X", "Y")
tsne_data <- cbind(ID = df$Internal.Interview.Numbers, tsne_data)
tsne_data$Cluster <- factor(df_seg_output$Class)

ggplot(aes(x = X, y = Y), data = tsne_data) +
  geom_point(aes(color=Cluster, shape = Cluster), size = 2) +
  scale_shape_manual(values = 1:11) +
  ggtitle("Clustering Graph_11") + 
  xlab("Dimension_1") + 
  ylab("Dimension_2") +
  theme_bw() 
