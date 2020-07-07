# A different name for latent class analysis is "binomial (finite) mixture model".
# Its Bayesian version is popular in the computer science literature as "latent Dirichlet
# Confusingly, sometimes latent class analysis is used as a broader term for mixture models.


rm(list = ls())
setwd("C:/Users/Jie.Hu/Desktop/Segmentation/0706")
#set.seed(1337)


install.packages("poLCA")
library(dplyr)
library(poLCA)
library(cluster)
library(Rtsne)
library(ggplot2) 


# read data
df <- read.csv('seg_0701.csv', header=T, stringsAsFactors=FALSE)
df <- subset(df, QCONCEPT_INTEREST %in% c(1,2,3))
df_seg <- df[, -c(1,12)] + 1

# check missing value
sapply(df_seg, function(x) sum(is.na(x)))


#

f <- cbind(Crowd.Choice, Crowd.Play, Curve.shots)~1
lca <- poLCA(f, 
            data = df_seg, 
            nclass= 2,
            graphs = TRUE,
            na.rm = TRUE)

df_seg$Cluster2 <- lca$predclass
table(df_seg$Cluster2)
