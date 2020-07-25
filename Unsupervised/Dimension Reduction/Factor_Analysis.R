# factor analysis
rm(list = ls())

setwd("C:/Users/Jie.Hu/Desktop/Segmentation/0701")
set.seed(1337)

#install.packages("polycor")
library(polycor)


df <- read.csv('fa_0701_v2.csv', header=T, stringsAsFactors=FALSE)

# QFEATS
df_B1 <- df[,2:30]

# check missing value ======================================================
sapply(df_B1, function(x) sum(is.na(x)))


# numerical
fac_ana <- factanal(df_B1, factors = 5, rotation = "varimax")
fac_ana


# top 1 box as numeric
df_B2 <- as.data.frame(ifelse(df_B1 == 1, 1, 0))

fac_ana <- factanal(df_B2, factors = 5, rotation = "varimax")
fac_ana


# top 1 box as factor
df_B3 <- as.data.frame(ifelse(df_B1 == 1, 1, 0))
df_B3 <- sapply(df_B3, as.factor)

het.mat <- hetcor(df_B3)$cor

fac_ana <- factanal(covmat = het.mat, factors = 5, rotation = "varimax")
fac_ana


# categorical
df_B4 <- sapply(df_B1, as.factor)

het.mat <- hetcor(df_B4)$cor

fac_ana <- factanal(covmat = het.mat, factors = 5, rotation = "varimax")
fac_ana

