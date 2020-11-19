rm(list = ls())

setwd("C:/Users/Jie.Hu/Desktop/IRT/Gaming/eg2")
set.seed(2020)

#install.packages("factoextra")
library(factoextra)


df <- read.csv('eg_2.csv', header=T, stringsAsFactors=FALSE)

# QREALITYINTEREST, binary
df_B1 <- df[1:1000,-c(1:4)]

# check missing value ======================================================
sapply(df_B1, function(x) sum(is.na(x)))


res.pca <- prcomp(df_B1, scale = TRUE)
fviz_eig(res.pca)



# Graph of variables. Positive correlated variables point to the same side of the plot. 
# Negative correlated variables point to opposite sides of the graph.
fviz_pca_var(res.pca,
             axes = c(1, 3),
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)


# Graph of individuals. Individuals with a similar profile are grouped together.
fviz_pca_ind(res.pca,
             axes = c(1, 2),
             col.ind = "cos2", # Color by the quality of representation
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)


# Biplot of individuals and variables
fviz_pca_biplot(res.pca, 
                axes = c(1, 2),
                repel = TRUE,
                col.var = "#2E9FDF", # Variables color
                col.ind = "#696969"  # Individuals color
)


# Eigenvalues
eig.val <- get_eigenvalue(res.pca)
eig.val

# Results for Variables
res.var <- get_pca_var(res.pca)
res.var$coord          # Coordinates
res.var$contrib        # Contributions to the PCs
res.var$cos2           # Quality of representation 

# Results for individuals
res.ind <- get_pca_ind(res.pca)
res.ind$coord          # Coordinates
res.ind$contrib        # Contributions to the PCs
res.ind$cos2           # Quality of representation rm(list = ls())
