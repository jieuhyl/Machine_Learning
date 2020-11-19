rm(list = ls())

setwd("C:/Users/Jie.Hu/Desktop/IRT/Gaming/eg2")
set.seed(2020)

df <- read.csv('eg_2.csv', header=T, stringsAsFactors=FALSE)

# QREALITYINTEREST, binary
df_B1 <- df[,-1]

# check missing value ======================================================
sapply(df_B1, function(x) sum(is.na(x)))


# Exploratory Factor Analysis
fac_ana <- factanal(df_B1, factors = 2, rotation = "varimax")
fac_ana
print(fac_ana, digits=2, cutoff=.3)

#load <- fac_ana$loadings[,c(1,3)]
#plot(load, type="n") # set up plot
#text(load,labels=names(df_B1)) # add variable names


load <- data.frame(matrix(as.numeric(fac_ana$loadings), attributes(fac_ana$loadings)$dim, dimnames=attributes(fac_ana$loadings)$dimnames))

ggplot(load, aes(x=Factor1, y=Factor2)) +
  geom_point() + 
  geom_text(label=rownames(load), position = position_dodge(0.1), size=4) +
  labs(title="FA scatterplot") +
  theme_bw()