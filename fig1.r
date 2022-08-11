

rm(list=ls())
library(openxlsx)
library(ggplot2)
library(dplyr)
library(factoextra)
library(blockcluster)
library(gridExtra)
library(RColorBrewer)
coul <- colorRampPalette(brewer.pal(9, "YlOrRd"))(50)


# iid
path = "./iid/mnist/client_emb.xlsx"
dat <- read.xlsx(xlsxFile = path, sheet = 1, skipEmptyRows = FALSE)
dat = dat[,-1]
dat = as.data.frame(t(as.matrix(dat)))
res.km = eclust(x=dat, FUNcluster = "kmeans", k= 2)
dd <- cbind(dat, cluster = res.km$cluster)
datt = rbind(dd[which(dd$cluster == 1),], dd[which(dd$cluster == 2),])
datt = datt[,-101]
d = as.vector(as.matrix(datt))
df = data.frame(clients = rep(c(1:100), each = 200), neurons = c(1:200), emb = d)


p1 = ggplot(df, aes(x = clients, y = neurons, fill = emb))+
  geom_tile(width = 1, height = 1)  +
  scale_fill_gradientn(colours = coul) +
  guides(fill=guide_legend(title="Value")) +
  theme(
    panel.background = element_blank(),
    panel.grid.major = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 20)
  )
  



# non-iid
path = "./non-iid/mnist/client_emb.xlsx"
dat <- read.xlsx(xlsxFile = path, sheet = 1, skipEmptyRows = FALSE)
dat = dat[,-1]
dat = as.data.frame(t(as.matrix(dat)))
res.km = eclust(x=dat, FUNcluster = "kmeans", k= 2)
dd <- cbind(dat, cluster = res.km$cluster)
datt = rbind(dd[which(dd$cluster == 1),], dd[which(dd$cluster == 2),])
datt = datt[,-101]
d = as.vector(as.matrix(datt))
df = data.frame(clients = rep(c(1:100), each = 200), neurons = c(1:200), emb = d)


p2 = ggplot(df, aes(x = clients, y = neurons, fill = emb))+
  geom_tile(width = 1, height = 1)  +
  scale_fill_gradientn(colours = coul) +
  guides(fill=guide_legend(title="Value")) +
  theme(
    panel.background = element_blank(),
    panel.grid.major = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 20)
  )



SaveDR <- c("./plot/")
png(filename = paste0(SaveDR,"emb_neurons21",".png"), width = 1200, height = 700)

grid.arrange(p1, p2,  ncol = 2)
dev.off()

