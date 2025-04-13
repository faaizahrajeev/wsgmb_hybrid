library(pROC)
library(ggplot2)
library(cowplot)
### CRC1

## auc_plot
data_wsgmb <- read.csv("score_dh_crc1.csv", header = TRUE, row.names = 1)

res <- roc(data_wsgmb$label, data_wsgmb$score_dh, aur=TRUE,
               ci=TRUE,
               levels=c("non_marker", "marker"), direction="<"
)


pic1 <- ggroc(res, legacy.axes = TRUE) +
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="darkgrey", linetype=4) +
  theme_bw() +
  theme(
    panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
    axis.title.x = element_text(colour="black",size=15, vjust = -1),
    axis.title.y = element_text(colour="black",size=15, vjust = 4),
    axis.text.x = element_text(colour="black",size=15),
    axis.text.y = element_text(colour="black",size=15),
    plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
    aspect.ratio=1,
    legend.title=element_blank(), 
    legend.position = "top"
  ) +
  ggsci::scale_color_lancet() +
  annotate("text",x=0.75,y=0.125,label=paste("AUC = ", round(res$auc,3))) +
  xlab("False positive fraction") +
  ylab("True positive fraction")
