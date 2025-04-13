micro_data_h <- read.csv("./control_crc1.csv", header = TRUE, row.names = 1)
micro_data_d <- read.csv("./case_crc1.csv", header = TRUE, row.names = 1)

micro_data_d <- data.matrix(t(micro_data_d)) * 100
micro_data_h <- data.matrix(t(micro_data_h)) * 100

library(Matrix)
library(SpiecEasi)

Graph_nums = 500
edges_list_h <- data.frame(x1=numeric(), x2=numeric(), x3=numeric(), x4=numeric())
edges_list_d <- data.frame(x1=numeric(), x2=numeric(), x3=numeric(), x4=numeric())

for (i in 1:Graph_nums) {

  sparcc.h <- sparcc(micro_data_h[c(sample(1: nrow(micro_data_h), 30)), ])
  sparcc.h.graph <- abs(sparcc.h$Cor) >= ((sum(abs(sparcc.h$Cor))-81)/(81*80))
  diag(sparcc.h.graph) <- 0
  sparcc.h.graph <- Matrix(sparcc.h.graph, sparse = TRUE)
  elist.sparcc.h <- summary(sparcc.h.graph * sparcc.h$Cor)
  names(elist.sparcc.h) <- c('source', 'target', 'weight')
  elist.sparcc.h <- elist.sparcc.h[order(elist.sparcc.h$source, elist.sparcc.h$target), ]
  
  sparcc.d <- sparcc(micro_data_d[c(sample(1: nrow(micro_data_d), 30)), ])
  sparcc.d.graph <- abs(sparcc.d$Cor) >= ((sum(abs(sparcc.h$Cor))-81)/(81*80))
  diag(sparcc.d.graph) <- 0
  sparcc.d.graph <- Matrix(sparcc.d.graph, sparse = TRUE)
  elist.sparcc.d <- summary(sparcc.d.graph * sparcc.d$Cor)
  names(elist.sparcc.d) <- c('source', 'target', 'weight')
  elist.sparcc.d <- elist.sparcc.d[order(elist.sparcc.d$source, elist.sparcc.d$target), ]
  
  graph_h_id <- matrix(c(rep(i, times = nrow(elist.sparcc.h))), ncol = 1)
  elist.sparcc.h <- cbind(graph_h_id, elist.sparcc.h)
  temp_h <- elist.sparcc.h[, c(1,3,2,4)]
  colnames(temp_h) <- colnames(elist.sparcc.h)
  elist.sparcc.h <- rbind(elist.sparcc.h, temp_h)
  edges_list_h <- rbind(edges_list_h, elist.sparcc.h)
  
  graph_d_id <- matrix(c(rep(i, times = nrow(elist.sparcc.d))), ncol = 1)
  elist.sparcc.d <- cbind(graph_d_id, elist.sparcc.d)
  temp_d <- elist.sparcc.d[, c(1,3,2,4)]
  colnames(temp_d) <- colnames(elist.sparcc.d)
  elist.sparcc.d <- rbind(elist.sparcc.d, temp_d)
  edges_list_d <- rbind(edges_list_d, elist.sparcc.d)
  
}

edges_list_h$source <- edges_list_h$source - 1
edges_list_h$target <- edges_list_h$target - 1

edges_list_d$source <- edges_list_d$source - 1
edges_list_d$target <- edges_list_d$target - 1

write.csv(edges_list_h, './Graph_ctrl_crc1.csv', row.names = FALSE)
write.csv(edges_list_d, './Graph_case_crc1.csv', row.names = FALSE)


