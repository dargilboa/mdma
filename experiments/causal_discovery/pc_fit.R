library(pcalg)
library(SID)
library(igraph)

pc_fit <- function(df) {
  n <- nrow(df)
  V <- colnames(df) # labels aka node names

  ## estimate CPDAG
  pc_fit <- pc(
    suffStat = list(C = cor(df), n = n),
    indepTest = gaussCItest, ## indep.test: partial correlations
    alpha = 0.01, labels = V, verbose = FALSE
  )

  pc_fit <- attr(pc_fit, "graph")
  graph_pc <- graph_from_graphnel(pc_fit)
  write_graph(graph_pc, "data/graph_pc.txt", format = "graphml")
  writeLines(names(V(graph_pc)), "data/graph_pc_nodes.txt")
}

# df <- read.csv("experiments/causal_discovery/sachs.csv")[, -1]
# graph_mdma <- read_graph("data/graph_mdma.txt", format = "graphml")
# graph_truth <- read_graph("data/graph_truth.txt", format = "graphml")

# par(mfrow = c(1, 3))
# plot(graph_pc, main = "Estimated CPDAG (PC)")
# plot(graph_mdma, main = "Estimated CPDAG (MDMA)")
# plot(graph_truth, main = "True DAG")


# structIntervDist(gmG8$g, gmG8$g)