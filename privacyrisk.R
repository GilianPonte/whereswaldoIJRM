rm(list = ls())

library(ggplot2)
library(dplyr)

privacy_risk = function(l,N,epsilon, iterations){
  (l/n)*epsilon*sqrt(iterations)
}

dd <- expand.grid(l = c(1), N=seq(from = 10, to = 100000, b = 10), epsilon=c(1), it = c(100000))

dd$result = (dd$l/dd$N)*dd$epsilon*sqrt(dd$it)
summary(dd)
dd %>% ggplot(aes(N, result, color = as.factor(it))) + geom_line() + theme_bw() + ylab("privacy risk (epsilon)") + xlab("sample size (n)") + scale_x_log10() + scale_color_grey() + 
  guides(color=guide_legend(title="iterations (T)"))  +
  annotate("point", x = 100, y = (1/100)*1*sqrt(100000)) +
  annotate("text", x = 100, y = (1/100)*1*sqrt(100000), label = "(100, 3.16)", vjust = -1, size = 4.5) +
  annotate("point", x = 1000, y = (1/1000)*1*sqrt(100000)) +
  annotate("text", x = 1000, y = (1/1000)*1*sqrt(100000), label = "(1,000, 0.32)", vjust = -1, size = 4.5) +
  annotate("point", x = 100000, y = (1/100000)*1*sqrt(100000)) +
  annotate("text", x = 100000, y = (1/100000)*1*sqrt(100000), label = "(100,000, .003)", vjust = -1, hjust = 1, size = 4.5) +
  theme(text = element_text(size=14, colour="black"), legend.text=element_text(size=12), axis.text.x = element_text(size = 12,color="black")) + theme(axis.text=element_text(size=12, color = "black"),axis.title=element_text(size=13)) + theme(legend.position = "none")
