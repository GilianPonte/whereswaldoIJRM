rm(list = ls())
library(dplyr)
library(ggplot2)
library("ggrepel")   

# real
collect_epsilons_real = c()
for (n in c("300","3000", "30000")){
  eps = Inf
  MAPDs = 0
  MSEs = 0
  MAEs = 0
  stats = bind_cols(eps,MAPDs,MSEs,MAEs)
  stats$n = n
  collect_epsilons_real = rbind(stats, collect_epsilons_real)
} 
collect_epsilons_real$method = "real"
colnames(collect_epsilons_real) = c("epsilon", "MAPD", "MAE", "MSE","n", "method")

summary_real = collect_epsilons_real %>% group_by(n) %>% summarize(epsilon_min = min(epsilon),
                                                                   epsilon_mean = mean(epsilon),
                                                                   epsilon_median = median(epsilon),
                                                                   epsilon_max = max(epsilon),
                                                                   min_MAPD = min(MAPD),
                                                                   max_MAPD = max(MAPD),
                                                                   mean_MAPD = mean(MAPD),
                                                                   median_MAPD = median(MAPD),
                                                                   
                                                                   min_MAE = min(MAE),
                                                                   max_MAE = max(MAE),
                                                                   mean_MAE = mean(MAE),
                                                                   median_MAE = median(MAE),
                                                                   
                                                                   min_MSE = min(MSE),
                                                                   max_MSE = max(MSE),
                                                                   mean_MSE = mean(MSE),
                                                                   median_MSE = median(MSE))
summary_real

# swap25
collect_epsilons_swap25 = c()
for (n in c("300","3000","30000")){
  eps = read.csv(paste0("epsilons_swap25_", n,".csv"), header = F)
  MAPDs = read.csv(paste0("MAPD_swap25_", n,".csv"), header = F)
  MAEs = read.csv(paste0("MAE_swap25_", n,".csv"), header = F)
  MSEs = read.csv(paste0("MSE_swap25_", n,".csv"), header = F)
  stats = bind_cols(eps,MAPDs,MAEs,MSEs)
  stats$n = n
  collect_epsilons_swap25 = rbind(stats, collect_epsilons_swap25)
} 

collect_epsilons_swap25$method = "swap25"

colnames(collect_epsilons_swap25) = c("epsilon", "MAPD", "MAE", "MSE","n", "method")

summary_swap25 = collect_epsilons_swap25 %>% group_by(n) %>% summarize(epsilon_min = min(epsilon),
                                                                       epsilon_mean = mean(epsilon),
                                                                       epsilon_median = median(epsilon),
                                                                       epsilon_max = max(epsilon),
                                                                       min_MAPD = min(MAPD),
                                                                       max_MAPD = max(MAPD),
                                                                       mean_MAPD = mean(MAPD),
                                                                       median_MAPD = median(MAPD),
                                                                       
                                                                       min_MAE = min(MAE),
                                                                       max_MAE = max(MAE),
                                                                       mean_MAE = mean(MAE),
                                                                       median_MAE = median(MAE),
                                                                       
                                                                       min_MSE = min(MSE),
                                                                       max_MSE = max(MSE),
                                                                       mean_MSE = mean(MSE),
                                                                       median_MSE = median(MSE))
summary_swap25

collect_epsilons_swap25 %>% ggplot(aes(x = n, y = epsilon)) + geom_boxplot() + theme_bw()
collect_epsilons_swap25 %>% ggplot(aes(x = n, y = MAE)) + geom_boxplot() + theme_bw()
collect_epsilons_swap25 %>% ggplot(aes(x = n, y = MSE)) + geom_boxplot() + theme_bw()


# swap50
collect_epsilons_swap50 = c()
for (n in c("300","3000","30000")){
  n
  eps = read.csv(paste0("epsilons_swap50_", n,".csv"), header = F)
  MAPDs = read.csv(paste0("MAPD_swap50_", n,".csv"), header = F)
  MAEs = read.csv(paste0("MAE_swap50_", n,".csv"), header = F)
  MSEs = read.csv(paste0("MSE_swap50_", n,".csv"), header = F)
  stats = bind_cols(eps,MAPDs,MAEs,MSEs)
  stats$n = n
  collect_epsilons_swap50 = rbind(stats, collect_epsilons_swap50)
} 

collect_epsilons_swap50$method = "swap50"

colnames(collect_epsilons_swap50) = c("epsilon", "MAPD", "MAE", "MSE","n", "method")

summary_swap50 = collect_epsilons_swap50 %>% group_by(n) %>% summarize(epsilon_min = min(epsilon),
                                                                       epsilon_mean = mean(epsilon),
                                                                       epsilon_median = median(epsilon),
                                                                       epsilon_max = max(epsilon),
                                                                       min_MAPD = min(MAPD),
                                                                       max_MAPD = max(MAPD),
                                                                       mean_MAPD = mean(MAPD),
                                                                       median_MAPD = median(MAPD),
                                                                       
                                                                       min_MAE = min(MAE),
                                                                       max_MAE = max(MAE),
                                                                       mean_MAE = mean(MAE),
                                                                       median_MAE = median(MAE),
                                                                       
                                                                       min_MSE = min(MSE),
                                                                       max_MSE = max(MSE),
                                                                       mean_MSE = mean(MSE),
                                                                       median_MSE = median(MSE))
summary_swap50

collect_epsilons_swap50 %>% ggplot(aes(x = n, y = epsilon)) + geom_boxplot() + theme_bw()
collect_epsilons_swap50 %>% ggplot(aes(x = n, y = MSE)) + geom_boxplot() + theme_bw()
collect_epsilons_swap50 %>% ggplot(aes(x = n, y = MAE)) + geom_boxplot() + theme_bw()

# copula
collect_epsilons_copula = c()
for (n in c("300","3000", "30000")){
  eps = read.csv(paste0("epsilons_copula_", n,".csv"), header = F)
  MAPDs = read.csv(paste0("MAPD_copula_", n,".csv"), header = F)
  MAEs = read.csv(paste0("MAE_copula_", n,".csv"), header = F)
  MSEs = read.csv(paste0("MSE_copula_", n,".csv"), header = F)
  stats = cbind(eps,MAPDs,MAEs,MSEs)
  stats$n = n
  collect_epsilons_copula = rbind(stats, collect_epsilons_copula)
} 

collect_epsilons_copula$method = "copula"
colnames(collect_epsilons_copula) = c("epsilon", "MAPD", "MAE", "MSE","n", "method")

summary_copula = collect_epsilons_copula %>% group_by(n) %>% summarize(epsilon_min = min(epsilon),
                                                                       epsilon_mean = mean(epsilon),
                                                                       epsilon_median = median(epsilon),
                                                                       epsilon_max = max(epsilon),
                                                                       min_MAPD = min(MAPD),
                                                                       max_MAPD = max(MAPD),
                                                                       mean_MAPD = mean(MAPD),
                                                                       median_MAPD = median(MAPD),
                                                                       
                                                                       min_MAE = min(MAE),
                                                                       max_MAE = max(MAE),
                                                                       mean_MAE = mean(MAE),
                                                                       median_MAE = median(MAE),
                                                                       
                                                                       min_MSE = min(MSE),
                                                                       max_MSE = max(MSE),
                                                                       mean_MSE = mean(MSE),
                                                                       median_MSE = median(MSE))

summary_copula

collect_epsilons_copula %>% ggplot(aes(x = n, y = epsilon)) + geom_boxplot() + theme_bw()
collect_epsilons_copula %>% ggplot(aes(x = n, y = MAE)) + geom_boxplot() + theme_bw()
collect_epsilons_copula %>% ggplot(aes(x = n, y = MAPD)) + geom_boxplot() + theme_bw()
collect_epsilons_copula %>% ggplot(aes(x = n, y = MSE)) + geom_boxplot() + theme_bw()

# anand_tenure
collect_epsilons_anand_tenure = c()
for (n in c("300", "3000", "30000")){
  eps = read.csv(paste0("epsilons_anand_", n,".csv"), header = F)
  MAPDs = read.csv(paste0("MAPD_anand_", n,".csv"), header = F)
  MAEs = read.csv(paste0("MAE_anand_", n,".csv"), header = F)
  MSEs = read.csv(paste0("MSE_anand_", n,".csv"), header = F)
  stats = bind_cols(eps,MAPDs,MAEs,MSEs)
  stats$n = n
  collect_epsilons_anand_tenure = rbind(stats, collect_epsilons_anand_tenure)
} 

collect_epsilons_anand_tenure$method = "anand_tenure"

colnames(collect_epsilons_anand_tenure) = c("epsilon", "MAPD", "MAE", "MSE","n", "method")

summary_anand_tenure = collect_epsilons_anand_tenure %>% group_by(n)  %>% summarize(epsilon_min = min(epsilon),
                                                                                   epsilon_mean = mean(epsilon),
                                                                                   epsilon_median = median(epsilon),
                                                                                   epsilon_max = max(epsilon),
                                                                                   min_MAPD = min(MAPD),
                                                                                   max_MAPD = max(MAPD),
                                                                                   mean_MAPD = mean(MAPD),
                                                                                   median_MAPD = median(MAPD),
                                                                                   
                                                                                   min_MAE = min(MAE),
                                                                                   max_MAE = max(MAE),
                                                                                   mean_MAE = mean(MAE),
                                                                                   median_MAE = median(MAE),
                                                                                   
                                                                                   min_MSE = min(MSE),
                                                                                   max_MSE = max(MSE),
                                                                                   mean_MSE = mean(MSE),
                                                                                   median_MSE = median(MSE))

summary_anand = summary_anand_tenure
summary_anand

collect_epsilons_anand_tenure %>% ggplot(aes(x = n, y = epsilon)) + geom_boxplot() + theme_bw()
collect_epsilons_anand_tenure %>% ggplot(aes(x = n, y = MAE)) + geom_boxplot() + theme_bw()
collect_epsilons_anand_tenure %>% ggplot(aes(x = n, y = MAPD)) + geom_boxplot() + theme_bw()
collect_epsilons_anand_tenure %>% ggplot(aes(x = n, y = MSE)) + geom_boxplot() + theme_bw()

# dp_005
collect_epsilons_dp_005 = data.frame(epsilon = c(), MAPD = c(), MAE = c(), MSE = c())
for (n in c("300","3000","30000")){
  print(n)
  eps = read.csv(paste0("epsilons_005_", n,".csv"), header = F)
  MAPDs = read.csv(paste0("MAPD_005_", n,".csv"), header = F)
  MAEs = read.csv(paste0("MAE_005_", n,".csv"), header = F)
  MSEs = read.csv(paste0("MSE_005_", n,".csv"), header = F)
  stats = bind_cols(eps,MAPDs,MAEs,MSEs)
  stats$n = n
  colnames(stats) = c("epsilon", "MAPD", "MAE", "MSE")
  collect_epsilons_dp_005 = rbind(stats, collect_epsilons_dp_005)
} 

collect_epsilons_dp_005$method = "dp_005"

colnames(collect_epsilons_dp_005) = c("epsilon", "MAPD", "MAE", "MSE","n", "method")

summary_dp_005 = collect_epsilons_dp_005 %>% filter(MSE < 0.2) %>% group_by(n)  %>% summarize(epsilon_min = min(epsilon),
                                                                       epsilon_mean = mean(epsilon),
                                                                       epsilon_median = median(epsilon),
                                                                       epsilon_max = max(epsilon),
                                                                       min_MAPD = min(MAPD),
                                                                       max_MAPD = max(MAPD),
                                                                       mean_MAPD = mean(MAPD),
                                                                       median_MAPD = median(MAPD),
                                                                       
                                                                       min_MAE = min(MAE),
                                                                       max_MAE = max(MAE),
                                                                       mean_MAE = mean(MAE),
                                                                       median_MAE = median(MAE),
                                                                       
                                                                       min_MSE = min(MSE),
                                                                       max_MSE = max(MSE),
                                                                       mean_MSE = mean(MSE),
                                                                       median_MSE = median(MSE))

summary_dp_005

collect_epsilons_dp_005 %>% ggplot(aes(x = n, y = epsilon)) + geom_boxplot() + theme_bw()
collect_epsilons_dp_005 %>% ggplot(aes(x = n, y = MAPD)) + geom_boxplot() + theme_bw()

summary_dp_005 %>% ggplot(aes(x = epsilon_mean, y = mean_MSE, label = round(as.numeric(n),2))) + geom_point() + geom_line() + geom_text() + theme_minimal() 


# dp_05
collect_epsilons_dp_05 = data.frame(epsilon = c(), MAPD = c(), MAE = c(), MSE = c())
for (n in c("300","3000","30000")){
  print(n)
  eps = read.csv(paste0("epsilons_05_", n,".csv"), header = F)
  MAPDs = read.csv(paste0("MAPD_05_", n,".csv"), header = F)
  MAEs = read.csv(paste0("MAE_05_", n,".csv"), header = F)
  MSEs = read.csv(paste0("MSE_05_", n,".csv"), header = F)
  stats = bind_cols(eps,MAPDs,MAEs,MSEs)
  stats$n = n
  colnames(stats) = c("epsilon", "MAPD", "MAE", "MSE")
  collect_epsilons_dp_05 = rbind(stats, collect_epsilons_dp_05)
} 

collect_epsilons_dp_05$method = "dp_05"

colnames(collect_epsilons_dp_05) = c("epsilon", "MAPD", "MAE", "MSE","n", "method")
collect_epsilons_dp_05 = na.omit(collect_epsilons_dp_05)

summary_dp_05 = collect_epsilons_dp_05   %>% group_by(n)  %>% summarize(epsilon_min = min(epsilon),
                                                                   epsilon_mean = mean(epsilon),
                                                                   epsilon_median = median(epsilon),
                                                                   epsilon_max = max(epsilon),
                                                                   min_MAPD = min(MAPD),
                                                                   max_MAPD = max(MAPD),
                                                                   mean_MAPD = mean(MAPD),
                                                                   median_MAPD = median(MAPD),
                                                                   
                                                                   min_MAE = min(MAE),
                                                                   max_MAE = max(MAE),
                                                                   mean_MAE = mean(MAE),
                                                                   median_MAE = median(MAE),
                                                                   
                                                                   min_MSE = min(MSE),
                                                                   max_MSE = max(MSE),
                                                                   mean_MSE = mean(MSE),
                                                                   median_MSE = median(MSE))

summary_dp_05

collect_epsilons_dp_05 %>% ggplot(aes(x = n, y = epsilon)) + geom_boxplot() + theme_bw()
collect_epsilons_dp_05 %>% ggplot(aes(x = n, y = MAE)) + geom_boxplot() + theme_bw()

summary_dp_05 %>% ggplot(aes(x = epsilon_mean, y = mean_MAE, label = round(as.numeric(n),2))) + geom_point() + geom_line() + geom_text() + theme_minimal() 


# dp_1
collect_epsilons_dp_1 = c()
for (n in c("300","3000","30000")){
  eps = read.csv(paste0("epsilons_1_", n,".csv"), header = F)
  MAPDs = read.csv(paste0("MAPD_1_", n,".csv"), header = F)
  MAEs = read.csv(paste0("MAE_1_", n,".csv"), header = F)
  MSEs = read.csv(paste0("MSE_1_", n,".csv"), header = F)
  stats = bind_cols(eps,MAPDs,MAEs,MSEs)
  stats$n = n
  collect_epsilons_dp_1 = rbind(stats, collect_epsilons_dp_1)
} 

collect_epsilons_dp_1$method = "dp_1"

colnames(collect_epsilons_dp_1) = c("epsilon", "MAPD", "MAE", "MSE","n", "method")

summary_dp_1 = collect_epsilons_dp_1  %>% group_by(n) %>% summarize(epsilon_min = min(epsilon),
                                                                     epsilon_mean = mean(epsilon),
                                                                     epsilon_median = median(epsilon),
                                                                     epsilon_max = max(epsilon),
                                                                     min_MAPD = min(MAPD),
                                                                     max_MAPD = max(MAPD),
                                                                     mean_MAPD = mean(MAPD),
                                                                     median_MAPD = median(MAPD),
                                                                     
                                                                     min_MAE = min(MAE),
                                                                     max_MAE = max(MAE),
                                                                     mean_MAE = mean(MAE),
                                                                     median_MAE = median(MAE),
                                                                     
                                                                     min_MSE = min(MSE),
                                                                     max_MSE = max(MSE),
                                                                     mean_MSE = mean(MSE),
                                                                     median_MSE = median(MSE))

summary_dp_1

collect_epsilons_dp_1 %>% ggplot(aes(x = n, y = epsilon)) + geom_boxplot() + theme_bw()
collect_epsilons_dp_1 %>% ggplot(aes(x = n, y = MAE)) + geom_boxplot() + theme_bw()

# dp_3
collect_epsilons_dp_3 = c()
for (n in c("300","3000","30000")){
  eps = read.csv(paste0("epsilons_3_", n,".csv"), header = F)
  MAPDs = read.csv(paste0("MAPD_3_", n,".csv"), header = F)
  MAEs = read.csv(paste0("MAE_3_", n,".csv"), header = F)
  MSEs = read.csv(paste0("MSE_3_", n,".csv"), header = F)
  stats = bind_cols(eps,MAPDs,MAEs,MSEs)
  stats$n = n
  collect_epsilons_dp_3 = rbind(stats, collect_epsilons_dp_3)
} 

collect_epsilons_dp_3$method = "dp_3"

colnames(collect_epsilons_dp_3) = c("epsilon", "MAPD", "MAE", "MSE","n", "method")

summary_dp_3 = collect_epsilons_dp_3 %>% group_by(n) %>% summarize(epsilon_min = min(epsilon),
                                                                       epsilon_mean = mean(epsilon),
                                                                       epsilon_median = median(epsilon),
                                                                       epsilon_max = max(epsilon),
                                                                       min_MAPD = min(MAPD),
                                                                       max_MAPD = max(MAPD),
                                                                       mean_MAPD = mean(MAPD),
                                                                       median_MAPD = median(MAPD),
                                                                       
                                                                       min_MAE = min(MAE),
                                                                       max_MAE = max(MAE),
                                                                       mean_MAE = mean(MAE),
                                                                       median_MAE = median(MAE),
                                                                       
                                                                       min_MSE = min(MSE),
                                                                       max_MSE = max(MSE),
                                                                       mean_MSE = mean(MSE),
                                                                       median_MSE = median(MSE))

summary_dp_3

collect_epsilons_dp_3 %>% ggplot(aes(x = n, y = epsilon)) + geom_boxplot() + theme_bw()
collect_epsilons_dp_3 %>% ggplot(aes(x = n, y = MAE)) + geom_boxplot() + theme_bw()

# dp_13
collect_epsilons_dp_13 = c()
for (n in c("300","3000", "30000")){
  n
  eps = read.csv(paste0("epsilons_13_", n,".csv"), header = F)
  MAPDs = read.csv(paste0("MAPD_13_", n,".csv"), header = F)
  MAEs = read.csv(paste0("MAE_13_", n,".csv"), header = F)
  MSEs = read.csv(paste0("MSE_13_", n,".csv"), header = F)
  stats = bind_cols(eps,MAPDs,MAEs,MSEs)
  stats$n = n
  collect_epsilons_dp_13 = rbind(stats, collect_epsilons_dp_13)
} 

collect_epsilons_dp_13$method = "dp_13"

colnames(collect_epsilons_dp_13) = c("epsilon", "MAPD", "MAE", "MSE","n", "method")
collect_epsilons_dp_13 = na.omit(collect_epsilons_dp_13)

summary_dp_13 = collect_epsilons_dp_13 %>% group_by(n)  %>% summarize(epsilon_min = min(epsilon),
                                                                     epsilon_mean = mean(epsilon),
                                                                     epsilon_median = median(epsilon),
                                                                       epsilon_max = max(epsilon),
                                                                       min_MAPD = min(MAPD),
                                                                       max_MAPD = max(MAPD),
                                                                       mean_MAPD = mean(MAPD),
                                                                       median_MAPD = median(MAPD),
                                                                       
                                                                       min_MAE = min(MAE),
                                                                       max_MAE = max(MAE),
                                                                       mean_MAE = mean(MAE),
                                                                       median_MAE = median(MAE),
                                                                       
                                                                       min_MSE = min(MSE),
                                                                       max_MSE = max(MSE),
                                                                       mean_MSE = mean(MSE),
                                                                       median_MSE = median(MSE))

summary_dp_13

# compare methods
summary_real$method = "real"
summary_swap25$method = "25% swapping"
summary_swap50$method = "50% swapping"
summary_copula$method = "Danaher & Smith (2011)"
summary_anand$method = "Anand & Lee (2022)"
summary_dp_005$method = "0.05"
summary_dp_05$method = "0.5"
summary_dp_1$method = "1"
summary_dp_3$method = "3"
summary_dp_13$method = "13"

summary_methods = rbind(summary_real, summary_swap25,summary_swap50,summary_copula,summary_anand,summary_dp_005,summary_dp_05,summary_dp_1,summary_dp_3,summary_dp_13)

summary_methods$differential_privacy = "without differential privacy"
summary_methods$differential_privacy[summary_methods$method == "0.05"] = "differential privacy"
summary_methods$differential_privacy[summary_methods$method == "0.5"] = "differential privacy"
summary_methods$differential_privacy[summary_methods$method == "1"] = "differential privacy"
summary_methods$differential_privacy[summary_methods$method == "3"] = "differential privacy"
summary_methods$differential_privacy[summary_methods$method == "13"] = "differential privacy"

# just dp
summary_methods %>% filter(differential_privacy == "differential privacy") %>% ggplot(aes(x = as.numeric(epsilon_mean), y = median_MAPD, label = method, color = differential_privacy, group = differential_privacy)) + geom_point() + geom_line() + geom_text(vjust =-1) +theme_bw() + facet_wrap(~n) +  scale_color_manual(values = c("red","black")) +
  theme(text = element_text(size=12, colour="black"), axis.text.x = element_text(size = 12,color="black")) + theme(legend.text=element_text(size=12), axis.text=element_text(size=12, color = "black"),axis.title=element_text(size=12),legend.title=element_text(size=12))+ guides(color=guide_legend(title="method"))  + theme(strip.text = element_text(size=12), legend.title=element_text(size=12)) + ylab("loss of utility (MAPD)") + xlab("empirical epsilon") 

# all methods
summary_methods$n[summary_methods$n == "300"] = "n = 300"
summary_methods$n[summary_methods$n == "3000"] = "n = 3,000"
summary_methods$n[summary_methods$n == "30000"] = "n = 30,000"
summary_methods$n = factor(summary_methods$n, levels = c("n = 300", "n = 3,000", "n = 30,000"))
summary_methods%>% ggplot(aes(x = as.numeric(epsilon_max), y = median_MAE, label = method, color = differential_privacy, group = method)) + geom_point()+ geom_text_repel(size = 4, arrow = arrow(length = unit(0.015, "npc"))) + xlim(-0.2,8) + theme_bw() + facet_wrap(~n) + scale_color_manual(values = c("red","black")) + 
  theme(text = element_text(size=12, colour="black"), axis.text.x = element_text(size = 12,color="black")) + theme(legend.text=element_text(size=12), axis.text=element_text(size=12, color = "black"),axis.title=element_text(size=12),legend.title=element_text(size=12))+ guides(color=guide_legend(title="method"))  + theme(strip.text = element_text(size=12), legend.title=element_text(size=12)) + ylab("loss of utility (MAE)") + xlab("empirical epsilon") +
  theme(legend.position="bottom")

# zooming in
summary_methods%>% filter(n == "n = 30,000") %>% filter(differential_privacy == "differential privacy") %>% ggplot(aes(x= factor(method, levels = c("0.05","0.5","1", "3", "13")),y = as.numeric(epsilon_max), fill = factor(method, levels = c("0.05","0.5","1", "3", "13")))) + geom_col()  + scale_fill_grey(start = 0.1, end = .9) + theme_bw()+ xlab('theoretical epsilon') + ylab("empirical epsilon")+ guides(fill=guide_legend(title="theoretical epsilon"))+ theme(legend.position = "none") +theme(text = element_text(size=12, colour="black"), axis.text.x = element_text(size = 12,color="black")) + theme(legend.text=element_text(size=12), axis.text=element_text(size=12, color = "black"),axis.title=element_text(size=12),legend.title=element_text(size=12))+ guides(color=guide_legend(title="method"))  + theme(strip.text = element_text(size=12), legend.title=element_text(size=12))  
summary_methods%>% filter(n == "n = 30,000") %>% filter(differential_privacy == "differential privacy") 

#  MLP --------------------------------------------------------------------
summary_methods$MLP = NA
summary_methods$MLP[summary_methods$method == 'real'] = c(42.26, 170.13, 369.628)
summary_methods$MLP[summary_methods$method == '25% swapping'] = c(26.51, 108.60, 192.65)
summary_methods$MLP[summary_methods$method == '50% swapping'] = c(20.15, 64.63, 166.569)
summary_methods$MLP[summary_methods$method == 'Danaher & Smith (2011)'] = c(3.25, 3.93, 4.9615)
summary_methods$MLP[summary_methods$method == 'Anand & Lee (2022)'] = c(1.16, 16.93, 42.8231)
summary_methods$MLP[summary_methods$method == '13'] = c(20.28,3.45 , 20.690131183613254)
summary_methods$MLP[summary_methods$method == '3'] = c(18.15,3.43 , 9.03)
summary_methods$MLP[summary_methods$method == '1'] = c(14.97,3.167 , 29.62)
summary_methods$MLP[summary_methods$method == '0.5'] = c(13.52,3.0409 , 15.08)
summary_methods$MLP[summary_methods$method == '0.05'] = c(18.01,2.555 , 26.72)


# all methods
summary_methods %>% ggplot(aes(x = as.numeric(MLP), y = median_MSE, label = method, color = differential_privacy)) + geom_point() + geom_text_repel(vjust = -1, size = 4) +theme_bw() + facet_wrap(~n, scales = "free_x")  + scale_color_manual(values = c("red","black")) + 
  theme(text = element_text(size=12, colour="black"), axis.text.x = element_text(size = 12,color="black")) + theme(legend.text=element_text(size=12), axis.text=element_text(size=12, color = "black"),axis.title=element_text(size=12),legend.title=element_text(size=12))+ guides(color=guide_legend(title="method"))  + theme(strip.text = element_text(size=12), legend.title=element_text(size=12)) + ylab("loss of utility (MSE)") + xlab("maximum loss of privacy (MLP)") +  theme(legend.position="bottom")
