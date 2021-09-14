rm(list = ls())
setwd("D:/Kaoulis/Desktop/Lancaster/Lancaster/Modules/SCC.461 - Programming for Data Scientists/Week 999")

library("MASS")
library(ggplot2)
library(grid)
library(gridExtra)
library(dplyr)
library(xtable)

# Metrics by splitting on the last feature with the highest information gain
df1 <- read.csv("mycsvfile.csv")
# Metrics by splitting on the first feature with the highest information gain
df2 <- read.csv("mycsvfile2.csv")
# Combined metrics
df <- bind_rows(df1, df2)


df$Depth <- as.factor(df$Depth)
df$Algorithm <- as.factor(df$Algorithm)

# EDA of the metrics
box1 <- ggplot(df) +
  geom_boxplot(aes(x = Depth, y  = Accuracy, fill = Algorithm)) +
  scale_fill_discrete(guide=FALSE)
box2 <- ggplot(df) +
  geom_boxplot(aes(x = Depth, y  = Precision, fill = Algorithm))
box3 <- ggplot(df) +
  geom_boxplot(aes(x = Depth, y  = Recall, fill = Algorithm)) +
  scale_fill_discrete(guide=FALSE)
box4 <- ggplot(df) +
  geom_boxplot(aes(x = Depth, y  = Time, fill = Algorithm))
grid.arrange(box1, box2, box3, box4, nrow = 2, widths = 8:9)


# The averages grouped by depth and algorithm
mean_df <- aggregate(df, by=list(df$Depth,df$Algorithm),
                     FUN=mean, na.rm=FALSE)

mean_df <- mean_df %>% 
  dplyr::select(-c(Depth, Algorithm))

names(mean_df)[names(mean_df) == 'Group.1'] <- 'Depth'
names(mean_df)[names(mean_df) == 'Group.2'] <- 'Algorithm'


# Confidence Intervals
Sk_Acc <- (df %>% filter(Depth == 6 & Algorithm == 'Sklearn'))$Accuracy
mu <- mean(Sk_Acc)
SE <- sd(Sk_Acc) / sqrt(length(Sk_Acc))
error <- qt(0.975, length(Sk_Acc)-1)*SE
Sk_Acc_left <- mu - error
Sk_Acc_right <- mu + error

My_Acc <- (df %>% filter(Depth == 6 & Algorithm == 'Mine'))$Accuracy
mu <- mean(My_Acc)
SE <- sd(My_Acc) / sqrt(length(My_Acc))
error <- qt(0.975, length(My_Acc)-1)*SE
My_Acc_left <- mu - error
My_Acc_right <- mu + error

# t-tests for depth = 6
Sk_Pre <- (df %>% filter(Depth == 6 & Algorithm == 'Sklearn'))$Precision
Sk_Rec <- (df %>% filter(Depth == 6 & Algorithm == 'Sklearn'))$Recall
My_Pre <- (df %>% filter(Depth == 6 & Algorithm == 'Mine'))$Precision
My_Rec <- (df %>% filter(Depth == 6 & Algorithm == 'Mine'))$Recall

t.test(Sk_Acc, My_Acc, paired = T, alt="two.sided")
t.test(Sk_Pre, My_Pre, paired = T, alt="two.sided")
t.test(Sk_Rec, My_Rec, paired = T, alt="two.sided")


# Time complexity analysis
time_df <- read.csv("timecsvx20.csv")

sk_time <- time_df %>% 
  filter(Algorithm=='Sklearn')

my_time <- time_df %>% 
  filter(Algorithm=='Mine')

sklm <- lm(data = sk_time, sk_time$Time ~ sk_time$Data_proportion)
par(mfrow = c(1,2))
plot(sklm, which=1:2, cex = 0.3)

# Box cox to find complexity for sklearn.
bc = boxcox(sklm)
lambda <- bc$x[which(bc$y == max(bc$y))]

mylm <- lm(data = my_time, my_time$Time ~ my_time$Data_proportion)
par(mfrow = c(1,2))
plot(mylm, which=1:2, cex = 0.3)

# Box cox to find complexity for implemented
bc = boxcox(mylm)
lambda <- bc$x[which(bc$y == max(bc$y))]




