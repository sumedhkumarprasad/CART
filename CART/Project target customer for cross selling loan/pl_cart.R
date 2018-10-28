# Set the Working Directory
setwd("D:/Great Lakes PGPDSE/Great Lakes/10 Supervised Learning - Classification/Supervised Learning Classification/Mini_Project")
# Import Dataset
pl=read.csv("PL_XSELL.csv")
sum(is.na(pl)) # There is no NA value in the dataset

# Converting Categorical value into the factor data type
pl$GENDER=as.factor(pl$GENDER) # Converting into factor varaiable
pl$OCCUPATION=as.factor(pl$OCCUPATION)
pl$AGE_BKT=as.factor(pl$AGE_BKT)
pl$ACC_TYPE=as.factor(pl$ACC_TYPE)
pl=pl[ , -c(1,3,11,13,14,16,17,18,19,20,22,23,24,25,26,29,32,34,35,36,37,38,40)]
dim(pl)
View(pl)

summary(pl) # There is no missing value 

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')

library(caTools)
set.seed(123)
split = sample.split(pl$TARGET, SplitRatio = 0.7)# Divided the data into 70:30 ratio
dev_sample = subset(pl, split == TRUE)
hold_sample = subset(pl, split == FALSE)

nrow(dev_sample) # 14000 rows
nrow(hold_sample) # 6000 rows

# Feature Scaling
#dev_sample[,c(3,6,7,9)] = scale(dev_sample[,c(3,6,7,9)])
#hold_sample[,c(3,6,7,9)] = scale(hold_sample[,c(3,6,7,9)])


## installing rpart package for CART
## install.packages("rpart")
## install.packages("rpart.plot")

## loading the library
library(rpart)
library(rpart.plot)
# For Development sample finding the target rate
table(dev_sample$TARGET) # 0's 12242 1's 1758

##  Devlopment sample Target Rate 
sum(dev_sample$TARGET)/14000 # 12.55% Target rate on development sample

# For Hold sample finding the target rate
table(hold_sample$TARGET) # 0's 5246  1's 754

##  Hold sample Target Rate 
sum(hold_sample$TARGET)/6000 #  12.56% Target rate on hold sample

## setting the control paramter inputs for rpart
r.ctrl = rpart.control(minsplit=100, minbucket = 10, cp = 0, xval = 10)

# Fitting Decision Tree Classification to the Development Dataset
# install.packages('rpart')

m1 <- rpart(formula = dev_sample$TARGET ~ ., 
            data = dev_sample[,-1], method = "class", 
            control = r.ctrl)
m1

## install.packages("rattle")
## install.packages("RcolorBrewer")
library(rattle)

library(RColorBrewer)
fancyRpartPlot(m1)
rpart.plot(m1)

printcp(m1)
plotcp(m1)

# Pruning criteria based on cp table 

ptree<-prune(m1,cp=0.002,"CP")
printcp(ptree)

fancyRpartPlot(ptree,uniform= TRUE, main="Pruned Classification Tree")

## Let's use rattle to see various model evaluation measures
##rattle()

#View(dev_sample)
head(dev_sample,2)

## deciling code
decile <- function(x){
  deciles <- vector(length=10)
  for (i in seq(0.1,1,.1)){
    deciles[i*10] <- quantile(x, i, na.rm=T)
  }
  return (
    ifelse(x<deciles[1], 1,
           ifelse(x<deciles[2], 2,
                  ifelse(x<deciles[3], 3,
                         ifelse(x<deciles[4], 4,
                                ifelse(x<deciles[5], 5,
                                       ifelse(x<deciles[6], 6,
                                              ifelse(x<deciles[7], 7,
                                                     ifelse(x<deciles[8], 8,
                                                            ifelse(x<deciles[9], 9, 10
                                                            ))))))))))
}

dev_sample$predict.class <- predict(ptree, dev_sample, type="class")
dev_sample$predict.score <- predict(ptree, dev_sample)
class(dev_sample$predict.score)

## deciling
dev_sample$deciles <- decile(dev_sample$predict.score[,2])
View(dev_sample)

## Ranking code
##install.packages("data.table")
library(data.table)
tmp_DT = data.table(dev_sample)

rank <- tmp_DT[, list(
  cnt = length(TARGET), 
  cnt_resp = sum(TARGET), 
  cnt_non_resp = sum(TARGET == 0)) , 
  by=deciles][order(-deciles)]

rank$rrate <- round(rank$cnt_resp * 100 / rank$cnt,2);
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_perct_resp <- round(rank$cum_resp * 100 / sum(rank$cnt_resp),2);
rank$cum_perct_non_resp <- round(rank$cum_non_resp * 100 / sum(rank$cnt_non_resp),2);
rank$ks <- abs(rank$cum_perct_resp - rank$cum_perct_non_resp);

View(rank)

##install.packages("ROCR")
## AUC for Development dataset
library(ROCR)
pred <- prediction(dev_sample$predict.score[,2], dev_sample$TARGET)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)

##install.packages("ineq")
library(ineq)
gini = ineq(dev_sample$predict.score[,2], type="Gini")

gini2=2*auc-1
with(dev_sample, table(TARGET, predict.class))
auc
KS
gini

View(rank)
## Syntax to get the node path
tree.path <- path.rpart(ptree, node = c(16, 18))

nrow(hold_sample)

## Scoring Holdout sample
hold_sample$predict.class <- predict(m1, hold_sample, type="class")
hold_sample$predict.score <- predict(m1, hold_sample)


hold_sample$deciles <- decile(hold_sample$predict.score[,2])
#View(hold_sample)

## Ranking code
##install.packages("data.table")
## Based upon the holding dataset.
library(data.table)
tmp_DT = data.table(hold_sample)
h_rank <- tmp_DT[, list(
  cnt = length(TARGET), 
  cnt_resp = sum(TARGET), 
  cnt_non_resp = sum(TARGET == 0)) , 
  by=deciles][order(-deciles)]
#
h_rank$rrate <- round(h_rank$cnt_resp * 100 / h_rank$cnt,2);
h_rank$cum_resp <- cumsum(h_rank$cnt_resp)
h_rank$cum_non_resp <- cumsum(h_rank$cnt_non_resp)
h_rank$cum_perct_resp <- round(h_rank$cum_resp * 100 / sum(h_rank$cnt_resp),2);
h_rank$cum_perct_non_resp <- round(h_rank$cum_non_resp * 100 / sum(h_rank$cnt_non_resp),2);
h_rank$ks <- abs(h_rank$cum_perct_resp - h_rank$cum_perct_non_resp);

View(h_rank)

## AUC for Development dataset
library(ROCR)
pred <- prediction(hold_sample$predict.score[,2], hold_sample$TARGET)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)
gini3=2*auc-1

auc
KS
gini
with(hold_sample, table(TARGET, predict.class))

