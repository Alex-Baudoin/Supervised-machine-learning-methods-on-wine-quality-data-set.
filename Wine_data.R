############################Importation of the data set #################################
data.wine_1 <- read.csv ("wine_dataset.csv", sep=",",stringsAsFactors=FALSE, header=TRUE)
summary(data.wine_1) #summary
############################# Transforming the multinomial problem to a binary problem####################
library(dplyr)
data.wine_1<- data.wine_1 %>%
  mutate(Quality=case_when(data.wine_1$quality <= 5 ~ 'poor',
                         data.wine_1$quality >5 ~ 'good'))
summary(data.wine_1)
View(data.wine_1)

data.wine <- data.wine_1[,-c(12,13)] #to drop style and quality
data.wine$Quality <- factor(data.wine$Quality) #convert the target as factor
str(data.wine)
View(data.wine)
summary(data.wine)
table(data.wine$Quality)


##############################EDA of our data set#################################################
library(xtable)
a<-t(summary(data.wine))
xtable(a)

# summarize the class distribution
percentage <- prop.table(table(data.wine$Quality)) * 100
b <- cbind(freq=table(data.wine$Quality), percentage=percentage)
b
xtable(b)

#correlation matrix
library(corrplot)
data_cor = cor(data.wine[,-c(12)])
corrplot(data_cor, method="color", order="alphabet")
anyNA(data.wine)
library(caTools)
library(e1071)
attach(data.wine)
anyNA(data.wine)

library(caret)
set.seed(100)
rPartMod <- train(Quality~.,data=data.wine, method="rpart") #features importance with varImp()
rpartImp <- varImp(rPartMod)
print(rpartImp)
plot(rpartImp)

#feature importance with Boruta
library(Boruta)
set.seed(123)
boruta <-Boruta(data.wine$Quality~.,data =data.wine ,doTrace =100)
print(boruta)
plot(boruta)

###############################partition of the data set####################################
library(caret)
test_index <- createDataPartition(data.wine$Quality, p=0.70, list=FALSE)
# select 30% of the data for validation
test <- data.wine[-test_index,]
# use the remaining 70% of data to training and testing the models
training <- data.wine[test_index,]
dim(test)
dim(training)
View(test)
View(training)
table(test$Quality)

## summarize attribute distributions
summary(training)

####################### split inputs and output of the training################################
x <- training[,1:11]
y <- training[,12] 

################################ Data Visualization###############################################
# Barplot for class breakdown
plot(y,xlab= "Quality Score",ylab = "frequency",main = "Barplot of Quality in the training set")
xtable(table(training$Quality))
percentage <- prop.table(table(test$Quality)) * 100
c <- cbind(freq=table(test$Quality), percentage=percentage)
c
xtable(c)

#Distribution of each attributes
featurePlot(x=x, y=y, plot="box")

# Density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

############################Performing 5-fold cross validation###############################

control <- trainControl(method="cv", number=5)
metric <- "Accuracy"
attach(training)
attach(test)
######################train the SVM model############################################
set.seed(7)
modelSvm <- train(Quality~., data=training, method="svmRadial", trControl=control)
modelSvm
predSvm=predict(modelSvm,newdata = test)
predSvm
conf_matrix <- confusionMatrix(predSvm,as.factor(test$Quality))
conf_matrix
library(ConfusionTableR)
mc_df <- ConfusionTableR::binary_class_cm(predSvm,as.factor(test$Quality),
                                         mode="everything")
mc_df$confusion_matrix
xtable(mc_df$confusion_matrix$table)
xtable(data.frame(mc_df$confusion_matrix$byClass),digits=4)
data.frame()


library(dplyr)
library(mlbench)
library(tidyr)
library(e1071)
library(randomForest)
glimpse(mc_df$record_level_cm)
mc_df$confusion_matrix
xtable(mc_df$confusion_matrix$byClass)
xtable(mc_df$confusion_matrix$table)

###########################################train the RF model###############################
set.seed(7)
#to get the best m corresponding to a prediction error of 0%
rfmodel1 <- train(Quality~., data=training, method="rf", trControl=control)
rfmodel1
predrf <- predict(rfmodel1, newdata = test)
predrf
conf_matrix_rf <- confusionMatrix(predrf,as.factor(test$Quality))
mc_df_rf <- ConfusionTableR::binary_class_cm(predrf,as.factor(test$Quality),
                                         mode="everything")
mc_df_rf$confusion_matrix
xtable(mc_df_rf$confusion_matrix$table)
xtable(data.frame(mc_df_rf$confusion_matrix$byClass),digits = 4)

#############################################train the KNN model##################################3
set.seed(7)
fit.knn <- train(Quality~.,
                 data = training , 
                 method = "knn",
                 trControl = control)
fit.knn
predknn=predict(fit.knn,newdata = test)
predknn
conf_matrix_knn <- confusionMatrix(predknn,as.factor(test$Quality))
mc_df_knn <- ConfusionTableR::binary_class_cm(predknn,as.factor(test$Quality),
                                            mode="everything")
mc_df_knn$confusion_matrix
xtable(mc_df_knn$confusion_matrix$table)
xtable(data.frame(mc_df_knn$confusion_matrix$byClass),digits = 4)

######################################train the NB model################################################
modelnb <-train(Quality~., data=training, method="naive_bayes", trControl=control)
modelnb
prednb=predict(modelnb,newdata = test)
prednb
conf_matrix_nb <- confusionMatrix(prednb,as.factor(test$Quality))
mc_df_nb <- ConfusionTableR::binary_class_cm(prednb,as.factor(test$Quality),
                                             mode="everything")
mc_df_nb$confusion_matrix
xtable(mc_df_nb$confusion_matrix$table)
xtable(data.frame(mc_df_nb$confusion_matrix$byClass),digits = 4)

##################################### Train the logistic regression ##################################
modellr <-train(Quality~., data=training, method="glm", trControl=control)
modellr
predlr=predict(modellr,newdata = test)
predlr
conf_matrix_lr <- confusionMatrix(predlr,as.factor(test$Quality))
mc_df_lr <- ConfusionTableR::binary_class_cm(predlr,as.factor(test$Quality),
                                             mode="everything")
mc_df_lr$confusion_matrix
xtable(mc_df_lr$confusion_matrix$table)
xtable(data.frame(mc_df_lr$confusion_matrix$byClass),digits = 4)



#################################### collect all resamples and compare##################################
results <- resamples(list(SVM=modelSvm,KNN=fit.knn, RF=rfmodel1, NB=modelnb, LR=modellr))
# summarize the distributions of the results 
summary(results)
################################## boxplots of performance metrics############################################
bwplot(results)
################################## dot plots of performance metrics ##################################
dotplot(results)
