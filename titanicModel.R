setwd("/Users/derek/Desktop/kaggle/titanic/")

library(dplyr)
library(glmnet)
library(MASS)
library(tree)
library(Amelia) ## useful for visualizing missing data

## Import raw data

raw_data = read.csv(file="train.csv",header=T,na.string=c(""))

## Exploratory analysis: 

## str(raw_data)
##'data.frame':	891 obs. of  12 variables:
##$ PassengerId: int  1 2 3 4 5 6 7 8 9 10 ...
##$ Survived   : int  0 1 1 1 0 0 0 0 1 1 ...
##$ Pclass     : int  3 1 3 1 3 3 1 3 3 2 ...
##$ Name       : Factor w/ 891 levels "Abbing, Mr. Anthony",..: 109 191 358 277 16 559 520 629 417 581 ...
##$ Sex        : Factor w/ 2 levels "female","male": 2 1 1 1 2 2 2 2 1 1 ...
##$ Age        : num  22 38 26 35 35 NA 54 2 27 14 ...
##$ SibSp      : int  1 1 0 1 0 0 0 3 0 1 ...
##$ Parch      : int  0 0 0 0 0 0 0 1 2 0 ...
##$ Ticket     : Factor w/ 681 levels "110152","110413",..: 524 597 670 50 473 276 86 396 345 133 ...
##$ Fare       : num  7.25 71.28 7.92 53.1 8.05 ...
##$ Cabin      : Factor w/ 147 levels "A10","A14","A16",..: NA 82 NA 56 NA NA 130 NA NA NA ...
##$ Embarked   : Factor w/ 3 levels "C","Q","S": 3 1 3 3 3 2 3 3 3 1 ...

sapply(raw_data,function(x) sum(is.na(x)))
##  PassengerId    Survived      Pclass        Name         Sex         Age       SibSp       Parch      Ticket        Fare       Cabin 
##            0           0           0           0           0         177           0           0           0           0         687 
##     Embarked 
##            2 

missmap(raw_data) 
## "Cabin" has too many missing values... get rid of this variable

## "Name" and "Ticket" are basically junk variables
## perhaps length of name can be used

## Cleaning data
## Taking care of missing values: 
## don't get rid of them!

cleanData <- function(raw_data) {
  clean_data = raw_data %>% dplyr::select(c(PassengerId,Pclass,Sex,Age,SibSp,Parch))
  clean_data$Age[is.na(clean_data$Age)] = median(raw_data$Age,na.rm=T)
  clean_data
}

clean_data = raw_data %>% dplyr::select(-c(Name,Ticket,Cabin))
## clean_data = raw_data[,c(-4,-9,-11)]

## Replace with the mean, median, or mode

clean_data$Age[is.na(clean_data$Age)] = median(raw_data$Age,na.rm=T)
## clean_data$Age[is.na(clean_data$Age)] = median(raw_data$Age,na.rm=T)

## glm.fit = glm(Survived~.,family="binomial",data=clean_data)
## Pclass,Sex,Age,SibSp seem relevant

## includes predictors PClass, Sex, Age, SibSp, Parch
clean_data_v2 = clean_data[,c(2:4,5:7)]

## includes predictors PClass, Sex, Age, SibSp
clean_data_v3 = clean_data[,c(2:4,5:6)]
## clean_data_v3 = clean_data %>% dplyr::select(c(PassengerId,Survived,Pclass,Sex,Age,SibSp,Parch))

## Find and remove outliers from training data 
outliers <- boxplot(clean_data_v2$Age, plot=FALSE)$out
clean_data_v5 <- clean_data_v2[-which(clean_data_v2$Age %in% outliers),]

clean_data_v6 <- clean_data_v3[-which(clean_data_v3$Age %in% outliers),]

## The following functions takes in a dataframe as an argument
## and runs glm, lda, qda, and tree analysis on the dataframe
## and returns the model with the highest predictive power
basicModels <- function(myData,name="Dataframe") {

  ## we will fill the bestModel with information from the model
  ## ... that gives the best predictions from the model [[1]]
  ## ... along with the corresponding test data [[2]]
  ## ... and the corresponding model itself [[3]]
  ## ... and the best_th value if applicable [[4]]
  
  successRate = 0
  
  bestModel = vector("list",length=4)
  names(bestModel) = c("testpreds","testdata","model","best_th")
  
  set.seed(2)
  
  train = sample(1:nrow(myData),nrow(myData)/2)
  test = -(train)
  test.preds = myData$Survived[test]
  
  glm.fit = glm(Survived~.,family="binomial",data=myData,subset=train)
  glm.probs = predict(glm.fit,newdata=myData[test,],type="response")
  glm.preds = rep(0,nrow(myData[test,]))
  
  best_th = glm_tune(myData)
  cat("The best_th is", best_th,"\n")
  
  glm.preds[glm.probs > best_th] = 1
  
  lda.fit = lda(Survived~.,data=myData,subset=train)
  lda.preds = predict(lda.fit,newdata=myData[test,])
  
  cat("Analysis of",name,"\n")
  
  cat("\nLogistic Regression:\n")
  print(table(glm.preds,test.preds))
  cat("Success rate:",mean(glm.preds==test.preds),"\n")
  
  cat("\nLDA:\n")
  print(table(lda.preds$class,test.preds))
  cat("Success rate:",mean(lda.preds$class==test.preds),"\n")
  
  qda.fit = qda(Survived~.,data=myData,subset=train)
  qda.preds = predict(qda.fit,newdata=myData[test,])
  
  cat("\nQDA:\n")
  print(table(qda.preds$class,test.preds))
  cat("Success rate:",mean(qda.preds$class==test.preds),"\n")
  
  tree.fit = tree(as.factor(Survived)~.,data=myData,subset=train)
  plot(tree.fit)
  text(tree.fit,pretty=0)
  tree.preds = predict(tree.fit,newdata=myData[test,],type="class")
  
  ## this will convert the factor into numeric type
  ## tree.preds = as.numeric(as.character(tree.preds)) 
  
  cat("\nClassification Tree:\n")
  print(table(tree.preds,test.preds))
  cat("Success rate:",mean(tree.preds==test.preds),"\n")
  
  if(mean(glm.preds==test.preds) > successRate) {
    successRate = mean(glm.preds==test.preds)
    bestModel$testpreds = as.numeric(as.character(glm.preds))
    bestModel$model = glm.fit
  }
  if(mean(lda.preds$class==test.preds) > successRate) {
    successRate = mean(lda.preds$class==test.preds)
    bestModel$testpreds = as.numeric(as.character(lda.preds$class))
    bestModel$model = lda.fit
  }
  if(mean(qda.preds$class==test.preds) > successRate) {
    successRate = mean(qda.preds$class==test.preds)
    bestModel$testpreds = as.numeric(as.character(qda.preds$class))
    print("qda model is best")
    bestModel$model = qda.fit
  }
  if(mean(tree.preds==test.preds) > successRate) {
    successRate = mean(tree.preds==test.preds)
    bestModel$testpreds = as.numeric(as.character(tree.preds))
    bestModel$model = tree.fit
  }
  
  bestModel$testdata = test.preds
  
  ## this returns the best model 
  bestModel
}

glm_tune <- function(myData) {
  
  set.seed(3)
  ## glm.fit = glm(Survived~.,family="binomial",data=clean_data)
  ## Pclass,Sex,Age,SibSp seem relevant
  train = sample(1:nrow(myData),nrow(myData)/2)
  test = -(train)
  test.preds = myData$Survived[test]
  
  th = seq(from=0.5, to=1.0, length=100)
  successRates = data.frame(Threshold = th,SuccessRate = rep(0,length(th)))
  for(i in 1:length(th)) {
    glm.fit = glm(Survived~.,family="binomial",data=myData,subset=train)
    glm.probs = predict(glm.fit,newdata=myData[test,],type="response")
    glm.preds = rep(0,nrow(myData[test,]))
    
    glm.preds[glm.probs > th[i]] = 1
    
    successRates[i,2] = mean(glm.preds==test.preds)
  }
  ## print(successRates)
  par(mfrow=c(1,1))
  par(mar=c(4.5,4.5,1.5,1.5))
  plot(successRates,main = "Logistic Regression Success Rates",
       xlab="Probability Threshold",ylab="Success Rate",type="l")
  ## print(which.max(successRates[,2]))
  points(successRates[which.max(successRates[,2]),1],
                      successRates[which.max(successRates[,2]),2],
                                   cex=1.5,col="purple",)
  best_th = successRates[which.max(successRates[,2]),1]
  best_th
}

## this function splits a dataframe into two dataframes split by gender
## ... performs analysis on each dataframe separately
## ... outputs the predictions together
gender_split <- function(myData) {
  if(sum(names(myData) == "Sex") == 0) {
    stop("No Sex column in data frame")
  }
  
  ## This means we are dealing with the training data
  ## which does NOT include a PassengerId column
  
  if(sum(names(myData) == "PassengerId") == 0) {
    male_data = filter(myData,Sex=="male")
    male_data = male_data %>% dplyr::select(-c(Sex))
    female_data = filter(myData,Sex=="female")
    female_data = female_data %>% dplyr::select(-c(Sex))
    
    genders_list = list(male_data,female_data)
    names(genders_list) = c("male","female")
    return(genders_list)
  }
  
  ## This means we are dealing with the test data
  ## which includes a PassengerId column to be removed
  
  else {
    male_data = filter(myData,Sex=="male")
    male_Ids = male_data$PassengerId
    male_data = male_data %>% dplyr::select(-c(Sex,PassengerId))
    
    female_data = filter(myData,Sex=="female")
    female_Ids = female_data$PassengerId
    female_data = female_data %>% dplyr::select(-c(Sex,PassengerId))
    
    genders_list = list(male_Ids,male_data,female_Ids,female_data)
    names(genders_list) = c("maleIds","male","femaleIds","female")
    return(genders_list)
  }

}

## This function takes in a list of lists containing 
## stratified prediction information (separated by Sex, Age...)
## combines the predictions to give an overall prediction accuracy...

## returns a dataframe containing column of test predictions, test data
combine_preds <- function(myPreds) {
    all_testpreds = c()
    all_testdata = c()
    for(i in 1:length(myPreds)) {
      all_testpreds = c(all_testpreds,myPreds[[i]]$testpreds)
      all_testdata = c(all_testdata,myPreds[[i]]$testdata)
    }
    full_model = as.data.frame(cbind(all_testpreds,all_testdata))
    names(full_model) = c("testpreds","testdata")
    full_model
}

displayTable <- function(model) {
  cat("\n")
  print(table("Test preds:"=model$testpreds, "Test data:"=model$testdata))
  cat("Overall success rate:",mean(model$testpreds==model$testdata),"\n")
}

getSuccessRate <- function(model) {
  successRate = mean(model$testpreds==model$testdata)
  successRate
}

## Workspace... to execute functions

## the final submission should use the models used on clean_data_v5 ! 
## ... in which outliers are removed 

gender_data = gender_split(clean_data_v5)
bestmodel_male = basicModels(gender_data$male,"male data")
bestmodel_female = basicModels(gender_data$female,"female data")

all_data = combine_preds(list(bestmodel_male,bestmodel_female))
## all_data_v2 = basicModels(clean_data_v3)
displayTable(all_data)

## bestmodel_male uses glm (logistic)
## bestmodel_female uses qda (!)

makePredictions <- function(test_data_clean) {
  
  ## From the training analysis:
  
  ## bestmodel_male uses glm (logistic)
  ## bestmodel_female uses qda (!)
  
  gender_data = gender_split(test_data_clean)
  
  if(sum(class(bestmodel_male$model) == c("glm","lm")) == 2) {
    best_th = bestmodel_male$best_th
    male_probs = predict(bestmodel_male$model,newdata=gender_data$male,type="response")
    male_preds = rep(0,nrow(gender_data$male))
    male_preds[male_probs > best_th] = 1
    cat(class(bestmodel_male$model),"was applied to male data\n")
  }
  else if(class(bestmodel_male$model) == "lda" | 
          class(bestmodel_male$model) == "qda") {
    male_preds = predict(bestmodel_male$model,newdata=gender_data$male)
    male_preds = as.numeric(as.character(male_preds$class))
    cat(class(bestmodel_male$model),"was applied to male data\n")
  }
  else if(class(bestmodel_female$model) == "tree") {
    male_preds = predict(bestmodel_male$model,newdata=gender_data$male)
    cat(class(bestmodel_male$model),"was applied to male data\n")
  }
  else {
    stop("None of the supplied models were applied to data\n")
  }
  
  if(sum(class(bestmodel_female$model) == c("glm","lm")) == 2) {
    best_th = bestmodel_female$best_th
    female_probs = predict(bestmodel_female$model,newdata=gender_data$female,type="response")
    female_preds = rep(0,nrow(gender_data$female))
    female_preds[male_probs > best_th] = 1
    cat(class(bestmodel_female$model),"was applied to female data\n")
  }
  else if(class(bestmodel_female$model) == "lda" | 
          class(bestmodel_female$model) == "qda") {
    female_preds = predict(bestmodel_female$model,newdata=gender_data$female)
    female_preds = as.numeric(as.character(female_preds$class))
    cat(class(bestmodel_female$model),"was applied to female data\n")
  }
  else if(class(bestmodel_female$model) == "tree") {
    female_preds = predict(bestmodel_female$model,newdata=gender_data$female)
    cat(class(bestmodel_female$model),"was applied to female data\n")
  }
  else {
    stop("None of the supplied models were applied to data\n")
  }
  

  ## female_preds = predict(bestmodel_female$model,newdata=gender_data$female)
  
  cat("\nThe length is",length(gender_data$maleIds),"\n",gender_data$maleIds)
  cat("\nThe length is",length(male_preds),"\n",male_preds)
  cat("\nThe length is",length(gender_data$femaleIds),"\n",gender_data$femaleIds)
  cat("\nThe length is",length(female_preds),"\n",female_preds)
  all_preds = data.frame(PassengerId = c(gender_data$maleIds,gender_data$femaleIds),
        Survived = c(male_preds,female_preds))
  final_preds = arrange(all_preds,PassengerId)
  final_preds
}

## Run the same stuff on test data
sample = read.csv(file="gender_submission.csv",header=T,na.string=c(""))

test_data_raw = read.csv(file="test.csv",header=T,na.string=c(""))
test_data_clean = cleanData(test_data_raw)
test_submission = makePredictions(test_data_clean)

## write.csv(test_submission,file="submission_6_22_v5.csv",row.names=FALSE)

