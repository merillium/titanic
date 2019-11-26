setwd("/Users/derek/Desktop/kaggle/titanic/")

library(dplyr)
library(glmnet)
library(MASS)
library(tree)
library(ggplot2)
library(caret) ## useful for machine learning, and one-hot encoding
library(Amelia) ## useful for visualizing missing data

raw_data = read.csv(file="train.csv",header=T,na.string=c(""))

## This function does some exploratory analysis 
## including generating plots between variables
## generating correlation tables
## running 

exploreData <- function(myData) {
  ## myData$Survived = as.factor(myData$Survived)
  par(mar=c(5,4,2,1)+0.2) 
  par(mfrow=c(2,2))
  ## boxplot(Age~Survived,data=myData,main="Age by survival")
  hist(myData$Age,xlab="Age",main="Distribution of Age")
  survival_by_sex = table(myData$Sex,myData$Survived)
  print(survival_by_sex)
  female_rate = survival_by_sex[1,2] / sum(survival_by_sex[1,1:2])
  male_rate = survival_by_sex[2,2] / sum(survival_by_sex[2,1:2])
  cat("Female rate of survival = ",female_rate)
  cat("\nMale rate of survival = ",male_rate)
  age_split_data = mutate(myData, Age_quantile = ntile(myData$Age,5))
  
  survival_by_age = rep(0,5)
  quantiles_by_age = quantile(myData$Age, probs=c(0.20,0.40,0.60,0.80),na.rm=TRUE)
  for(i in 1:5) {
    age_quantile_data = age_split_data %>% filter(Age_quantile == i)
    survival_by_age[i] = sum(age_quantile_data$Survived) / nrow(age_quantile_data)
  }
  ## print(survival_by_age)
  ## print(quatiles_by_age)
  quantiles_by_age_text = c("0-19","19-25","25-31.8","31.8-41","41-80")
  barplot(survival_by_age,names.arg=quantiles_by_age_text,cex.names=0.8,
          xlab="Ages",ylab="Survival Rate",main="Survival by Age")
  
  hist(myData$Pclass,xlab="Pclass",main="Distribution of Pclass")
  survival_by_Pclass = rep(0,3)
  for(i in 1:3) {
    Pclass_data = myData %>% filter(Pclass == i)
    survival_by_Pclass[i] = sum(Pclass_data$Survived)/nrow(Pclass_data)
  }
  barplot(survival_by_Pclass,names.arg=c("1","2","3"),cex.names=1,
          xlab="PClass",ylab="Survival Rate",main="Survival by Pclass")
  
  ## Exploratory analysis of missing value data
  data_NA <- myData[rowSums(is.na(myData)) > 0,]
  par(mfrow=c(2,2))
  survival_by_sex_NA = table(data_NA$Sex,data_NA$Survived)
  print(survival_by_sex_NA)
  female_rate_NA = survival_by_sex_NA[1,2] / sum(survival_by_sex_NA[1,1:2])
  male_rate_NA = survival_by_sex_NA[2,2] / sum(survival_by_sex_NA[2,1:2])
  cat("NA Female rate of survival = ",female_rate_NA)
  cat("\nNA Male rate of survival = ",male_rate_NA)
}

# cleanData:
# (1) determines the percentage of NAs in each column, 
# prints out table, displays missingness map 
# (2) selects preliminary variables 


cleanData <- function(raw_data) {
  percentNA <- sapply(raw_data,function(x) sum(is.na(x))/nrow(raw_data)*100)
  ## print(percentNA)
  missmap(raw_data) 
  # Cabin has too many NA values to be considered; Name,Ticket are NOT useful variables 
  selected_data <- raw_data %>% dplyr::select(c(Survived,Pclass,Sex,Age,
                                                SibSp,Parch,Fare,Embarked))
  
  ## to remove NAs...
  ## selected_data = na.omit(selected_data)
  # clean_data$Age[is.na(clean_data$Age)] = median(raw_data$Age,na.rm=T)
  selected_data
}

cleaned_data = cleanData(raw_data)
exploreData(cleaned_data)