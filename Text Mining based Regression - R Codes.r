##---------------------------------------------------------------------------------------------------------
##| Name			: Predicting XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
##| Created By		: Shashank Sharma
##| Date Created	: 2016 03 18
##| Date Updated	: _
##| Updated By		: _
##| Purpose			: Part of calculating # Application / $ 
##---------------------------------------------------------------------------------------------------------

##| Setting Directory

getwd()
setwd("C:/Users/shashank/Desktop/JS/00 Datasets/")
getwd()

##| Importing Required Packages
library(caret)
library(rpart)
library(rpart.plot)
library(caTools)
library(e1071)
library(DAAG)
library(tm)
library(xgboost)
library(dummies)


##| Importing Datasets 
Inp <- read.csv("data.csv")  ##| Cleaned up Data with MSA to State Area Mapping

summary(Inp)
str(Inp)

##| Creating TermDocumentMatrix from JOB_TITLE

#| Text Column
 
txtData = paste(Inp$JOB_TITLE)
reviewSource = VectorSource(txtData)

#| Creating corpus and cleaning
corpus = Corpus(reviewSource)
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, stripWhitespace)
corpus = tm_map(corpus, removeWords, stopwords("english"))

#| DTM

dtm = DocumentTermMatrix(corpus)


###| Small DTM
smallDTM = removeSparseTerms(dtm, sparse = 0.99)
x = as.matrix(smallDTM)

##| Choice of Variables

##| Using PLAN_TYPE, STATE_AREA (from MSA_NAME), JOB_CATEGORY and DocumentTermMatrix(JOB_TITLE),  to predict APPLICATIONS (# of applications)

##| Creating dummy variables for calculations

temp = dummy.data.frame(Inp[,c(3,7,8,11,13)], names = NULL, omit.constants=TRUE, dummy.classes = getOption("dummy.classes"))

df = data.frame(temp, x)



Inp = df


rm(corpus, txtData, reviewSource, dtm, temp)

##| Data Partitioning
spl1 <- sample.split(Inp$APPLICATIONS, SplitRatio = 0.8)

Test <- Inp[!spl1,]
temp <- Inp[spl1,]

summary(Test)
summary(temp)
str(Test)
str(temp)


spl2 <- sample.split(temp$APPLICATIONS, SplitRatio = 0.8)
Train <- temp[spl2,]
Validate <- temp[!spl2,]

summary(Train)
summary(Validate)
str(Train)
str(Validate)

rm(temp)

##---------------------------------------------------------------------------------------------------------
##| Testing Different Models
##---------------------------------------------------------------------------------------------------------


##|| Trying Regression Tree

model1 <- rpart(APPLICATIONS ~ ., data = Train)
prp(model1)
summary(model1)
pred1 <- predict(model1, newdata = Validate)

err = (Validate$APPLICATIONS - pred1)^2
sum(err)		#Validation Error


##|| Trying Linear Regression

model2 <- lm(APPLICATIONS ~ ., data = Train)
summary(model2)

coefficients(model2) # model coefficients
confint(model2, level=0.95) # CIs for model parameters
fitted(model2) # predicted values
residuals(model2) # residuals
anova(model2) # anova table
vcov(model2) # covariance matrix for model parameters
influence(model2) # regression diagnostics

pred2 <- predict(model2, newdata = Validate)

err = (Validate$APPLICATIONS - pred2)^2
sum(err)		#Validation Error

##| lm works better than regression tree

##---------------------------------------------------------------------------------------------------------
##| Cross Validation With Boosting
##---------------------------------------------------------------------------------------------------------

##| Using Extreme Gradient Boosting with Cross Validation (XGBoost)


##| Matrices for XGBOOST

XTRAIN = subset(Train, select = -APPLICATIONS)
YTRAIN = as.vector(subset(Train, select = APPLICATIONS))

XVALIDATE = subset(Validate, select = -APPLICATIONS)
YVALIDATE = subset(Validate, select = APPLICATIONS)

XTEST = subset(Test, select = -APPLICATIONS)
YTEST = subset(Test, select = APPLICATIONS)



##| XGBOOST

params <- list(booster = "gblinear", objective = "reg:linear", eta = 0.01)
			   
model3 <- xgboost(params =  params, data = as.matrix(XTRAIN), label = as.matrix(YTRAIN), nrounds = 10000, nfold = 10, early.stop.round = NULL, maximize = T)

pred3 <- predict(model3, as.matrix(XVALIDATE))

err = (YVALIDATE - pred3)^2

sum(err)

##| Final Training and Test

XTRAINING = as.matrix(rbind(XTRAIN, XVALIDATE))
YTRAINING = as.matrix(rbind(YTRAIN, YVALIDATE))

model4 <- xgboost(params =  params, data = XTRAINING, label = YTRAINING, nrounds = 10000, nfold = 10, early.stop.round = NULL, maximize = T)

pred3 <- predict(model3, as.matrix(XTEST))

err = (YTEST - pred3)^2

sum(err)

plot(pred3 , YTEST$APPLICATIONS)  ##| Hetroskedicity

####| Needs further evaluation

##---------------------------------------------------------------------------------------------------------
##| XGBoost with log transformation
##---------------------------------------------------------------------------------------------------------

LYTRAIN = log(1+YTRAIN)

LYVALIDATE = log(1+YVALIDATE)

LYTEST = log(1+YTEST)

LYTRAINING = log(1+YTRAINING)

params <- list(booster = "gblinear", objective = "reg:linear", eta = 0.01)

model4 <- xgboost(params =  params, data = XTRAINING, label = LYTRAINING, nrounds = 1000, nfold = 10, early.stop.round = NULL, maximize = T)

pred4 <- predict(model4, as.matrix(XTEST))

err = (LYTEST - pred4)^2

sum(err)

plot(pred4 , LYTEST$APPLICATIONS) ##| Hetroskedicity still present

##| Need (higher order / more) features or will have to perform the family of Box-Cox transformations for improvement 
