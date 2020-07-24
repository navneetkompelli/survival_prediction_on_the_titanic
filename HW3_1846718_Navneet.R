# This nootebook describes the process of creating a model that predicts which passengers survived the Titanic shipwreck.
# 
# The train dataset consists of 891 observations and test consists 418 observations whose "Survived" feature (1=survived 0=dead) needs to predicted.
# 
# Exploratory Data Analysis
# 
# Creating new features from existing variables
# Data exploration using plots
# Impute Missing Values
# Data preparation
# Prediction and Evaluation
# 
# Logistic regression model
# Decision tree model using rpart
# Naive Bayes model
# Decision tree model using caret and cross-validation
# Random forest models
# Neural network models
# SVM
# Model Deployment
# 
# Conclusion






# Required libraries are loaded below:
library(lmtest) # for likelihood ratio test
library(randomForest)
library(e1071) # for Naive Bayes
library(caret)
library(rpart.plot)
library(rattle)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(rpart)


# Exploratory data analysis


# Datasets are imported, their summary indicates NAs exist in both datasets. 
# To effectively impute missing values, these two datasets are merged temporarily into a new dataframe.

df1 <- read.csv("train.csv")
df2 <- read.csv("test.csv")

summary(df1)
summary(df2)

df3 <- bind_rows(df1, df2)


# number of missing values, NAs and blanks existing in each variable:
#   
# 418 missing values in Survived belong to test dataset which wouldn't be imputed
# 263 passengers' age values are missing in both datasets (177 in train + 86 in test)
# 1 passenger's fare missing in test dataset (row# 153 or 1044 in merged dataset)
# 1014 cabin values are blanks (687 in train + 327 in test)
# 2 embarked values missing in train dataset (row# 62 and 830)


NAcount <- function(attr){sum(is.na(attr))}
sapply(df3,NAcount)

emptycount <- function(attr){sum(attr=="")}
sapply(df3,emptycount)
sapply(df1,emptycount)
sapply(df2,emptycount)

which(is.na(df3$Fare))
which(is.na(df2$Fare))
which(df3$Embarked=="")


# Creating new features from existing variables:
# Size variable indicates family size, calulated from # of siblings / spouses and # of parents / children related to a passenger

df3$Size <- df3$SibSp + df3$Parch + 1 #includes self

# Name column includes title of the passenger which is split as a separate variable. This regex code is inspired from Meg Risdal's titanic tutorial "Exploring Survival on the Titanic".

df3$Title <- gsub("^.*, (.*?)\\..*$","\\1",df3$Name)
table(df3$Sex, df3$Title)

# The french titles are merged into their english equivalents:
  
df3$Title[df3$Title %in% c("Ms", "Mlle")] <- "Miss"
df3$Title[df3$Title == "Mme"] <- "Mrs"

# Other minority titles are grouped as special titles in order to avoid training for each individual title which could lead to over-fitting

df3$Title[df3$Title %in% c("Capt","Col","Don","Dona","Dr","Jonkheer","Lady","Major","Rev","Sir","the Countess")] <- "Special"
table(df3$Sex, df3$Title)


# Data exploration using plots:
  
# The three boxplots below assumes many outliers. In this case these outliers need not be eliminated because it is generally plausible to have passengers with ages>70, have family sizes greater than 10, and pay higher than average fares. Including these values could help model training result in better prediction of survival.
boxplot(df3$Age, ylab="Age value", xlab="Age", main="Age boxplot")
boxplot(df3$Fare, ylab="Fare value", xlab="Fare", main="Fare boxplot")
boxplot(df3$Size, ylab="Family size value", xlab="Size", main="Family Size boxplot")

# Below boxplot indicates that passengers of class 1 have higher fares than that of passengers of classes 2 and 3. Similarly class 3 passengers paid lowest fares:
  
ggplot(na.omit(df1), aes(x=as.factor(Pclass), y=Fare)) + geom_boxplot()

# Below plot indiactes that class1 passengers that paid high fares have higher chance of survival, whereas class3 passengers have lowest probability of survival.

ggplot(na.omit(df1), aes(x=as.factor(Pclass), fill = as.factor(Survived))) + geom_bar(position="dodge")

# Below histogram suggests that passengers that paid higher fares could have higher probablility of survival as implied in above plot

ggplot(na.omit(df1), aes(x=Fare, fill = as.factor(Survived))) + geom_histogram()

# Below plot indicates that female passengers have high survival rates than male passengers

ggplot(na.omit(df1), aes(x=as.factor(Sex), fill=as.factor(Survived))) + geom_bar()

# From the below plot, it could be deduced that toddlers and young kids (age less than 8 years) have higher survival probability than adult passengers.

ggplot(na.omit(df1), aes(x=Age, fill = as.factor(Survived))) + geom_histogram()


# Impute Missing Values
# Embarked: It could be observed that both the passengers with missing embarking location have the same fare of $80. The boxplot shows that the median fare of first class passengers aligns with those of missing emarkation. Hence the missing values could be imputed with this median value.

df3[df3$Embarked=="",]
ggplot(df1, aes(x=as.factor(Pclass), y=Fare, color=Embarked)) + geom_boxplot()
median(df1$Fare[df1$Pclass=="1" & df1$Embarked=="C"])
df3$Embarked[df3$Embarked==""] <- 'C'
df1$Embarked[df1$Embarked==""] <- 'C'
df3[62,]
df3[830,]

# Age: Imputing missing age using linear regression model yields few negative age values which is erroneous.
lm1 <- lm(Age ~ SibSp + Parch + Fare + Sex + Pclass,data = df1[!is.na(df1$Age),-1])
lm_age <- predict(lm1, df1[is.na(df1$Age),])
summary(lm_age) #Min indicates negative values for age. So not an useful model

# Using rpart to impute missing age values. Here train and test datasets are imputed separately to prevent data leaking into test dataset, eliminating potential bias.
summary(df1$Age)
rpart1 <- rpart(Age ~ SibSp + Parch + Fare + Sex + Pclass, data = df1[!is.na(df1$Age),-1])
df1$Age[is.na(df1$Age)] <- predict(rpart1, df1[is.na(df1$Age),])
summary(df1$Age)

summary(df2$Age)
rpart2 <- rpart(Age ~ SibSp + Parch + Fare + Sex + Pclass, data = df2[!is.na(df2$Age),-1])
df2$Age[is.na(df2$Age)] <- predict(rpart1, df2[is.na(df2$Age),])
summary(df2$Age)

df3$Age[1:891] <- df1$Age
df3$Age[892:1309] <- df2$Age

# Fare: Similarly rpart is used to impute missing fare value in test dataset.
df3[is.na(df3$Fare),]
df2[is.na(df2$Fare),]

rpart3 <- rpart(Fare ~ Embarked + Pclass, data = df2[!is.na(df2$Fare),-1])
df3$Fare[is.na(df3$Fare)] <- predict(rpart3, df3[is.na(df3$Fare),])
summary(df3$Fare)
df3[1044,]

# Cabin: 1014 (77% of 1309) passengers have missing cabin values, hence this variable could be eliminated from the dataset
emptycount(df3$Cabin) ##77% missing values
df3$Cabin <- NULL

sapply(df3,NAcount)
sapply(df3,emptycount)

# Approprite classes are assigned to the variables:
df3$PassengerId <- as.factor(df3$PassengerId)
df3$Survived <- as.factor(df3$Survived)
df3$Pclass <- as.factor(df3$Pclass)
df3$Name <- as.factor(df3$Name)
df3$Ticket <- as.factor(df3$Ticket)
df3$Embarked <- as.factor(df3$Embarked)
df3$Size <- as.integer(df3$Size)
df3$Title <- as.factor(df3$Title)


# Data preparation
# train and test datasets are reverted to their original configurations with no missing values
train <- df3[1:891,]
test <- df3[892:1309,]
test$Survived <- NULL
summary(train)
summary(test)

# Train dataset is split into train(80%) and validation(20%) sets to be able to evaluate model performance on labelled 
# validation data before applying on unknown or unlabelled (test) data.
set.seed(729375)
inTraining <- createDataPartition(train$Survived, p=0.8, list = FALSE)
training <- train[inTraining,]
validation <- train[-inTraining,]
summary(training)
summary(validation)


# Prediction and Evaluation:
#   Logistic regression model: Name and Ticket variables are excluded from the model because name is unique like 
# passenger ID and Ticket number does not hold predictive power to determine survival of a passenger. The results 
# indicate that Pclass, SibSp, Title are one of the most important variables in classifying Survived variable. 
# Confusion matrix results imply that this model was able to accurately classify survival of 84% validation set passengers.

mylogit <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + Size, 
               data = training,  family = "binomial")
summary(mylogit)
lrtest(mylogit)
glm_pred <- predict(mylogit, newdata = validation, type = "response")
glm_pred_log <- ifelse(glm_pred > 0.5, 1, 0)
confusionMatrix(as.factor(glm_pred_log), as.factor(validation$Survived), positive = '1')


# Decision tree model using rpart: Results indicate slightly inproved accuracy of 85%, but an increase in false positives. 
# From the decision tree plot, it could be deduced that young males & specials of classes 2 and 3 are not likely to 
# survive. The tree need not be pruned because it is already small and it doesn't include important variables like 
# passenger's sex. But as deduced earlier, female passengers had higher probability of survival. Additionally, defining 
# lower complexity parameters and remodelling resulted in the same decision tree.

dtree <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + 
                 Embarked + Size + Title, data = training, method = 'class')
fancyRpartPlot(dtree)
predValidation <- predict(dtree, newdata=validation, type = 'class')
confusionMatrix(as.factor(validation$Survived), as.factor(predValidation), positive = '1')

dtree2 <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Size + Title, data = training, method = 'class', 
                control=rpart.control(cp=0.0001,minsplit=20,maxdepth=4))
fancyRpartPlot(dtree2)


# Naive Bayes model: Resulted in substantially lower balanced accuracy and decreased sensitivity

bayesModel = naiveBayes(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Size + Title, data=training)
NB_Predictions2 = predict(bayesModel,validation)
confusionMatrix(as.factor(validation$Survived), as.factor(NB_Predictions2), positive = '1')


# Since an imbalance of Survival classification exists in training dataset (549 survived and 342 dead), accuracy is not a 
# good metric to evaluate models, hence metric="Kappa" is used to train caret to choose the best model. Additionally, since 
# the dataset is small (891), model training misses on the 20% (177) data assigned to validation set. To avoid this, 10-fold 
# cross validation is implemented.

trctrl <- trainControl(method = "cv", number = 10)

# False positive error occurs when the prediction is true (1) but the truth is false (0). In this case, FP error occurs when 
# survival prediction is 1 but truth is 0, i.e. falsely predicting a dead passenger as survived. Thus, FP errors are relatively 
# costlier than FN (falsely predicting a survived passenger as dead). Hence, the objective in model evaluation is to minimize 
# the number of false positive errors and ensure FP < FN.

# Decision tree model using caret and cross-validation: Highest kappa value and accuracy is achieved when complexity 
# parameter=0. Also, the decision tree evolved larger than the one yielded earlier by rpart. Results show that accuracy is now 
# around 81% implying previous accuracies were impacted by the imbalance of the dataset. Also, the number of false positives 
# are lowered by this decision tree model.

dtree3 <- train(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Size + Title, 
                data = train, method = "rpart", parms = list(split = "information"), metric="Kappa", trControl=trctrl, tuneLength = 20)
dtree3
prp(dtree3$finalModel)
confusionMatrix(dtree3, positive = '1')


# Random forest models: Constructs several decision trees to be able to individually predict each passenger's survival and
# vote thetree if the classification is correct. Thus during training, the best decision tree with most votes would be chosen. 
# The model chosen here is according to the highest kappa value of 0.65 and mtry(# preditors chosen)=6, yielding an accuracy 
# of 83%. It could also be observed that this model resulted much lower false positives(6.3), lower than the previous decision 
# tree using caret.
randomf_fit <- train(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Size + Title, 
                     data = train, method = "rf",
                     parms = list(split = "information"),
                     metric="Kappa",
                     trControl=trctrl,
                     tuneLength = 10)
randomf_fit
confusionMatrix(randomf_fit, positive = '1')


# # Each variable's importance is indicated below along with plot against mean decrease in node impurity. Implies Fare, 
# Title=Mr, Sex=male, and Age are the four predictors which could accurately classify Survived variable. Furthermore, 
# training this model with grid search and mtry between 1 and 10 predictors yielded in similar results.
varImp(randomf_fit)
varImpPlot(randomf_fit$finalModel,type=2)

control <- trainControl(method="cv", number=10, search="grid")
tunegrid <- expand.grid(mtry=c(1:10)) #Only parameter searchable
rf_gridsearch <- train(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Size + Title, 
                       data=train, method="rf", metric='Kappa', 
                       ntree=100,
                       tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)
confusionMatrix(rf_gridsearch, positive = '1')


# # Neural network models: A simple neural network model is trained with intermediate layer size=1 and decay=0.017 which 
# yielded better accuracy and lower false positives resulting in better precision (PPV = TP/TP+FP) of 83%. Additionally, 
# oversampling is implemented to fix the imbalance in dataset, and variables are preprocessed by scaling and centering 
# (size=5 and decay=0.042) which did not result in an improved accuracy.
nn_fit <- train(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Size + Title, 
                data = train, method = "nnet",
                metric="Kappa",
                trControl=trctrl,
                tuneLength = 10)
nn_fit
confusionMatrix(nn_fit, positive = '1')

trctrlup <- trainControl(method = "cv", number = 10, sampling = "up")
nn_fit_upp <- train(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Size + Title, 
                    data = train, method = "nnet", preProcess = c('center','scale'), 
                    metric="Kappa",
                    trControl=trctrlup,
                    tuneLength = 10)
nn_fit_upp
confusionMatrix(nn_fit_upp, positive = '1')


# SVM: Finally, SVM model using polynomial classification yielded better results than the previous models by a substantially 
# lower false positives (4.3) and maintained a good accuracy of 83.39% resulting in an higher precision and positive 
# predictive value of 95%. Thus this model is chosen as the ideal one for this dataset.
mSVM <- train(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Size + Title, 
              data = train, method = "svmLinear", trControl=trainControl(method='cv',number=10))
mSVM2 <- train(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Size + Title, 
               data = train, method = "svmRadial", trControl=trainControl(method='cv',number=10))
mSVM3 <- train(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Size + Title, 
               data = train, method = "svmPoly", trControl=trainControl(method='cv',number=10))

confusionMatrix(mSVM, positive = '1')
confusionMatrix(mSVM2, positive = '1')
confusionMatrix(mSVM3, positive = '1')


# Model Deployment:
# The chosen model (polynomial SVM) is applied on the test dataset to predict the unknown variable "Survived" (1=survived 
# 0=dead). The resulting predictions and passenger IDs are merged into a dataframe which is saved as the final output file.

# Predict using the test set
pred_mSVM3 <- predict(mSVM3, newdata = test)
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(PassengerID = test$PassengerId, Survived = pred_mSVM3)
# Write the solution to file
write.csv(solution, file = 'mSVM3_polyn.csv', row.names = F)


# Conclusion:
# Merged the imported datasets, performed exploratory data analysis by creating new features, increasing higher predictive 
# power to classify the survival of passengers. Missing data is appropriately imputed or eliminated. Data is split back into 
# their original configurations. Training data is split into train and validation sets for initial model deployments. The 
# initial regression and decision tree model results and plots were interpreted to estimate the most important features that 
# impact survival of passengers. To avoid losing many observations to validation set, 10-fold cross validation set is 
# implemented in next models. Random Forest and Neural Network models are implemented. Models are analyzed and evaluated by 
# their accuracy, precision, sensitivity, false positive and false negative trade-offs specified in the confusion matrix 
# results. Additionally, these models were tuned by oversampling the dataset and preprocessing variables which yielded 
# approximately similar results as earlier. Furthermore, SVM models were implemented which yielded ideal results by effectively 
# minimizing false positive errors than false negative errors and also ensuring 84% accuracy. Finally, the chosen polynomial 
# SVM model is deployed on the test dataset to predict the survival.
