Practical Machine Learning - Final Course Project
================
Alejandra Quiñones
07 May 2020

### Introduction

This document is being submitted as part of the Final Project of the Coursera Practical Machine Learning Course. Using data from accelerometers on the belt, forearm, arm and dumbell (cite on README.md), a model was created to predict if barbell lifts were performed correctly or incorrectly. Class A corresponds to a correct execution, while classes B to E correspond to common mistakes.

### Data and package loading

``` r
library(caret); library(corrplot); library(randomForest); library(pROC)
     
training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
testing <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
```

### Exploring the data

``` r
dim(training); dim(testing)
```

    ## [1] 19622   160

    ## [1]  20 160

At first glance, both datasets have 160 variables. The training set is a large dataset with 19622 observations while the testing set holds 20. Next, a function was created to show the name, the amount of complete cases and the class of each variable.

``` r
showna <- function(df){
     nadf <- data.frame(colnames(df))
     nadf$complete <- NA
     nadf$colclass <- NA
     for (i in 1:ncol(df)){
          na <- sum(complete.cases(df[,i]))
          nadf[i,][,2] <- na
          nadf[i,][,3] <- class(df[,i])
     }
     return(nadf)
}

sumtrain <- showna(training)
head(sumtrain, 20)
```

    ##            colnames.df. complete colclass
    ## 1                     X    19622  integer
    ## 2             user_name    19622   factor
    ## 3  raw_timestamp_part_1    19622  integer
    ## 4  raw_timestamp_part_2    19622  integer
    ## 5        cvtd_timestamp    19622   factor
    ## 6            new_window    19622   factor
    ## 7            num_window    19622  integer
    ## 8             roll_belt    19622  numeric
    ## 9            pitch_belt    19622  numeric
    ## 10             yaw_belt    19622  numeric
    ## 11     total_accel_belt    19622  integer
    ## 12   kurtosis_roll_belt    19622   factor
    ## 13  kurtosis_picth_belt    19622   factor
    ## 14    kurtosis_yaw_belt    19622   factor
    ## 15   skewness_roll_belt    19622   factor
    ## 16 skewness_roll_belt.1    19622   factor
    ## 17    skewness_yaw_belt    19622   factor
    ## 18        max_roll_belt      406  numeric
    ## 19       max_picth_belt      406  integer
    ## 20         max_yaw_belt    19622   factor

``` r
sumtest <- showna(testing)
head(sumtest, 20)
```

    ##            colnames.df. complete colclass
    ## 1                     X       20  integer
    ## 2             user_name       20   factor
    ## 3  raw_timestamp_part_1       20  integer
    ## 4  raw_timestamp_part_2       20  integer
    ## 5        cvtd_timestamp       20   factor
    ## 6            new_window       20   factor
    ## 7            num_window       20  integer
    ## 8             roll_belt       20  numeric
    ## 9            pitch_belt       20  numeric
    ## 10             yaw_belt       20  numeric
    ## 11     total_accel_belt       20  integer
    ## 12   kurtosis_roll_belt        0  logical
    ## 13  kurtosis_picth_belt        0  logical
    ## 14    kurtosis_yaw_belt        0  logical
    ## 15   skewness_roll_belt        0  logical
    ## 16 skewness_roll_belt.1        0  logical
    ## 17    skewness_yaw_belt        0  logical
    ## 18        max_roll_belt        0  logical
    ## 19       max_picth_belt        0  logical
    ## 20         max_yaw_belt        0  logical

At first glance, both sets have many variables with a considerable amount of NAs, and some wrongly classified. To prevent future issues, the columns 12 to 159 will be reclassified as numeric. In addition, since both data frames hold the same variables, their levels were made equal.

``` r
#Fixing variable classification
     
for (i in 12:159){
     training[,i] <- as.numeric(as.character(training[,i]))
}
rm(i)
     
for (i in 12:159){
     testing[,i] <- as.numeric(as.character(testing[,i]))
}
rm(i)
     
#Adjusting levels
for (i in 1:ncol(testing)){levels(testing[,i]) <- levels(training[,i])}
rm(i)
```

### Data preprocessing

Once classification errors were fixed, training set preprocessing was carried out. Since multiple variables held a significant amount of missing values, imputing them could lead to overfitting. Thus, variables with less than 70% of complete cases were removed. This left the new training set with 60 variables.

``` r
trainclean <- training
trainclean <- training[, colSums(is.na(training)) <= nrow(training)*0.3]
```

A correlation analysis was carried out.

``` r
cormat <- cor(trainclean[,7:59])
corrplot(cormat, method="color", type="lower", order="AOE", tl.cex = .3, tl.col = "black", diag = F)
```

![](Analysis_Report_files/figure-markdown_github/Correlation%20Matrix-1.png)

A few variables showed a high level of correlation. Preprocessing by PCA could be used to solve this, but since a lot of variables were already dismissed, this could be bypassed by using a robust model.

Next, the preProcess function of the caret package was used to detect and dismiss variables with near zero variance. Variables unrelated to the accelerometers were removed as well. Once this was done, the training set was ready for the modeling algorithms.

``` r
preobj <- preProcess(trainclean, method="zv")
     
#New values
pptrain <- predict(preobj, trainclean)
pptrain$classe <- trainclean$classe 
```

### Model fitting

Training set spliting
---------------------

Before creating the prediction models, the training set was split into a building set, based on which the models were created, and a validation set, where the models were then tested.

``` r
set.seed(3333)
inBuild <- createDataPartition(y=pptrain$classe, p=0.7, list=F)
validation <- pptrain[-inBuild,]
build <- pptrain[inBuild,] 
```

Next up, three modeling algorithms that could predict multiclass outcomes with high accuracy were used to fit two models. Traincontrol arguments were set with 3-fold cross-validation. All were then tested on the validation set.

Decision Tree Model
-------------------

``` r
modtree <- train(classe ~., data=build, method="rpart", trControl = trainControl(method="cv", number=3))
modtree
```

    ## CART 
    ## 
    ## 13737 samples
    ##    59 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold) 
    ## Summary of sample sizes: 9158, 9158, 9158 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp         Accuracy   Kappa    
    ##   0.2437188  0.7195894  0.6434158
    ##   0.2568406  0.5389095  0.4103205
    ##   0.2703692  0.2843416  0.0000000
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was cp = 0.2437188.

``` r
#Predicting outcome in validation set     
predtree <- predict(modtree, newdata = validation)
confusionMatrix(predtree, validation$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    0    0    0    0
    ##          B    0 1139    0    0    0
    ##          C    0    0    0    0    0
    ##          D    0    0    0    0    0
    ##          E    0    0 1026  964 1082
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.6619          
    ##                  95% CI : (0.6496, 0.6739)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.5696          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   0.0000   0.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   0.5857
    ## Pos Pred Value         1.0000   1.0000      NaN      NaN   0.3522
    ## Neg Pred Value         1.0000   1.0000   0.8257   0.8362   1.0000
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1935   0.0000   0.0000   0.1839
    ## Detection Prevalence   0.2845   0.1935   0.0000   0.0000   0.5220
    ## Balanced Accuracy      1.0000   1.0000   0.5000   0.5000   0.7928

``` r
acctree <- confusionMatrix(predtree, validation$classe)$overall["Accuracy"]
```

Random Forest Model
-------------------

For this model, an mtry value of 2 was considered based on a previous random forest model in which the highest accuracy was gotten with said mtry value. Specifying this parameter will make the processing much faster.

``` r
modrf <- randomForest(classe ~., data=build, mtry=2, importance=TRUE, trControl = trainControl(method="cv"), number = 3)
modrf
```

    ## 
    ## Call:
    ##  randomForest(formula = classe ~ ., data = build, mtry = 2, importance = TRUE,      trControl = trainControl(method = "cv"), number = 3) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 0.12%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 3906    0    0    0    0 0.0000000000
    ## B    1 2657    0    0    0 0.0003762227
    ## C    0    3 2393    0    0 0.0012520868
    ## D    0    0    8 2244    0 0.0035523979
    ## E    0    0    0    4 2521 0.0015841584

``` r
#Predicting outcome in validation set     
predrf <- predict(modrf, newdata = validation)
cm <- confusionMatrix(predrf, validation$classe)

accrf <- confusionMatrix(predrf, validation$classe)$overall["Accuracy"]
```

Based on their accuracy on the validation set, the random forest model is the best one out of both models.

``` r
t(data.frame(acctree, accrf))
```

    ##          Accuracy
    ## acctree 0.6618522
    ## accrf   0.9988105

As further analysis of the random forest model, the error rate was plotted against the number of trees and a multiclass ROC curve was calculated.

``` r
#Ploting the model
plot(modrf, main="Random forest model")
legend("right", cex =1, legend=colnames(modrf$err.rate), lty=c(1,2,3), col=c(1,2,3))
```

![](Analysis_Report_files/figure-markdown_github/Random%20Forest%20Model%20-%20error%20vs%20trees-1.png)

``` r
#Calculating ROC curve
roc <- multiclass.roc(as.numeric(validation$classe), as.numeric(predrf))
     
roc
```

    ## 
    ## Call:
    ## multiclass.roc.default(response = as.numeric(validation$classe),     predictor = as.numeric(predrf))
    ## 
    ## Data: as.numeric(predrf) with 5 levels of as.numeric(validation$classe): 1, 2, 3, 4, 5.
    ## Multi-class area under the curve: 0.9997

The area under the curve calculated for the random forest is above 0.80, which means it is a good predictor.

Predictions of the random forest model on the validation set
============================================================

``` r
plot(cm$table, col=cm$byClass, main=paste("Random Forest Model    ", "Accuracy:", round(cm$overall["Accuracy"], 4)))
```

![](Analysis_Report_files/figure-markdown_github/Random%20Forest%20Model%20-%20Accuracy-1.png)

### Predictions on the test set

In order to predict the classes in the 20 observations on the testing set, the same preprocessing as in the training set was carried out: Variables with missing values and/or near zero variance were dismissed.

``` r
testclean <- testing
testclean <- testing[, colSums(is.na(testing)) == 0]
testclean$problem_id <- NULL
     
preot <- preProcess(testclean, method="zv")
pptest <- predict(preot, testclean)
pptest$classe <- as.factor(NA)
```

Finally, as the best predictor, the random forest model was used to predict the class in the 20 observations included in the testing set. All were then plotted.

``` r
predtest <- predict(modrf, newdata = pptest)
     
     
predictions <- data.frame(predtest)
     
predictions
```

    ##    predtest
    ## 1         B
    ## 2         A
    ## 3         B
    ## 4         A
    ## 5         A
    ## 6         E
    ## 7         D
    ## 8         B
    ## 9         A
    ## 10        A
    ## 11        B
    ## 12        C
    ## 13        B
    ## 14        A
    ## 15        E
    ## 16        E
    ## 17        A
    ## 18        B
    ## 19        B
    ## 20        B

``` r
qplot(x=seq_along(1:20), y=predictions$predtest, ylab="Classes", xlab="Problem_ID", colour=predictions$predtest, show.legend = FALSE, size=3, main="Predicted class in test set") + theme(plot.title = element_text(hjust = 0.5))     
```

![](Analysis_Report_files/figure-markdown_github/Predictions%20in%20testing%20set-1.png)
