# Downloading all required libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(dplyr)) install.packages("dplyr")
if(!require(Metrics)) install.packages("Metrics", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes")
if(!require(knitr)) install.packages("knitr")
if(!require(dplyr)) install.packages("dplyr")
if(!require(gbm)) install.packages("gbm")
library(tidyverse)
library(caret)
library(data.table)
library(readr)
library(ggplot2)
library(dplyr)
library(Metrics)
library(penalized)
library(ggthemes)
library(knitr)

# Downloading the dataset
path <- getwd()
filename <- "diamonds_data.csv"
fullpath <- file.path(path, filename)
diamonds <- read.csv(fullpath)
rm(path, filename, fullpath)

# Changing the 'cut' to numerical
diamonds$cut[diamonds$cut == "Fair"] = 0
diamonds$cut[diamonds$cut == "Good"] = 1
diamonds$cut[diamonds$cut == "Very Good"] = 2
diamonds$cut[diamonds$cut == "Premium"] = 3
diamonds$cut[diamonds$cut == "Ideal"] = 4
diamonds$cut = as.numeric(diamonds$cut)

# Changing the 'color' to numerical
diamonds$color[diamonds$color == "J"] = 0
diamonds$color[diamonds$color == "I"] = 1
diamonds$color[diamonds$color == "H"] = 2
diamonds$color[diamonds$color == "G"] = 3
diamonds$color[diamonds$color == "F"] = 4
diamonds$color[diamonds$color == "E"] = 5
diamonds$color[diamonds$color == "D"] = 6
diamonds$color = as.numeric(diamonds$color)

# Changing the 'clarity' to numerical
diamonds$clarity[diamonds$clarity == "I1"] = 0
diamonds$clarity[diamonds$clarity == "SI2"] = 1
diamonds$clarity[diamonds$clarity == "SI1"] = 2
diamonds$clarity[diamonds$clarity == "VS2"] = 3
diamonds$clarity[diamonds$clarity == "VS1"] = 4
diamonds$clarity[diamonds$clarity == "VVS2"] = 5
diamonds$clarity[diamonds$clarity == "VVS1"] = 6
diamonds$clarity[diamonds$clarity == "IF"] = 7
diamonds$clarity = as.numeric(diamonds$clarity)

# Selecting the variables that are going to be used
diamonds <- diamonds %>% select(carat, cut, color, clarity, depth, table, price)

# Splitting the data into train and test sets 
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = diamonds$price, times = 1, p = 0.1, list = FALSE)
train <- diamonds[-test_index,]
test <- diamonds[test_index,]

# Guessing the mean model
m <- mean(train$price)
model1_mape <- 100 * mape(test$price, m)
model1_mape

# Plotting caret vs price
train %>% ggplot(aes(carat, price)) + geom_point()

# Linear model: Carat
model_lm_1 <- lm(price ~ carat, data = train)
predictions_lm_1 <- predict(model_lm_1, test)
model2_mape <- 100 * mape(test$price, predictions_lm_1)

# Plotting cut vs price
train %>% ggplot(aes(cut, price)) + geom_point(alpha = 0.05)

# Linear model: Carat + Cut
model_lm_2 <- lm(price ~ carat + cut, data = train)
predictions_lm_2 <- predict(model_lm_2, test)
model3_mape <- 100 * mape(test$price, predictions_lm_2)

# Plotting color vs price
train %>% ggplot(aes(color, price)) + geom_point(alpha = 0.05)

# Linear model: Carat + Cut + Color
model_lm_3 <- lm(price ~ carat + cut + color, data = train)
predictions_lm_3 <- predict(model_lm_3, test)
model4_mape <- 100 * mape(test$price, predictions_lm_3)

# Linear model: Carat + Cut + Color + Clarity
model_lm_4 <- lm(price ~ carat + cut + color + clarity, data = train)
predictions_lm_4 <- predict(model_lm_4, test)
model5_mape <- 100 * mape(test$price, predictions_lm_4)

# Linear model: Carat + Cut + Color + Clarity + Depth
model_lm_5 <- lm(price ~ carat + cut + color + clarity + depth, data = train)
predictions_lm_5 <- predict(model_lm_5, test)
model6_mape <- 100 * mape(test$price, predictions_lm_5)

# Linear model: Carat + Cut + Color + Clarity + Depth + Table
model_lm_6 <- lm(price ~ carat + cut + color + clarity + depth + table, data = train)
predictions_lm_6 <- predict(model_lm_6, test)
model7_mape <- 100 * mape(test$price, predictions_lm_6)

# KNN model
set.seed(1, sample.kind="Rounding")
ctrl <- trainControl(method="repeatedcv",repeats = 3) 
model_knn <- train(price ~ ., data = train, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
knnPredict <- predict(model_knn, newdata = test)
model8_mape <- 100 * mape(test$price, knnPredict)

# Projection pursuit regression (PPR) model
model_ppr <- train(price ~ ., method = "ppr", data = train)
predictions_ppr <- predict(model_ppr, test)
model9_mape <- 100 * mape(test$price, predictions_ppr)

# Gradient boosting model
model_gbm <- train(price ~ ., method = "gbm", data = train)
predictions_gbm <- predict(model_gbm, test)
model10_mape <- 100 * mape(test$price, predictions_gbm)



