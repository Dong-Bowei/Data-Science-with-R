# Capital One Data Scientist Internship Data Challenge

# Objective:
# Capital One is committed to enhancing customer experiences by providing tailored financial products. 
# As a Data Scientist, you’ll work with large datasets to uncover insights that help improve customer satisfaction, 
# engagement, and retention. In this challenge, you’ll be tasked with analyzing a dataset related to financial transactions.

# Dataset:
# You will use the **Fraud Detection** dataset from Kaggle, which contains transaction details with 
# features like customer location, transaction amount, and whether the transaction was fraudulent.

# Data source:
# Fraud Detection Dataset - Kaggle
# https://www.kaggle.com/datasets/kartik2112/fraud-detection

# Challenge:
# Your task is to analyze this dataset and develop a machine learning model that can effectively identify fraudulent transactions. 
# You should also provide insights into the data and the model's performance.

# Steps:

# 1. Data Exploration & Preprocessing:
# Goal: Perform an exploratory data analysis (EDA) to understand the structure of the data and identify any potential issues.

# - Load the dataset into R.

library(tidyverse)
library(data.table)

card <- fread("/Users/boweidong/Desktop/fraudTest.csv") 

# - Visualize and summarize the dataset.

summary(card)

ggplot(card, aes(x = amt, fill = as.factor(is_fraud))) +
  geom_density(alpha = 0.5) +
  labs(title = 'Fraud and Non-Fraud Statistics',
       x = 'Transaction Amount',
       y = 'Density') +
  scale_x_log10(label = scales::dollar_format()) +
  theme_minimal()

ggplot(card, aes(x = amt, fill = as.factor(gender))) +
  geom_density(alpha = 0.5) +
  scale_x_log10(label = scales::dollar_format()) +
  labs(title = 'Amount by Gender',
       x = 'Amount',
       y = 'Density') +
  theme_minimal()

# - Analyze class imbalance between fraud and non-fraud transactions.

isfraud <- table(card$is_fraud)
print(isfraud)

# - Handle missing values, if applicable.

colSums(is.na(card))
card <- card %>%
  drop_na()
#No missing data, not applicable

# - Scale or normalize features, if necessary.

card1 <- card %>%
  mutate(trans_date_trans_time = as.Date(trans_date_trans_time),
         amount = log(amt),
         gender = as.factor(gender),
         pop = log(city_pop),
         dob = as.Date(dob),
         is_fraud = as.factor(is_fraud))

# 2. Feature Engineering:
# Goal: Identify and create relevant features that may help in predicting fraud.

# - Analyze patterns or correlations between variables (e.g., time of transaction, amount).

ggplot(card1, aes(x = trans_date_trans_time, y = amt)) +
  geom_point(alpha = 0.5, color = 'pink') +
  geom_smooth(method = lm, color = 'blue') +
  scale_y_log10(label = scales::dollar_format()) +
  labs(title = 'Time of Transaction vs Transaction Amount',
       x = 'Date',
       y = 'Amount') +
  scale_x_date(date_labels = "%b %Y")+
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1))

#the pattern is constant across month/year

#Create a heat map instead

ggplot(card1, aes(x = trans_date_trans_time, y = amt)) +
  geom_hex(alpha = 0.7) +
  scale_fill_viridis_c() +
  scale_y_log10(label = scales::dollar_format()) +
  scale_x_date(date_breaks = '1 month', date_labels = '%b %Y') +
  labs(title = 'Time of Transaction vs Transaction Amount',
       x = 'Date',
       y = 'Amount') +
  theme_minimal() +
  theme(axis.text.x = element_text(hjust = 1, angle = 45))

# - Generate any useful new features that could potentially boost the model’s performance.

card2 <- card1 %>%
  mutate(age = round(difftime('2024-10-09', dob, units = 'weeks')/52.14),
         age = as.numeric(gsub('[^0-9.]', '', age)),
         location = paste0('(', merch_lat, ',', merch_long, ')'))

# 3. Model Development:
# Goal: Build and train a machine learning model to classify transactions as fraudulent or not.

# - Split the dataset into training and test sets.

library(caret)

train_index <- createDataPartition(card2$is_fraud, p = 0.7, list = FALSE)
train_data <- card2[train_index, ]
test_data <- card2[-train_index, ]

# - Choose a classification model suitable for dealing with imbalanced data (e.g., Logistic Regression, Random Forest, XGBoost).

model <- glm(is_fraud ~ gender + age + amount + pop, data = card2, family = binomial)
summary(model)

model.train <- glm(is_fraud ~ gender + age + amount + pop, data = train_data, family = binomial)
summary(model.train)

predictions <- predict(model.train, newdata = test_data)
print(predictions)

# - Use resampling techniques to handle class imbalance (e.g., oversampling fraud transactions or undersampling non-fraud ones).

table <- table(train_data$is_fraud)
print(table)

install.packages('ROSE')
library(ROSE)

#OverSampling

oversampled <- ovun.sample(is_fraud ~ ., data = train_data, method = 'over', N = nrow(train_data))$data
table.over <- table(oversampled$is_fraud)
print(table.over)

#UnderSampling

train_data$is_fraud <-as.numeric(train_data$is_fraud)

undersampled <- ovun.sample(is_fraud ~., data = train_data, method = 'under', N = nrow(train_data))$data
table.under <- table(undersampled$is_fraud)
print(table.under)

#Balanced

balanced <- ovun.sample(is_fraud ~., data = train_data, method = 'both', p = 0.5, N = nrow(train_data))$data

table.both <- table(balanced$is_fraud)
print(table.both)

# 4. Model Evaluation:
# Goal: Evaluate the performance of your model.

# - Compute evaluation metrics such as:
#   - Precision
#   - Recall
#   - F1-score
#   - Area Under the Curve (AUC)
# - Discuss the performance and potential trade-offs between false positives and false negatives.

library(pROC)
library(caret)

balanced$is_fraud <- as.factor(balanced$is_fraud)

model.train_balanced <- glm(is_fraud ~ gender + age + amount + pop, data = balanced, family = binomial)

prediction_prob <- predict(model.train_balanced, newdata = test_data, type = 'response')
prediction_class <- ifelse(prediction_prob > 0.5, 1, 0)

conf_matrix <- confusionMatrix(factor(prediction_class), factor(test_data$is_fraud))
print(conf_matrix)

precision <- conf_matrix$byClass['Pos Pred Value']
print(precision)

recall <- conf_matrix$byClass['Sensitivity']
print(recall)

f1 <- conf_matrix$byClass['F1']
print(f1)

# AUC

roc <- roc(test_data$is_fraud, prediction_prob)
auc_value <- auc(roc)

cat("AUC is", auc_value)

plot(roc, main = 'ROC Curve')

