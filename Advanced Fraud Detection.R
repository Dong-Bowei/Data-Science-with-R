# Capital One Data Scientist Internship Data Challenge: Advanced Fraud Detection

# Objective:
# Capital One strives to protect customers from fraudulent activities and improve customer trust in its financial services.
# As a Data Scientist, you are tasked with developing a predictive model that accurately identifies fraudulent transactions.
# This challenge emphasizes handling class imbalance, feature engineering, and model evaluation.

# Dataset:
# Use the Fraud Detection Dataset from Kaggle (https://www.kaggle.com/datasets/kartik2112/fraud-detection).
# The dataset contains a mixture of non-fraudulent and fraudulent transactions with features related to customer behavior.

# Challenge:
# You will analyze this dataset and develop a machine learning model to identify fraudulent transactions. 
# The focus is on feature engineering, handling class imbalance, and evaluating the model's performance.

# Steps:

# 1. Data Exploration & Preprocessing (Keep this concise)

# - Load the dataset and provide a brief summary of its structure.

library(data.table)

cc <- fread("/Users/boweidong/Desktop/fraudTest.csv")

summary(cc)

# - Visualize key relationships between features and the target variable `is_fraud`.

ggplot(cc, aes(x = amt, fill = as.factor(is_fraud))) +
  geom_density(alpha = 0.5) +
  scale_x_log10(label = scales::dollar_format()) +
  labs(title = 'Transaction Amount and Fraudulent Activity',
       x = 'Amount',
       y = 'Density') +
  theme_minimal()

ggplot(cc, aes(x = as.Date(trans_date_trans_time), y = amt)) +
  geom_hex(alpha = 0.5) +
  scale_fill_viridis_c() +
  scale_x_date(date_breaks = '1 month', date_labels = '%b %Y') +
  scale_y_log10(label = scales::dollar_format()) +
  labs(title = "time of transaction vs. fraud",
        x = 'Transaction Date',
        y = 'Amount') +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 2. Feature Engineering

# - Create at least 3 new features that might improve model performance. Justify why these features add value.

cc1 <- cc %>%
  mutate(age = round(difftime(as.Date(trans_date_trans_time), as.Date(dob), units = 'weeks')/52.14),
         age = as.numeric(gsub('[^0-9.]', '', age)),
         date = as.Date(trans_date_trans_time),
         time = as.numeric(format(trans_date_trans_time, '%H')),
         time_of_day = case_when(
           time >= 0 & time < 6 ~ "dawn",
           time >=6 & time < 12 ~ "morning",
           time >= 12 & time < 18 ~ "afternoon",
           time >= 18 & time <= 23 ~ "night"))

# - Transform features where necessary (e.g., log transformations for skewed features, interaction terms).

cc2 <- cc1 %>%
  mutate(city_pop = log(city_pop),
         amt_log = log(amt),
         is_fraud = as.factor(is_fraud),
         gender = as.factor(gender))

 # Check the transaction trend on hour of day

gg <- cc2 %>%
  group_by(time) %>%
  summarise(totalobs = n(),
    total = sum(amt)/totalobs)

ggplot(gg, aes(x = time, y = total)) +
  geom_line(color = 'pink') +
  scale_y_log10(label = scales::dollar_format()) +
  labs(title = 'transaction trend on hour of day',
       x = 'Hour of Day',
       y = 'Transaction Amount') +
  theme_minimal()

# 3. Handling Class Imbalance

# - Examine the class distribution between fraudulent and non-fraudulent transactions using a table or summary.

class <- table(cc2$is_fraud)
print(class)

# - Implement at least one technique to handle class imbalance:
#   Option 1: Use oversampling (e.g., SMOTE) to balance the minority class.

library(DMwR)

over <- SMOTE(is_fraud ~ ., data = cc2, perc.over = 200, perc.under = 100)

balanced_data <- ovun.sample(is_fraud ~ ., data = cc2, method = "over", p = 0.2, seed = 123)$data
table(balanced_data$is_fraud)

library(ROSE)

set.seed(123)
both <- ovun.sample(is_fraud ~., data = cc2, method = 'both', p = 0.5, N = nrow(cc2))$data
table(both$is_fraud)

#   Option 2: Apply undersampling on the majority class.
#   Option 3: Use class weights in your model (for example, if using Logistic Regression or Random Forest).
# Example:
# fraud_distribution <- table(card$is_fraud)
# card_balanced <- SMOTE(is_fraud ~ ., data = card, perc.over = 100, perc.under = 200)

# 4. Model Development

# - Split the dataset into training and test sets (e.g., 70% training, 30% testing).

library(caret)

set.seed(123)
train.index <- createDataPartition(balanced_data$is_fraud, p = 0.7, list = FALSE)
train.data <- balanced_data[train.index, ]
test.data <- balanced_data[-train.index, ]

# - Train at least two classification models:
#   Model 1: Logistic Regression or Random Forest.

lg_model <- glm(is_fraud ~ amt_log + gender + city_pop + age + as.factor(time_of_day), data = train.data, family = binomial)
summary(lg_model)

lg_model1 <- glm(is_fraud ~ amt_log + gender + city_pop + age + as.factor(time_of_day), data = test.data, family = binomial)
summary(lg_model1)

both$time_of_day <- as.factor(both$time_of_day)

library(ranger)
rf_model <- ranger(is_fraud ~ amt_log + gender + city_pop + age + time_of_day, data = train.data, num.trees = 100, importance = 'impurity')
print(rf_model)

rf_model1 <- ranger(is_fraud ~ amt_log + gender + city_pop + age + time_of_day, data = test.data, num.trees = 100, importance = 'impurity')
print(rf_model1)

importance(rf_model)

#   Model 2: Advanced model such as XGBoost or LightGBM, which works well for imbalanced datasets.
# - Explain why you chose the models and how they handle imbalanced datasets.

# Load necessary libraries
library(xgboost)
library(dplyr)

# Step 1: Convert categorical variables to numeric
# Assuming 'balanced_data' is your dataset after handling class imbalance (e.g., oversampling, undersampling)
balanced_data_numeric <- balanced_data %>%
  mutate(across(where(is.factor), as.numeric)) %>%  
  mutate(across(where(is.character), as.factor)) %>% 
  mutate(across(where(is.factor), as.numeric)) %>%
  select(16, 23, 24, 26:28) %>%
  mutate(is_fraud = ifelse(is_fraud == 2, 1, 0))

train_index <- createDataPartition(balanced_data_numeric$is_fraud, p = 0.7, list = FALSE)
train.data <- balanced_data_numeric[train_index, ]
test.data <- balanced_data_numeric[-train_index, ]

# Step 2: Prepare train and test data for XGBoost
train_matrix <- xgb.DMatrix(
  data = as.matrix(train.data %>% select(-is_fraud)),  # Exclude target column
  label = train.data$is_fraud                          # Target variable: is_fraud
)
test_matrix <- xgb.DMatrix(
  data = as.matrix(test.data %>% select(-is_fraud)),   # Exclude target column
  label = test.data$is_fraud                           # Target variable: is_fraud
)

# Step 3: Set XGBoost parameters for binary classification
params <- list(
  objective = "binary:logistic",    # Binary classification
  eval_metric = "auc",              # Use AUC as the evaluation metric
  max_depth = 6,                    # Max tree depth (controls complexity)
  eta = 0.3,                        # Learning rate (step size)
  subsample = 0.8,                  # Subsample ratio of training instances
  colsample_bytree = 0.8,           # Subsample ratio of columns when constructing each tree
  scale_pos_weight = sum(train.data$is_fraud == 0) / sum(train.data$is_fraud == 1)  # Balance class weights
)

# Step 4: Train the XGBoost model
xgb_model <- xgb.train(
  params = params,
  data = train_matrix,
  nrounds = 100,                    # Number of boosting rounds
  watchlist = list(train = train_matrix, test = test_matrix),  # Track train and test performance
  early_stopping_rounds = 10,       # Stop early if performance doesn't improve
  verbose = 0                       # Silent output (set to 1 for more details)
)

# Step 5: (Optional) Print model summary or plot importance
xgb.importance(model = xgb_model) %>%
  xgb.plot.importance()


# 5. Model Evaluation

# - Evaluate model performance using the following metrics:
#   Precision
#   Recall
#   F1-score
#   AUC (Area Under the Curve)

predictions <- predict(xgb_model, test_matrix, type = 'response')
predict_class <- ifelse(predictions > 0.5, 1, 0)

cont_matrix <- confusionMatrix(factor(predict_class), factor(test.data$is_fraud))
print(cont_matrix)

precision <- cont_matrix$byClass['Pos Pred Value']
recall <- cont_matrix$byClass['Sensitivity']
f1 <- cont_matrix$byClass['F1']

cat('Precision:', precision)
cat('Recall:', recall)
cat('F1 Score:', f1)

library(pROC)

roc <- roc(test.data$is_fraud, predictions)
auc <- auc(roc)

plot(roc, main = 'ROC')

# - Discuss trade-offs between Precision and Recall. In fraud detection, which metric is more important and why?

# 6. Business Insights & Recommendations

# - Interpret model results and offer actionable insights for Capital One’s fraud detection system.
#   Identify patterns in fraudulent behavior based on feature importance or exploratory data analysis (EDA).
# - Provide recommendations: Could the model be implemented in real-time fraud detection? What challenges might arise (e.g., handling false positives)?
# Example:
# key_features <- importance(model_rf)
# insights <- ggplot(key_features, aes(x = reorder(Feature, Importance), y = Importance)) + geom_bar(stat='identity')

