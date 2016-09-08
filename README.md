# Titanic_Passenger_Survival_Prediction
This predicts and analyzes Titanic passenger survival rate with random forest, decision tree, and logistic regression.  
The workflow of this data mining project inludes:
- Preliminary analysis with R, including Apriori algorithm, correlation analysis and data visualization(using ggplot2) --- refer to `Apriori.R`
-	Data preprocessing and feature engineering with python
-	Random Forest modeling with python (to get an out-of-bag error of 0.162) --- refer to `RandomForest.py` and script with out-of-sample test `RandomForest_with_test.py`, `RandomForest_with_test_dummy.py`
- Decision Tree modeling with R to get the "survival tree" --- refer to `Decision_tree.R`
- Logistic Regression modeling with R to explore factors influencing the survival rate (Feature importance ranking by Random Forest is used as a feature selecting tool to choose important features for Logistic Regression)

## Data
You can download the dataset [here](https://www.kaggle.com/c/titanic/data) on [Kaggle](https://www.kaggle.com/).
