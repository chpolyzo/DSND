
Check out the blog at: https://chpolyzo.medium.com/the-power-of-spark-in-churn-prediction-8a83eeb65190

# The Power of spark in churn prediction

In this udacity project we use PySpark to predict churn based on a 12GB dataset of a fictitious music service platform, "Sparkify". 

## 1. Motivation

Streaming services have to predict churn rate successfully, not only for developing profitable products and services for their customers but also for making everyday lives easier. We understand that the structured use of data drives digital transformation and improves productivity. However, data is still used more for manipulation rather than effectively provide better decisions.

## 2. Datasets

User activity dataset from Udacity
The dataset logs user demographic information (e.g. user name, gender, location State) and activity (e.g. song listened, event type, device used) at individual timestamps.
A small subset (~120MB) of the full dataset was used for the entire analysis due to budjet limitations to run the models in a cluster

## 3. Data Exploration

- Data loading
- Assess missing values
- Exploratory data analysis
- Overview of numerical columns: descriptive statistics
- Overview of non-numerical columns: possible categories
- Define churn as cancellation of service
- Compare behavior of churn vs. non-churn users in terms of:
1. Session length
2. Numbers of sessions
3. Gender
4. Event types (e.g. add a friend, advertisement, thumbs up)
- Feature engineering for machine learning
- Split training and testing sets
- Choose evaluation metrics
- Create functions to build cross validation pipeline, train machine learning model, and evaluate model performance
- Initial model evaluation with:
1. Logistic regression
2. Random forest (documentation)
3. Gradient-boosted tree (documentation)
- Tune hyperparameters of gradient-boosted tree
- Evaluate model performance
- Evaluate feature importance

## 4. Results

Model performance on testing set:

Testing logloss score: 0.255286996

Testing F1 score: 0.926573

![Other Metrics](https://github.com/chpolyzo/DSND/blob/master/Sparkify/visuals/metrics.png)
