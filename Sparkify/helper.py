# import libraries
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
#from pyspark import SparkFiles
#from pyspark.sql.functions import avg, col, concat, count, desc, \
#asc, explode, lit, min, max, split, stddev, udf, isnan, when, rank, \
#log, sqrt, cbrt, exp, sum

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, \
LogisticRegressionModel, RandomForestClassifier, \
RandomForestClassificationModel, GBTClassifier, \
GBTClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

from pyspark.ml.feature import CountVectorizer, IDF, Normalizer, \
PCA, RegexTokenizer, Tokenizer, StandardScaler, StopWordsRemover, \
StringIndexer, VectorAssembler, MaxAbsScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import time
import tqdm
import re
import numpy as np
import scipy
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import random
%matplotlib inline
random.seed(42)

def create_spark_session():
    '''
    Create a spark session
    INPUT
    spark session builder
    OUTPUT
    spark object
    '''
    spark = SparkSession\
    .builder \
    .master("local") \
    .appName("sparkify") \
    .config("config option", "config value")\
    .getOrCreate()
    print('Spark parameters:')
    for parameter in spark.sparkContext.getConf().getAll():
        print(parameter)

    return spark

def load_dataset(spark, path = 'mini_sparkify_event_data.json'):
    '''
    Data loader
    INPUT
    spark object
    data path
    OUTPUT
    spark dataframe
    '''
    print(f'\nReadinng File from: {path}')
    df = spark.read.json(path)
    print(f'\nColumns in dataset:')
    for column in df.columns:
        print(column)
    print(f'\nDatatypes in dataset: {df.persist()}')
    print(f'\nFirst DataFrame record: {df.head()}')

    return df

def clean_dataset(df):
    '''
    Data Cleaning
    INPUT
    Spark DataFrame
    OUTPUT
    Cleaned Spark DataFrame
    '''
    log_events = df.dropna(how = "any", subset = ["firstName"])
    log_events = log_events.withColumn('song', F.when(log_events.song.isNull(), F.lit("no_song")).otherwise(df.song))
    log_events = log_events.withColumn('artist', F.when(log_events.artist.isNull(), F.lit("no_artist")).otherwise(df.artist))
    log_events = log_events.withColumn('length', F.when(log_events.length.isNull(), F.lit(0)).otherwise(df.length))

    return log_events


def down_selection(page):
    '''
    helper funcction to create churn double selection downgrade
    INPUT
    page column
    OUTPUT
    no output
    '''
    if page == 'Cancellation Confirmation' or page == 'Downgrade':
        return 1
    return 0

def create_churn_col(log_events):
    '''
    create churn column in log_events Spark DataFrame
    INPUT
    log_events DataFrame
    OUTPUT
    log_events DataFrame with churn column
    '''
    flag_downgrade_event = F.udf(down_selection, IntegerType())

    log_events = log_events.withColumn('churn',flag_downgrade_event(df['page']))

    return log_events

def create_phase_col(log_events):
    '''
    create phase column in log_events Spark DataFrame
    INPUT
    log_events DataFrame
    OUTPUT
    log_events DataFrame with phase column
    '''

    # for each user order actions between certain events
    windowval = Window.partitionBy("userId").orderBy(F.desc("ts")).rangeBetween(Window.unboundedPreceding, 0)

    log_events = log_events.withColumn("phase", F.sum("churn").over(windowval))

    return log_events

def get_device_col(log_events):
    '''
    create device column in log_events Spark DataFrame
    INPUT
    log_events DataFrame
    OUTPUT
    log_events DataFrame with device column
    '''
    get_device = F.udf(lambda x: x.split('(')[1].replace(";", " ").split(" ")[0])

    log_events = log_events.withColumn("device", get_device(log_events["userAgent"]))
    return log_events

def create_num_df(log_events):
    '''
    aggregate numeric featuresin the log_events Spark DataFrame
    INPUT
    log_events DataFrame
    OUTPUT
    separate dataframe containing only numeric Features
    '''
    num_df = log_events.select("userId","itemInSession", "phase", "length")\
    .groupBy("userId")\
    .agg(F.count(log_events.userId).cast('int').alias("count_user_logs"),\
        F.max(log_events.itemInSession).cast('int').alias("max_session"),\
        #F.avg(log_events.phase).cast('int').alias("avg_phase"),\
        F.avg(log_events.length).cast('int').alias("avg_length"))

    return num_df

def get_dummy_expression(column):
    '''
    create expressions to separate Categorical Features
    INPUT
    One column of the log_events DataFrame
    OUTPUT
    the expression to be used
    '''
    unique_values = log_events.select(column).distinct().rdd.flatMap(lambda element:element).collect()
    expression = [F.when(F.col(column) == str(el), 1).otherwise(0).alias(column + "_" + str(el)) for el in unique_values]

    return expression

def create_cat_df(log_events, column):
    expression = get_dummy_expression(column)
    cat_df = log_events.select(log_events.userId, *expression)
    cat_df = cat_df.groupBy("userId").avg()

    for colm in cat_df.columns:
        if colm != "userId":
            cat_df = cat_df.withColumn(colm, F.col(colm).cast('int')).withColumnRenamed(colm, colm[4:-1])

    return cat_df

def generate_cats(log_events):

    device_df = create_cat_df(log_events, 'device')

    get_location = F.udf(lambda location: location.split(", ")[-1])
    log_events = log_events.withColumn("location", get_location(log_events['location']))
    locations_df = create_cat_df(log_events, 'location')

    pages_df = create_cat_df(log_events, 'page')
    pages_df = pages_df.drop('page_Cancellation_Confirmation', 'page_Downgrade')

    status_df = create_cat_df(log_events, 'status')
    auth_df = create_cat_df(log_events, 'auth')
    gender_df = create_cat_df(log_events, 'auth')
    level_df = create_cat_df(log_events, 'level')
    method_df = create_cat_df(log_events, 'method')

    return device_df, locations_df, pages_df, status_df, auth_df, gender_df, level_df, method_df

def fix_cat_type(cat_df):
    '''
    fix data type in cat_df

    INPUT
    cat_df Spark DataFrame

    OUTPUT
    cat_df Spark DataFrame
    '''
    for col in cat_df.columns:
        if col != "userID":
            cat_df = cat_df.withColumn(col, F.col(col).cast("int"))

    return cat_df


def create_cat_dict(cat_df):
    keywords = ['device', 'locations', 'page', 'status', 'auth', 'gender', 'level', 'method']
    cat_dict = {}
    for wrd in keywords:
        if wrd not in ['page_Cancellation_Confirmation', 'page_Downgrade']:
            col_names = [col for col in cat_df.columns if wrd in col] + ['userId']
            cat_dict[wrd + '_df'] = col_names

    return cat_dict


def data_for_modeling(log_events, num_df, pages_df, gender_df):
    churn_func = F.udf(lambda x: 0 if x == 0 else 1)
    churn_vals = log_events\
        .select('userId','churn')\
        .groupBy('userId').sum()\
        .withColumnRenamed("sum(churn)", "sum_churn")\
        .withColumn('churn_bin', churn_func('sum_churn'))

    data = num_df\
        .join(pages_df, ["userID"])\
        .join(gender_df, ["userID"])\
        .join(churn_vals.select('userID','churn_bin'), ["userID"])

    # Rename churn column into label
    data = data.withColumn('label', data['churn_bin'].cast('float')).drop('churn_bin') #important to have float type
    data = fix_cat_type(data)
    # Feature columns to be converted into vector
    cols = data.drop('label').drop('userId').columns

    # Train-test split
    train, test = data.drop('userId').randomSplit([0.6, 0.4], seed = 42)

    return data, cols, train, test

def build_pipeline(classifier, paramGrid):
    '''
    Build a cross validation pipeline

    INPUT
    classifier: untrained machine learning classifier
    paramGrid: a grid of parameters to search over

    OUTPUT
    crossval: K-fold cross validator performing model selection by splitting
    the dataset into a set of non-overlapping randomly partitioned folds which
    are used as separate training and test datasets
    e.g., with k=3 folds, K-fold cross validation will generate 3 (training, test)
    dataset pairs, each of which uses 2/3 of the data for training and 1/3 for testing
    '''
    # feature transformer that merges multiple columns into a vector column
    assembler = VectorAssembler(inputCols = cols, outputCol = "features")
    #mascaler = MaxAbsScaler(inputCol = "features", outputCol = "scaled")
    #mmScaler = MinMaxScaler(inputCol = "features", outputCol = "scaled")
    sdscaler = StandardScaler(inputCol = "features", outputCol = "scaled", withStd=True, withMean=True)
    pipeline = Pipeline(stages = [assembler, sdscaler, classifier])

    # Cross validation
    crossval = CrossValidator(
        estimator = pipeline,
        estimatorParamMaps = paramGrid,
        evaluator = MulticlassClassificationEvaluator(metricName='f1'),
        numFolds = 3
    )
    return crossval

def train_model(classifier, train, paramGrid):
    '''
    Train the machine learning model

    INPUT
    classifier: untrained machine learning classifier
    paramGrid: a grid of parameters to search over
    train: Spark DataFrame to use for training

    OUTPUT
    model: trained machine learning model
    training_time: excecution time
    '''
    crossval = build_pipeline(classifier, paramGrid)
    start = time.time()

    model = crossval.fit(train)
    stop = time.time()
    training_time = round((stop - start)/60, 2)
    print(f'Training time: {training_time} minutes')
    return model, training_time

def evaluate_model(model, data):

    start = time.time()
    # make prediction yhat
    yhat = model.transform(data)
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
    stop = time.time()
    return yhat, evaluator

def results_classifier(lr = False, rf = False):

    if lr:
        classifier = LogisticRegression(labelCol="label", featuresCol="scaled")
    elif rf:
        classifier = RandomForestClassifier(labelCol="label", featuresCol="scaled")
    else:
        classifier = GBTClassifier(labelCol="label", featuresCol="scaled")
    paramGrid = ParamGridBuilder().build() # build parameters
    model, training_time = train_model(classifier, train, paramGrid) # train model on train set
    yhat_train, train_evaluator = evaluate_model(model, train) # make predictions from train set
    yhat_test, test_evaluator = evaluate_model(model, test) # make predictions from test set
    train_f1 = train_evaluator.evaluate(yhat_train, {train_evaluator.metricName: "f1"}) # calc f1 from train
    test_f1 = train_evaluator.evaluate(yhat_test, {test_evaluator.metricName: "f1"}) # calc f1 from test
    train_conf_matrix = yhat_train.groupby("label").pivot("prediction").count() # mk confusion matrix for train
    test_conf_matrix = yhat_train.groupby("label").pivot("prediction").count() # mk confusion matrix for test
    best_model = model.bestModel # save the best model of the pipeline

    return train_evaluator, test_evaluator, train_f1, test_f1, train_conf_matrix, test_conf_matrix, best_model
def export_f1(rf_train_f1, rf_test_f1, lr_train_f1, lr_test_f1, gbt_train_f1, gbt_test_f1):
    names = ['rf_train_f1', 'rf_test_f1', 'lr_train_f1', 'lr_test_f1', 'gbt_train_f1', 'gbt_test_f1']
    vals = [rf_train_f1, rf_test_f1, lr_train_f1, lr_test_f1, gbt_train_f1, gbt_test_f1]
    f1_dict = dict(list(zip(names, vals)))
    f1_df = pd.DataFrame.from_dict(f1_dict, orient = 'index', columns = ['f1'])

    return f1_df, f1_df.to_csv('f1.csv', index=False)

def save_models():
        lr_best_model.stages[2].write().overwrite().save('logistic_stages')
        rf_best_model.stages[2].write().overwrite().save('random_forest_stages')
        gbt_best_model.stages[2].write().overwrite().save('gradient_boosted_tree_stages')

def load_models():
    lr_best_model = LogisticRegressionModel.load("logistic")
    rf_best_model = RandomForestClassificationModel.load("random_forest")
    gbt_best_model = GBTClassificationModel.load("gradient_boosted_tree")

    return lr_best_model, rf_best_model, gbt_best_model
def create_coef_df(cols, lr_best_model, rf_best_model, gbt_best_model):

    lr_feature_coef = lr_best_model.coefficients.values.tolist()
    lr_dict = dict(list(zip(cols, lr_feature_coef)))
    lr_df = pd.DataFrame.from_dict(lr_dict, orient = 'index', columns = ['coefficient'])\
    .sort_values('coefficient', ascending=False)

    rf_feature_coef = rf_best_model.stages[2].featureImportances.values.tolist()
    rf_dict = dict(list(zip(cols, rf_feature_coef)))
    rf_df = pd.DataFrame.from_dict(rf_dict, orient = 'index', columns = ['coefficient'])\
    .sort_values('coefficient', ascending=False)

    gbt_feature_coef = gbt_best_model.featureImportances.values.tolist()
    gbt_dict = dict(list(zip(cols, gbt_feature_coef)))
    gbt_df = pd.DataFrame.from_dict(gbt_dict, orient = 'index', columns = ['coefficient'])\
    .sort_values('coefficient', ascending=False)

    return lr_df, rf_df, gbt_df

def plot_coef(df):

    plt.figure(figsize=(30, 15))
    sns.barplot(x = df.index, y = df.coefficient, data = df)
    plt.title('Feature Importance', fontsize = 14)
    plt.xlabel('Features', fontsize = 30)
    plt.ylabel('Feature Importance', fontsize = 14)
    plt.xticks(rotation = 60, fontsize = 20)
    plt.yticks(fontsize = 14)
    plt.show()

start = time.time()
spark = create_spark_session()
stop = time.time()
print(f'\n Creating spark session time: {round((stop - start)/60, 2)} minutes')

start = time.time()
df = load_dataset(spark)
stop = time.time()
print(f'\n Loading dataset time: {round((stop - start)/60, 2)} minutes')

start = time.time()
log_events = clean_dataset(df)
log_events = create_churn_col(log_events)
log_events = create_phase_col(log_events)
log_events = get_device_col(log_events)
stop = time.time()
print(f'\n Cleaning dataset time: {round((stop - start)/60, 2)} minutes')

start = time.time()
num_df = create_num_df(log_events)
device_df = create_cat_df(log_events, 'device')

get_location = F.udf(lambda location: location.split(", ")[-1])
log_events = log_events.withColumn("location", get_location(log_events['location']))
locations_df = create_cat_df(log_events, 'location')

pages_df = create_cat_df(log_events, 'page')
pages_df = pages_df.drop('page_Cancellation_Confirmation', 'page_Downgrade')

status_df = create_cat_df(log_events, 'status')
auth_df = create_cat_df(log_events, 'auth')
gender_df = create_cat_df(log_events, 'auth')
level_df = create_cat_df(log_events, 'level')
method_df = create_cat_df(log_events, 'method')
stop = time.time()
print(f'\n Engineering Features time: {round((stop - start)/60, 2)} minutes')

start = time.time()
data,\
cols,\
train,\
test = data_for_modeling(log_events, num_df, pages_df, gender_df)
stop = time.time()
print(f'\n Joining Features Together: {round((stop - start)/60, 2)} minutes')

rf_train_evaluator, \
rf_test_evaluator, \
rf_train_f1, \
rf_test_f1, \
rf_train_conf_matrix, \
rf_test_conf_matrix, \
rf_best_model = results_classifier(rf = True)

lr_train_evaluator, \
lr_test_evaluator, \
lr_train_f1, \
lr_test_f1, \
lr_train_conf_matrix, \
lr_test_conf_matrix, \
lr_best_model = results_classifier(lr = True)

gbt_train_evaluator, \
gbt_test_evaluator, \
gbt_train_f1, \
gbt_test_f1, \
gbt_train_conf_matrix, \
gbt_test_conf_matrix, \
gbt_best_model = results_classifier()

lr_best_model, rf_bestCLModel, gbt_best_model = load_models()
f1_df, f1_csv = export_f1(rf_train_f1, rf_test_f1, lr_train_f1, lr_test_f1, gbt_train_f1, gbt_test_f1)
lr_df, rf_df, gbt_df = create_coef_df(cols, lr_best_model, rf_best_model, gbt_best_model)
plot_coef(gbt_df)
plot_coef(rf_df)
plot_coef(lr_df)
