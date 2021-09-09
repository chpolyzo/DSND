
https://github.com/elifinspace/sparkify/blob/master/Sparkify_final_.ipynb
class helper():

def __init__(self, path):
    self.path = path


def create_spark_session():
    spark = SparkSession\
    .builder \
    .master("local") \
    .appName("sparkify") \
    .config("config option", "config value") \
    .getOrCreate()
    print('Spark parameters:')
    [print(parameter) for parameter in spark.sparkContext.getConf().getAll()]

    return spark

def load_dataset(spark, path = 'mini_sparkify_event_data.json'):
    print(f'Readinng File: {path}')
    df = spark.read.json(path)
    print(f'Columns in dataset: {df.columns}')
    print(f'Datatypes in dataset: {df.persist()}')
    print(f'First DataFrame record: {df.head()}')

    return df

def clean_dataset(df):
    log_events = df.dropna(how = "any", subset = ["firstName"])
    log_events = log_events.withColumn('song', F.when(log_events.song.isNull(), F.lit("no_song")).otherwise(df.song))
    log_events = log_events.withColumn('artist', F.when(log_events.artist.isNull(), F.lit("no_artist")).otherwise(df.artist))
    log_events = log_events.withColumn('length', F.when(log_events.length.isNull(), F.lit(0)).otherwise(df.length))

    return log_events


def down_selection(page):
    if page == 'Cancellation Confirmation' or page == 'Downgrade':
        return 1
    return 0

def create_churn_col(log_events):
    flag_downgrade_event = F.udf(down_selection, IntegerType())

    log_events = log_events.withColumn('churn',flag_downgrade_event(df['page']))

    return log_events

def create_phase_col(log_events):

    # for each user order actions by between certain events
    windowval = Window.partitionBy("userId").orderBy(F.desc("ts")).rangeBetween(Window.unboundedPreceding, 0)

    log_events = log_events.withColumn("phase", F.sum("churn").over(windowval))

    return log_events

def get_device_col(log_events):
    get_device = F.udf(lambda x: x.split('(')[1].replace(";", " ").split(" ")[0])

    log_events = log_events.withColumn("device", get_device(log_events["userAgent"]))
    return log_events

def create_num_df(log_events):
    num_df = log_events.select("userId","itemInSession", "phase", "length")\
    .groupBy("userId")\
    .agg(F.count(log_events.userId).alias("count_user_logs"),\
        F.max(log_events.itemInSession).alias("max_session"),\
        F.avg(log_events.phase).alias("avg_phase"),\
        F.avg(log_events.length).alias("avg_length"))

    return num_df

def get_dummy_expression(column):
    unique_values = log_events.select(column).distinct().rdd.flatMap(lambda element:element).collect()
    expression = [F.when(F.col(column) == str(el), 1).otherwise(0).alias(column + "_" + str(el)) for el in unique_values]

    return expression

def rename_cols(cat_df):

    cat_df = cat_df\
    .withColumnRenamed('location_Minneapolis-St. Paul-Bloomington, MN-WI','location_Minneapolis-St Paul-Bloomington, MN-WI')\
    .withColumnRenamed('location_St. Louis, MO-IL','location_St Louis, MO-IL')\
    .withColumnRenamed('location_Port St. Lucie, FL','location_Port St Lucie, FL')\
    .withColumnRenamed('location_Tampa-St. Petersburg-Clearwater, FL','location_Tampa-St Petersburg-Clearwater, FL')\
    .withColumnRenamed('page_Cancellation Confirmation', 'page_Cancellation_Confirmation')

    return cat_df

def binarize(df, df_list):
    df = df.select(df_list).groupBy('userID').avg()
    get_bins = F.udf(lambda x: 0 if x == 0 else 1)
    for col in df.columns:
        df = df.withColumn(col, get_bins(df[col]))
    return df

def separate_cat_dfs(cat_df):
    devices = [col for col in cat_df.columns if 'device' in col] + ['userId']
    locations = [col for col in cat_df.columns if 'location' in col] + ['userId']
    pages = [col for col in cat_df.columns if 'page' in col] + ['userId']
    pages = [col for col in pages if col not in ['page_Cancellation_Confirmation', 'page_Downgrade']]
    status = [col for col in cat_df.columns if 'status' in col] + ['userId']
    auth = [col for col in cat_df.columns if 'auth' in col] + ['userId']
    gender = [col for col in cat_df.columns if 'gender' in col] + ['userId']
    level = [col for col in cat_df.columns if 'level' in col] + ['userId']
    method = [col for col in cat_df.columns if 'method' in col] + ['userId']

    device_df = binarize(cat_df, devices)
    for col in device_df.columns:
        if col != "userID":
            device_df = device_df.withColumnRenamed(col,col[4:-1])
    locations_df = binarize(cat_df, locations)
    for col in locations_df.columns:
        if col != "userID":
            locations_df = locations_df.withColumnRenamed(col,col[4:-1])
    pages_df = binarize(cat_df, pages)
    for col in pages_df.columns:
        if col != "userID":
            pages_df = pages_df.withColumnRenamed(col,col[4:-1])
    status_df = binarize(cat_df, status)
    for col in status_df.columns:
        if col != "userID":
            status_df = status_df.withColumnRenamed(col,col[4:-1])
    auth_df = binarize(cat_df, auth)
    for col in auth_df.columns:
        if col != "userID":
            auth_df = auth_df.withColumnRenamed(col,col[4:-1])
    gender_df = binarize(cat_df, gender)
    for col in gender_df.columns:
        if col != "userID":
            gender_df = gender_df.withColumnRenamed(col,col[4:-1])
    level_df = binarize(cat_df, level)
    for col in level_df.columns:
        if col != "userID":
            level_df = level_df.withColumnRenamed(col,col[4:-1])
    method_df = binarize(cat_df, method)
    for col in method_df.columns:
        if col != "userID":
            method_df = method_df.withColumnRenamed(col,col[4:-1])

    return device_df, locations_df, pages_df, status_df, auth_df, gender_df, level_df, method_df








spark = create_spark_session()
df = load_dataset(spark)
log_events = clean_dataset(df)
log_events = create_churn_col(log_events)
log_events = create_phase_col(log_events)
log_events = get_device_col(log_events)


num_df = create_num_df(log_events)
cat_df = create_cat_df(log_events)
cat_df = rename_cols(cat_df)

device_df, locations_df, pages_df, status_df, auth_df, gender_df, level_df, method_df = separate_cat_dfs(cat_df)





page_df = cat_df.select('userId','page_Cancel',
 'page_Submit Downgrade',
 'page_Thumbs Down',
 'page_Home',
 'page_Downgrade',
 'page_Roll Advert',
 'page_Logout',
 'page_Save Settings',
 'page_Cancellation Confirmation',
 'page_About',
 'page_Settings',
 'page_Add to Playlist',
 'page_Add Friend',
 'page_NextSong',
 'page_Thumbs Up',
 'page_Help',
 'page_Upgrade',
 'page_Error',
 'page_Submit Upgrade')\
.groupBy("userId").agg({'page_Cancel':'sum',
 'page_Submit Downgrade':'sum',
 'page_Thumbs Down':'sum',
 'page_Home':'sum',
 'page_Downgrade':'sum',
 'page_Roll Advert':'sum',
 'page_Logout':'sum',
 'page_Save Settings':'sum',
 'page_Cancellation Confirmation':'sum',
 'page_About':'sum',
 'page_Settings':'sum',
 'page_Add to Playlist':'sum',
 'page_Add Friend':'sum',
 'page_NextSong':'sum',
 'page_Thumbs Up':'sum',
 'page_Help':'sum',
 'page_Upgrade':'sum',
 'page_Error':'sum',
 'page_Submit Upgrade':'sum'})

 status_df = cat_df.select('userId',
    'status_307',
    'status_404',
    'status_200')\
    .groupBy("userId")\
    .agg({'status_307':'sum',
      'status_404':'sum',
      'status_200':'sum'})

auth_expression = get_dummy_expression("auth")
gender_expression = get_dummy_expression("gender")
level_expression = get_dummy_expression("level")
method_expression = get_dummy_expression("method")

bin_df = log_events.select("userId", *auth_expression + gender_expression + level_expression + method_expression)

auth_df = bin_df.select('userId',
        'auth_Cancelled',
        'auth_Logged In')\
        .groupBy('userId')\
        .agg({'auth_Cancelled': 'sum',
             'auth_Logged In': 'sum'})\
        .orderBy('sum(auth_Cancelled)', ascending = False)

auth_df = auth_df.withColumnRenamed("sum(auth_Cancelled)","sum_auth_canc")\
                 .withColumnRenamed("sum(auth_Logged In)","sum_auth_log").printSchema()

level_df = bin_df.select('userId',
        'level_free',
        'level_paid')\
        .groupBy('userId')\
        .agg({'level_free': 'sum',
             'level_paid': 'sum'})

level_df = level_df.withColumnRenamed("sum(level_paid)","sum_level_paid")\
                 .withColumnRenamed("sum(level_free)","sum_level_free").printSchema()

method_df = bin_df.select('userId',
        'method_PUT',
        'method_GET')\
        .groupBy('userId')\
        .agg({'method_PUT': 'sum',
             'method_GET': 'sum'})

method_df = method_df.withColumnRenamed("sum(method_GET)","sum_method_GET)")\
                 .withColumnRenamed("sum(method_PUT)","sum_method_PUT").printSchema()
