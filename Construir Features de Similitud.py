# coding: utf-8

# In[1]:


import math
import os

import pyspark.sql.functions as F
from pyspark import SparkContext, SQLContext, SparkConf
from pyspark.sql import Window
from pyspark.sql.functions import rand, broadcast
from pyspark.sql.functions import sum as sum_sql, mean, count, desc, col, log, sqrt, \
    row_number, lit, size, explode, sin, cos, atan2

# In[2]:


SPARK_IP = "192.168.240.10"
SPARK_URL = "spark://" + SPARK_IP + ":7077"
MIN_RATINGS = 10
TRAIN_TEST_SPLIT = 0.8
DIR_ROOT = "hdfs://" + SPARK_IP
FOLDS = 5
CREATE_FOLDS = False
CREATE_DISTANCES = False

# In[3]:


conf = SparkConf()
conf.set("spark.executor.memory", "5g")
# conf.set("spark.sql.shuffle.partitions", "1000")
# conf.set("spark.yarn.executor.memoryOverhead", "512m")
conf.set("spark.network.timeout", "2000")
conf.set("spark.sql.broadcastTimeout", "300000")
# conf.set("spark.dynamicAllocation.enabled","true")
# conf.set("spark.shuffle.service.enabled", "true")
# conf.set("spark.local.dir", "/yelp-dataset/spark-tmp")
# conf.set("spark.driver.memory","512m")
# conf.set("spark.driver.maxResultSize","10g")
# sc = SparkContext("local[*]", "Simple App", conf=conf)
# sc.setCheckpointDir('/tmp')
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3.6'
conf.setMaster(SPARK_URL)
sc = SparkContext(conf=conf)
sql_sc = SQLContext(sc)


# In[4]:


def save(df, target_dir, name):
    df.write.mode("overwrite").parquet(DIR_ROOT + "/" + target_dir + "/" + name)


def ren(df, exclude=[]):
    replacements = {c: c + "_2" for c in df.columns if str(c) not in exclude}
    replacements = [col(c).alias(replacements.get(c)) for c in df.columns if str(c) not in exclude]
    replacements = replacements + exclude
    return df.select(replacements)


def read(a):
    return sql_sc.read.parquet(DIR_ROOT + "/yelp/" + a).repartition(5000, "user_id", "user_id_2")


def join_datasets(a, b):
    dfa = sql_sc.read.parquet(DIR_ROOT + "/yelp/" + a)
    dfb = sql_sc.read.parquet(DIR_ROOT + "/yelp/" + b)
    return dfa.join(dfb, ["user_id", "user_id_2"])


def get_offset(fold, folds):
    return (((col("rn") - 1) + fold * (col("count") / folds)) % col("count")) + 1


def get_fold(fold, folds):
    return get_offset(fold, folds) * (1 / col("count"))


def train_test_split_randomized(df, fold, folds):
    w = Window.partitionBy(col("user_id")).orderBy("random")
    counts = df.groupBy("user_id").agg(count("*").alias("count"));

    randomized = df.join(counts, "user_id", "left").orderBy("user_id", "business_id").withColumn("random", rand(
        seed=123)).withColumn("rn", row_number().over(w))

    randomized.cache()

    randomized = randomized.withColumn("fold", get_fold(fold, folds))

    train = randomized.where(col("fold") <= TRAIN_TEST_SPLIT).drop("rn", "count", "random", "fold").orderBy("user_id",
                                                                                                            "business_id")
    test = randomized.where(col("fold") > TRAIN_TEST_SPLIT).drop("rn", "count", "random", "fold").orderBy("user_id",
                                                                                                          "business_id")
    return train, test


def deg2rad(deg):
    return deg * (lit(math.pi) / 180)


def km(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in km
    dLat = deg2rad(lat2 - lat1)  # deg2rad below
    dLon = deg2rad(lon2 - lon1)
    a = sin(dLat / 2) * sin(dLat / 2) + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dLon / 2) * sin(dLon / 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    d = R * c  # Distance in km
    return d


# In[5]:


users = sql_sc.read.json(DIR_ROOT + "/yelp-dataset/yelp_academic_dataset_user.json")
business_data = sql_sc.read.json(DIR_ROOT + "/yelp-dataset/yelp_academic_dataset_business.json")
checkin_data = sql_sc.read.json(DIR_ROOT + "/yelp-dataset/yelp_academic_dataset_checkin.json")
tips_data = sql_sc.read.json(DIR_ROOT + "/yelp-dataset/yelp_academic_dataset_tip.json")

user_id = users.select("user_id").orderBy("user_id").distinct().rdd.map(lambda x: x[0]).zipWithIndex().toDF(
    ["user_name", "user_id"]).select("user_name", col("user_id").cast("integer"))

business_id = business_data.select("business_id").orderBy("business_id").distinct().rdd.map(
    lambda x: x[0]).zipWithIndex().toDF(["business_name", "business_id"]).select("business_name",
                                                                                 col("business_id").cast("integer"))

users = users.withColumnRenamed("user_id", "user_name").join(user_id, "user_name")
business_data = business_data.withColumnRenamed("business_id", "business_name").join(business_id, "business_name")
ratings = sql_sc.read.json(DIR_ROOT + "/yelp-dataset/yelp_academic_dataset_review.json")
ratings = ratings.withColumnRenamed("user_id", "user_name").join(user_id, "user_name")
ratings = ratings.withColumnRenamed("business_id", "business_name").join(business_id, "business_name")

if CREATE_DISTANCES:
    lat_and_long = business_data.select("business_id", "longitude", "latitude")
    business_distance = lat_and_long.crossJoin(ren(lat_and_long)).withColumn("distance",
                                                                             km(col("latitude"), col("longitude"),
                                                                                col("latitude_2"),
                                                                                col("longitude_2"))).where(
        col("distance") <= 1).select("business_id", "business_id_2", "distance")
    business_distance.write.mode("overwrite").parquet(DIR_ROOT + "/business_distance/")

business_distance = sql_sc.read.parquet(DIR_ROOT + "/business_distance/")
# print(users.count())
# print(ratings.count())

###########################################################
# Esto se puede usar para filtrar por cantidad de ratings.
###########################################################
users_filtered = ratings.groupBy("user_id").agg(count("business_id").alias("num_ratings")).filter(
    col("num_ratings") >= MIN_RATINGS).select("user_id")

all_ratings = ratings.join(users_filtered, "user_id", "right")
users = users.join(users_filtered, "user_id", "right")
# print(users.count())
# print(ratings.count())


# all_ratings.orderBy("user_id", "business_id").show()

user_friend = users.select("user_id", "friends").filter(
    (size(col("friends")) > 1) | (col("friends").getItem(0) != "None")).select("user_id",
                                                                               explode("friends").alias("friend"),
                                                                               size("friends").alias("nf"))
user_friend = user_friend.withColumnRenamed("friend", "friend_name").join(
    user_id.select(col("user_id").alias("friend_id"),
                   col("user_name").alias("friend_name")),
    "friend_name").select("user_id", "friend_id", "nf")

# *********************************************
# Agregar Connected Components y PageRank PRECALCULADOS
# *********************************************
cc = sql_sc.read.parquet("hdfs://192.168.240.10/yelp-graph/connected_components")
pagerank = sql_sc.read.parquet("hdfs://192.168.240.10/yelp-graph/user_pagerank")

# In[6]:


###########################################################
# Crear Folds Con Ratings Originales
###########################################################
if CREATE_FOLDS:
    for fold in range(5):
        print("Building Fold {}".format(fold))
        ratings, test_ratings = train_test_split_randomized(all_ratings, fold, FOLDS)
        ratings.write.mode("overwrite").parquet(DIR_ROOT + "/fold_" + str(fold) + "/train")
        test_ratings.write.mode("overwrite").parquet(DIR_ROOT + "/fold_" + str(fold) + "/test")
        sql_sc.clearCache()


# # Basados en Rating

# In[7]:


def ratingBasedMetrics(ratings):
    ratings_quad = ratings.select("user_id", "business_id", "stars").withColumn("stars_quad",
                                                                                col("stars") * col("stars")).alias(
        "user_business_rating")
    sum_stars = ratings_quad.groupBy("user_id").agg(
        sum_sql("stars_quad").alias("sum_quad_stars"),
        count(lit(1)).alias("nr")
    ) \
        .alias("user_business_stars_quad")

    ratings_sum = ratings_quad.join(sum_stars, "user_id").select("business_id", "user_id", "stars", "stars_quad",
                                                                 "sum_quad_stars", "nr")

    all_pairs = ratings_sum.join(ren(ratings_sum, ["business_id"]), "business_id").filter(
        col("user_id") < col("user_id_2"))

    cosine_data = all_pairs.groupBy("user_id", "user_id_2", "sum_quad_stars", "sum_quad_stars_2").agg(
        sum_sql("stars").alias("sum_stars"),
        sum_sql("stars_2").alias("sum_stars_2"),
        sum_sql(col("stars") * col("stars_2")).alias("sum_xy"),
        sum_sql((col("stars") - col("stars_2")) * (col("stars") - col("stars_2"))).alias("sumxy_diff_quad"))
    cosine_rating = cosine_data.withColumn("cosine_rating",
                                           ((col("sum_xy")) / (sqrt("sum_quad_stars") * sqrt("sum_quad_stars_2"))).cast(
                                               "float")).select("user_id", "user_id_2", "cosine_rating").filter(
        col("cosine_rating") > 0)

    item_count = ratings.select("business_id").distinct().count()
    item_count_sqrt = math.sqrt(item_count)

    dfDiff = all_pairs.withColumn("diff",
                                  (col("stars") - col("stars_2")) * (col("stars") - col("stars_2"))
                                  - col("stars_quad") - col("stars_quad_2"))

    euclidean = dfDiff.groupBy("user_id", "user_id_2", "sum_quad_stars", "sum_quad_stars_2").agg(
        sum_sql("diff").alias("sum_diff")).withColumn("diff_quad",
                                                      col("sum_diff") + col("sum_quad_stars") + col("sum_quad_stars_2"))

    euclidean_rating = euclidean.withColumn("euclidean_rating",
                                            (1 / (1 + sqrt("diff_quad") / item_count_sqrt)).cast("float")).select(
        "user_id", "user_id_2", "euclidean_rating").filter(col("euclidean_rating") > 0)

    intersection = all_pairs.groupBy("user_id", "user_id_2", "nr", "nr_2").agg(count(lit(1)).alias("intersection"))
    jaccard_rating = intersection.withColumn("jaccard_rating", (
                col("intersection") / (col("nr") + col("nr_2") - col("intersection"))).cast("float")).select("user_id",
                                                                                                             "user_id_2",
                                                                                                             "jaccard_rating").filter(
        col("jaccard_rating") > 0)

    mean_ratings = ratings_quad.groupBy("user_id").agg(
        mean("stars").alias("mean_stars")
    ).alias("mean_ratings")

    centered_stars = ratings_quad.join(mean_ratings, "user_id").withColumn("centered_stars",
                                                                           col("stars") - col("mean_stars")).withColumn(
        "centered_quad_stars", col("centered_stars") * col("centered_stars"))

    centered_stars_sums = centered_stars.groupBy("user_id").agg(sum_sql("centered_stars").alias("sum_centered_stars"),
                                                                sum_sql("centered_quad_stars").alias(
                                                                    "sum_centered_quad_stars")) \
        .alias("centered_stars_sums")

    centered_stars = centered_stars.join(centered_stars_sums, "user_id")
    centered_stars = centered_stars.join(ren(centered_stars, ["business_id"]), "business_id").filter(
        col("user_id") < col("user_id_2"))

    centered_grouped = centered_stars.groupBy("user_id", "user_id_2", "sum_centered_quad_stars",
                                              "sum_centered_quad_stars_2").agg(
        sum_sql(col("centered_stars") * col("centered_stars_2")).alias("sum_xy_centered")) \
        .alias("centered_sum_quad")

    pearson_rating = centered_grouped.withColumn("pearson_rating", (
                (col("sum_xy_centered")) / (sqrt("sum_centered_quad_stars") * sqrt("sum_centered_quad_stars_2"))).cast(
        "float")).select("user_id", "user_id_2", "pearson_rating").filter(col("pearson_rating") > 0)

    return cosine_rating.join(jaccard_rating, ["user_id", "user_id_2"], "outer").join(euclidean_rating,
                                                                                      ["user_id", "user_id_2"],
                                                                                      "outer").join(pearson_rating,
                                                                                                    ["user_id",
                                                                                                     "user_id_2"],
                                                                                                    "outer")


# # Distancia Grafo Social

# In[8]:


def socialBasedMetrics(ratings):
    fold_user_friend = user_friend.join(ratings.select("user_id").distinct(), "user_id", "right")
    fu_with_friendsize = fold_user_friend.join(fold_user_friend.select(col("user_id").alias("friend_id"),
                                                                       col("nf").alias("nf_friend")).distinct(),
                                               "friend_id") \
        .select("user_id", "nf", "friend_id", "nf_friend")

    ufJoin = fu_with_friendsize.join(ren(fu_with_friendsize, ["friend_id"]), "friend_id").filter(
        col("user_id") < col("user_id_2"))

    intersection = ufJoin.groupBy("user_id", "user_id_2", "nf", "nf_2").agg(count(lit(1)).alias("intersection"),
                                                                            sum_sql(1 / log("nf_friend")).cast(
                                                                                "float").alias("adamic_adar_graph"))

    graph = intersection.withColumn("jaccard_graph",
                                    (col("intersection") / (col("nf") + col("nf_2") - col("intersection"))).cast(
                                        "float")).withColumn("cosine_graph", (
                col("intersection") / (sqrt(col("nf") * col("nf_2")))).cast("float")).withColumn(
        "preferential_attachment", col("nf") * col("nf_2")).select("user_id", "user_id_2", "adamic_adar_graph",
                                                                   "jaccard_graph", "cosine_graph",
                                                                   "preferential_attachment").filter(
        (col("adamic_adar_graph") > 0) | (col("jaccard_graph") > 0) | (col("cosine_graph") > 0))

    return graph


# # Intersección de Categorías

# In[9]:


def getFoldCategoriesAndRatings(ratings, business_data):
    b_categories = business_data.select("business_id", explode("categories").alias("category"), "latitude", "longitude")
    category_id = b_categories.select("category").distinct().rdd.map(lambda x: x[0]).zipWithIndex().toDF(
        ["category_name", "category"]).select("category_name", col("category").cast("integer"))

    b_categories = b_categories.withColumnRenamed("category", "category_name").join(category_id,
                                                                                    "category_name").select(
        "business_id", "category")

    # Esto filtra por cantidad de categorias
    w = Window.partitionBy(col("business_id")).orderBy(col("business_id").desc())
    b_categories_first_two = b_categories.withColumn("rn", row_number().over(w)).where(col("rn") <= 3).drop("rn")

    cat_rating = ratings.select("user_id", "business_id", "date", "stars").join(b_categories, "business_id", "right")

    return b_categories, cat_rating


def jaccard_category(ratingdf, filterCol, outputcol):
    users_nr = ratingdf.groupBy("user_id").agg(
        count(lit(1)).alias("nr")
    ) \
        .alias("users_nr")

    week_ratings = ratingdf.select("user_id", "category", filterCol)

    ratingsJoin = week_ratings.join(ren(week_ratings, ["category", filterCol]),
                                    ["category", filterCol]) \
        .filter(col("user_id") < col("user_id_2"))

    intersection = ratingsJoin.groupBy("user_id", "user_id_2").agg(count(lit(1)).alias("intersection"))

    intersection = intersection.join(users_nr, "user_id").join(ren(users_nr), "user_id_2")

    jaccard_df = intersection.withColumn(outputcol,
                                         (col("intersection") / (col("nr") + col("nr_2") - col("intersection"))).cast(
                                             "float"))
    return jaccard_df.select("user_id", "user_id_2", outputcol)


def jaccard_user_reviews(ratingdf, filterCol, outputcol):
    users_nr = ratingdf.groupBy("user_id").agg(
        count(lit(1)).alias("nr")
    ) \
        .alias("users_nr")
    week_ratings = ratingdf.join(users_nr, "user_id").select("user_id", "business_id", filterCol, "nr")

    renamed = ren(week_ratings, ["business_id", filterCol])

    ratingsJoin = week_ratings.join(renamed,
                                    ["business_id", filterCol]) \
        .filter(col("user_id") < col("user_id_2"))

    intersection = ratingsJoin.groupBy("user_id", "user_id_2", "nr", "nr_2").agg(count(lit(1)).alias("intersection"))
    jaccard_df = intersection.withColumn(outputcol,
                                         (col("intersection") / (col("nr") + col("nr_2") - col("intersection"))).cast(
                                             "float"))
    return jaccard_df.select("user_id", "user_id_2", outputcol)


def categoryAndTemporalBasedMetrics(ratings, business_data):
    b_categories, cat_rating = getFoldCategoriesAndRatings(ratings, business_data)

    ucs = cat_rating.select("user_id", "category").distinct()
    # Esto filtra categorias con N apariciones
    categories_appearances = ucs.groupBy("category").agg(count(lit(1)).alias("nc")).orderBy(desc("nc")).filter(
        col("nc") <= 2000).drop("nc")

    cat_rating = cat_rating.join(categories_appearances, "category", "right")

    ucs = ucs.join(categories_appearances, "category", "right")

    ucs_all_pairs = ucs.join(ucs.select("category"
                                        , col("user_id").alias("user_id_2"))
                             , "category") \
        .filter(col("user_id") < col("user_id_2"))
    intersection = ucs_all_pairs.groupBy("user_id", "user_id_2").agg(count(lit(1)).alias("intersection"))
    ucs_grouped = ucs.select("user_id").groupBy("user_id").agg(count(lit(1)).alias("nr"))

    intersection = intersection.join(ucs_grouped, "user_id")
    intersection = intersection.join(ren(ucs_grouped), "user_id_2")

    jaccard_cat = intersection.withColumn("jaccard_cat",
                                          (col("intersection") / (col("nr") + col("nr_2") - col("intersection"))).cast(
                                              "float")).select("user_id", "user_id_2", "jaccard_cat")

    #     ratings_day = ratings.select("user_id", "date", "business_id").withColumn("day", date_format("date", "E"))
    #     ratings_weekend = ratings_day.withColumn("is_weekend", (col("day")=="Sun") | (col("day")=="Sat"))
    #     ratings_week = ratings_day.withColumn("is_week", (col("day")!="Sun") & (col("day")!="Sat"))

    #     jaccard_day = jaccard_user_reviews(ratings_day, "day", "jaccard_day")
    #     jaccard_weekend = jaccard_user_reviews(ratings_weekend, "is_weekend", "jaccard_weekend")
    #     jaccard_week = jaccard_user_reviews(ratings_week, "is_week", "jaccard_week")

    #     week_day = cat_rating.withColumn("day", date_format("date", "E"))

    #     week_day = week_day.withColumn("day_id", when(col("day")=="Sun", 0)\
    #                                             .when(col("day")=="Sat", 1)\
    #                                             .when(col("day")=="Mon", 2)\
    #                                             .when(col("day")=="Tue", 3)\
    #                                             .when(col("day")=="Wed", 4)\
    #                                             .when(col("day")=="Thu", 5)\
    #                                             .when(col("day")=="Fri", 6)\
    #                                             .cast("integer"))

    #     jaccard_cat_day = jaccard_category(week_day, "day_id", "jaccard_cat_day")
    #     jaccard_cat_weekend = jaccard_category(week_day.withColumn("is_weekend", (col("day")=="Sun") | (col("day")=="Sat")), "is_weekend", "jaccard_cat_weekend")
    #     jaccard_cat_week = jaccard_category(week_day.withColumn("is_week", (col("day")!="Sun") & (col("day")!="Sat")), "is_week", "jaccard_cat_week")

    return jaccard_cat


# # Popularidad

# In[10]:


def popularityBasedMetrics(ratings):
    total_reviews = ratings.groupBy("business_id").agg(count(lit(1)).alias("total_reviews"))

    all_pairs = ratings.join(ren(ratings, ["business_id"]), "business_id").filter(col("user_id") < col("user_id_2"))

    all_pairs = all_pairs.join(total_reviews, "business_id")

    adamic_ratings = all_pairs.groupBy("user_id", "user_id_2").agg(
        sum_sql(1 / log("total_reviews")).cast("float").alias("aa_pop_ratings"))

    tips = tips_data.join(ratings.select("user_id").distinct(), "user_id", "right")

    total_tips = tips.groupBy("business_id").agg(count(lit(1)).alias("total_tips"))

    all_pairs = ratings.join(ren(ratings, ["business_id"]), "business_id").filter(col("user_id") < col("user_id_2"))

    all_pairs = all_pairs.join(total_tips, "business_id")

    adamic_tips = all_pairs.groupBy("user_id", "user_id_2").agg(
        sum_sql(1 / log("total_tips")).cast("float").alias("aa_pop_tips"))

    return adamic_ratings.join(adamic_tips, ["user_id", "user_id_2"], "outer")


def usefulness(ratings):
    usefulness = ratings.groupBy("user_id").agg(mean("useful").alias("avg_usefulness"))
    rating_count = ratings.count()
    usefulness = usefulness.withColumn("usefulness", col("avg_usefulness") / lit(rating_count))
    return usefulness.select("user_id", "usefulness")


def liked_tips(ratings):
    tips = tips_data.join(ratings.select("user_id").distinct(), "user_id", "right")
    usefulness = tips.groupBy("user_id").agg(mean("likes").alias("avg_usefulness"))
    rating_count = tips.count()
    usefulness = usefulness.withColumn("liked_tips", col("avg_usefulness") / lit(rating_count))
    return usefulness.select("user_id", "liked_tips")


# # Distancia Geográfica

# In[11]:


def checkList(x):
    if type(x) is list:
        return x
    else:
        return [x]


def jaccard(ratingdf, columns, outputcol):
    reduced_df = ratingdf.select(["user_id"] + checkList(columns)).distinct()

    users_nr = reduced_df.groupBy("user_id").agg(
        count(lit(1)).alias("nr")
    ) \
        .alias("users_nr")

    ratingsJoin = reduced_df.join(ren(reduced_df, checkList(columns)), columns).filter(
        col("user_id") > col("user_id_2"))

    intersection = ratingsJoin.groupBy("user_id", "user_id_2").agg(count(lit(1)).alias("intersection"))

    intersection = intersection.join(users_nr, "user_id").join(ren(users_nr), "user_id_2").filter(
        col("user_id") < col("user_id_2"))

    jaccard_df = intersection.withColumn(outputcol,
                                         (col("intersection") / (col("nr") + col("nr_2") - col("intersection"))).cast(
                                             "float"))
    return jaccard_df.select("user_id", "user_id_2", outputcol)


def filterBusinessDistance(business_distance, distance):
    return business_distance.filter((col("distance") < distance))


def jaccard_distance(ratingdf, distancedf, outputcol):
    ratingdf = ratingdf.select("user_id", "business_id")
    users_nr = ratingdf.groupBy("user_id").agg(
        count(lit(1)).alias("nr")
    ) \
        .alias("users_nr")
    # Usuarios que fueron al primer business
    ratingsJoin = ratingdf.join(distancedf, "business_id")

    ratingsJoin = ratingsJoin.select("user_id", "business_id", "business_id_2").join(ren(ratingdf),
                                                                                     "business_id_2").filter(
        col("user_id") < col("user_id_2")).join(users_nr, "user_id").join(ren(users_nr), "user_id_2")

    intersection = ratingsJoin.groupBy("user_id", "user_id_2", "nr", "nr_2").agg(count(lit(1)).alias("intersection"))
    jaccard_df = intersection.withColumn(outputcol,
                                         (col("intersection") / (col("nr") + col("nr_2") - col("intersection")))
                                         .cast("float"))
    return jaccard_df.select("user_id", "user_id_2", outputcol)


def user_home(ratings, filter_best_ratings=False, max_distance=1):
    w = Window.partitionBy(col("user_id")).orderBy(col("neighbours").desc())

    if filter_best_ratings:
        ratings = ratings.join(users.select("user_id", "average_stars"), "user_id")
        ratings = ratings.filter(col("stars") >= col("average_stars")).drop("average_stars")

    business_with_radius = business_distance.filter(col("distance") <= max_distance)

    user_home_df = ratings.select("user_id", "business_id").join(business_with_radius, "business_id").join(
        ratings.select("user_id", col("business_id").alias("business_id_2")),
        ["user_id", "business_id_2"],
        "right") \
        .groupBy("user_id", "business_id") \
        .agg(count(lit(1)).alias("neighbours"))

    user_home_df = user_home_df.withColumn("rn", row_number().over(w)).where(col("rn") == 1).drop("rn")

    venues_next_to_home_business = user_home_df.join(business_with_radius, "business_id").select("user_id", col(
        "business_id_2").alias("business_id"))

    user_home_df = user_home_df.select("user_id", "business_id").union(venues_next_to_home_business).join(
        business_data.select("business_id", "latitude", "longitude"),
        "business_id") \
        .groupBy("user_id") \
        .agg(mean("latitude").alias("latitude"),
             mean("longitude").alias("longitude"))

    return user_home_df


def geoDistanceMetrics(ratings, business_data):
    b_categories, cat_rating = getFoldCategoriesAndRatings(ratings, business_data)

    fold_business_distance = business_distance.join(ratings.select("business_id").distinct(), "business_id",
                                                    "right").join(
        ratings.select(col("business_id").alias("business_id_2")).distinct(),
        "business_id_2", "right")

    jaccard_business_distance = jaccard_distance(ratings, filterBusinessDistance(fold_business_distance, 0.2),
                                                 "jaccard_business_distance")

    business_with_cat = fold_business_distance.join(b_categories, "business_id").join(ren(b_categories),
                                                                                      "business_id_2")
    business_with_cat = filterBusinessDistance(business_with_cat, 0.2).filter(col("category") == col("category_2"))
    jaccard_distance_cat = jaccard_distance(ratings, business_with_cat, "jaccard_distance_cat")

    ratings_with_business = cat_rating.join(business_data.select("business_id", "city", "state"), "business_id")
    jaccard_city_cat = jaccard(ratings_with_business, ["category", "city"], "jaccard_city_cat")
    return jaccard_business_distance.join(jaccard_distance_cat, ["user_id", "user_id_2"], "outer").join(
        jaccard_city_cat, ["user_id", "user_id_2"], "outer")


# In[12]:


# ratings = sql_sc.read.parquet(DIR_ROOT + "/fold_0/train")

# ratings_test = ratings.filter(col("user_id") == 0)

# user_home(ratings_test, filter_best_ratings=False).orderBy("user_id").show()
# user_home(ratings_test, filter_best_ratings=True).orderBy("user_id").show()


# In[13]:


# ratings.filter(col("user_id")==0).join( users.select("user_id", "average_stars"), "user_id" ).show()


# In[14]:


# save(euclidean_rating, "euclidean_rating")
# save(cosine_rating, "cosine_rating")
# save(jaccard_rating, "jaccard_rating")
# save(pearson_rating, "pearson_rating")
# save(jaccard_cat, "jaccard_cat")
# save(jaccard_day, "jaccard_day")
# save(jaccard_weekend, "jaccard_weekend")
# save(jaccard_week, "jaccard_week")
# save(jaccard_cat_day, "jaccard_cat_day")
# save(jaccard_cat_weekend, "jaccard_cat_weekend")
# save(jaccard_cat_week, "jaccard_cat_week")
# save(jaccard_distance_cat, "jaccard_distance_cat")
# save(jaccard_city_cat, "jaccard_city_cat")
# save(graph, "graph")

# results = [
#             "euclidean_rating", 
#             "cosine_rating",  
#             "jaccard_rating", 
#             "pearson_rating", 
#             "graph",
#             "jaccard_cat", 
#             "jaccard_day", 
#             "jaccard_weekend", 
#             "jaccard_week", 
#             "jaccard_cat_day", 
#             "jaccard_cat_weekend", 
#             "jaccard_cat_week", 
#             "jaccard_distance_cat", 
#             "jaccard_city_cat"
#             ]

# prev = None
# start = None
# cont = 0
# for r in results:
#     if prev is None:
#         if start is not None:
#             prev =  sql_sc.read.parquet(start)
#             prev.printSchema()
#         else:
#             prev = read(r)
#     else:
#         sql_sc.sparkSession.catalog.clearCache()
#         prev = prev.join(read(r), ["user_id", "user_id_2"], "outer")
#         aux_name = "dataset-" + str(cont)
#         prev.write.mode("overwrite").parquet("hdfs://192.168.240.10/yelp-temp/" + aux_name)
#         prev = sql_sc.read.parquet("hdfs://192.168.240.10/yelp-temp/"+ aux_name)
#         prev.printSchema()
#         cont=cont+1

# prev = prev.join(cc, "user_id", "leftouter").join(cc.select(col("user_id").alias("user_id_2")), "user_id_2", "leftouter")
# prev = prev.join(pagerank, "user_id", "leftouter").join(pagerank.select(col("user_id").alias("user_id_2")), "user_id_2", "leftouter")
# prev.printSchema()


# In[16]:


are_friends = user_friend.select("user_id", col("friend_id").alias("user_id_2")).withColumn("are_friends",
                                                                                            lit(True).cast("boolean"))

for fold in [0]:
    print("Building fold " + str(fold))
    ratings = sql_sc.read.parquet(DIR_ROOT + "/fold_" + str(fold) + "/train")
    ratingBased = ratingBasedMetrics(ratings)
    # Filtro los features sociales por los usuarios del fold
    socialBased = socialBasedMetrics(ratings)
    popularityBased = popularityBasedMetrics(ratings)

    user_home_df = user_home(ratings, filter_best_ratings=False)

    userHomeDistance = user_home_df.crossJoin(broadcast(ren(user_home_df))).withColumn("home_distance",
                                                                                       km(col("latitude"),
                                                                                          col("longitude"),
                                                                                          col("latitude_2"),
                                                                                          col("longitude_2"))) \
        .where(col("home_distance") < 1) \
        .select("user_id", "user_id_2", (lit(1) / (lit(1) + col("home_distance"))).alias("home_sim"))

    userHomeDistance.write.mode("overwrite").parquet(DIR_ROOT + "/tmp/fold_" + str(fold) + "_home_distance")
    userHomeDistance = sql_sc.read.parquet(DIR_ROOT + "/tmp/fold_" + str(fold) + "_home_distance")

    fold_business_data = business_data.join(ratings.select("business_id").distinct(), "business_id", "right")
    #     catBased = categoryAndTemporalBasedMetrics(ratings, fold_business_data)
    geoBased = geoDistanceMetrics(ratings, fold_business_data)
    dfs = [ratingBased, socialBased,
           #            catBased,
           geoBased, popularityBased, userHomeDistance]

    current_df = None
    df_count = 0
    for df in dfs:
        if current_df is None:
            current_df = df
        else:
            current_df = current_df.join(df, ["user_id", "user_id_2"], "outer")

        current_df.write.mode("overwrite").parquet(DIR_ROOT + "/tmp/fold_" + str(fold) + "_" + str(df_count))
        current_df = sql_sc.read.parquet(DIR_ROOT + "/tmp/fold_" + str(fold) + "_" + str(df_count))
        df_count = df_count + 1

    all_features = current_df.join(cc, "user_id", "leftouter").join(
        cc.select(col("user_id").alias("user_id_2"), col("cc").alias("cc_2")), "user_id_2", "leftouter").withColumn(
        "cc_sim", F.when(col("cc") == col("cc_2"), 1.0).otherwise(0.0))

    all_features = all_features.join(pagerank, "user_id", "leftouter").join(
        pagerank.select(col("user_id").alias("user_id_2"),
                        col("pagerank").alias("pagerank_2")), "user_id_2", "leftouter")

    usefulness_df = usefulness(ratings)
    all_features = all_features.join(usefulness_df, "user_id", "leftouter").join(
        usefulness_df.select(col("user_id").alias("user_id_2"),
                             col("usefulness").alias("usefulness_2")), "user_id_2", "leftouter")

    liked_tips_df = liked_tips(ratings)
    all_features = all_features.join(liked_tips_df, "user_id", "leftouter").join(
        liked_tips_df.select(col("user_id").alias("user_id_2"),
                             col("liked_tips").alias("liked_tips_2")), "user_id_2", "leftouter")

    #     with_class = all_features.join(are_friends, ["user_id", "user_id_2"], "leftouter")\
    #                      .na.fill({"are_friends": False}).fillna(0)

    all_features.write.mode("overwrite").parquet(DIR_ROOT + "/fold_" + str(fold) + "/train_features")

    all_features.printSchema()

    print("Finished building fold " + str(fold))

# In[17]:


# Sanity Check

train_ratings = sql_sc.read.parquet(DIR_ROOT + "/fold_0/train")
test_ratings = sql_sc.read.parquet(DIR_ROOT + "/fold_0/test")
# train_ratings.select("user_id", "business_id",  "fold").orderBy("user_id", "business_id").show()
# test_ratings.select("user_id", "business_id",  "fold").orderBy("user_id", "business_id").show()
intersect = train_ratings.select("user_id", "business_id").intersect(test_ratings.select("user_id", "business_id"))
train_ratings.select("user_id", "business_id").join(intersect, ["user_id", "business_id"], "right").show()
test_ratings.select("user_id", "business_id").join(intersect, ["user_id", "business_id"], "right").show()

print(intersect.count())
intersect.show()
