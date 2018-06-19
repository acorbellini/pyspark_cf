import numpy as np

from pyspark.sql import Window
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.utils import AnalysisException


def has_column(df, col):
    try:
        df[col]
        return True
    except AnalysisException:
        return False


def extract(cols, vector_col):
    return lambda row: tuple([row[c] for c in cols] + row[vector_col].toArray().tolist())


def to_cols(features):
    return [col(f) for f in features]


def colwise_mean(features):
    cols = to_cols(features)
    ret = None
    for c in cols:
        if ret is None:
            ret = c
        else:
            ret = ret + c
    ret = ret / F.lit(len(features))
    return ret


first = F.udf(lambda vector: float(vector[0]), FloatType())
max_udf = F.udf(lambda vector: np.max(vector).item(), FloatType())
min_udf = F.udf(lambda vector: np.min(vector).item(), FloatType())


# def save(df, target_dir, name):
#     df.write.mode("overwrite").parquet(DIR_ROOT + "/" + target_dir + "/" + name)


def ren(df, exclude=[]):
    replacements = {c: c + "_2" for c in df.columns if str(c) not in exclude}
    replacements = [col(c).alias(replacements.get(c)) for c in df.columns if str(c) not in exclude]
    replacements = replacements + exclude
    return df.select(replacements)


# def read(a):
#     return sql_sc.read.parquet(DIR_ROOT + "/yelp/" + a).repartition(5000, "user_id", "user_id_2")
#
#
# def join_datasets(a, b):
#     dfa = sql_sc.read.parquet(DIR_ROOT + "/yelp/" + a)
#     dfb = sql_sc.read.parquet(DIR_ROOT + "/yelp/" + b)
#     return dfa.join(dfb, ["user_id", "user_id_2"])


def get_offset(fold, folds):
    return (((col("rn") - 1) + fold * (col("count") / folds)) % col("count")) + 1


def get_fold(fold, folds):
    return get_offset(fold, folds) * (1 / col("count"))


def train_test_split_randomized(df, fold, folds, split):
    w = Window.partitionBy(col("user_id")).orderBy("random")
    counts = df.groupBy("user_id").agg(F.count("*").alias("count"));

    randomized = df.join(counts, "user_id", "left").orderBy("user_id", "business_id").withColumn("random", F.rand(
        seed=123)).withColumn("rn", F.row_number().over(w))

    randomized.cache()

    randomized = randomized.withColumn("fold", get_fold(fold, folds))

    train = randomized.where(col("fold") <= split).drop("rn", "count", "random", "fold").orderBy("user_id",
                                                                                                 "business_id")
    test = randomized.where(col("fold") > split).drop("rn", "count", "random", "fold").orderBy("user_id",
                                                                                               "business_id")
    return train, test


def convert_column_to_id(df, target, renamed):
    id_table = df.select(target).orderBy(target).distinct().rdd.map(
        lambda x: x[0]).zipWithIndex().toDF(
        [renamed, target]).select(renamed, col(target).cast("integer"))
    ren_df = df.withColumnRenamed(target, renamed).join(id_table, renamed)
    return id_table, ren_df


def filter_dataframe(target, fromdf, column):
    return target.join(fromdf.select(column).distinct(), column, "right")


def replace_ids(df, target_col, from_df, from_col):
    return df.withColumnRenamed(target_col, from_col)\
             .join(from_df, from_col)\
             .drop(from_col)
