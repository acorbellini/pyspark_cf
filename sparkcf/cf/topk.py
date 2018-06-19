from pyspark.sql import Window
from pyspark.sql.functions import col
import pyspark.sql.functions as f


def select_topk_users(df, k, boost=None):
    if boost is None:
        df = df.withColumn("sort", col("similarity"))
        df = df.withColumn("sort_2", col("similarity"))
    else:
        df = df.withColumn("sort", boost["user_id_2"]("similarity"))
        df = df.withColumn("sort_2", boost["user_id"]("similarity"))

    w = Window.partitionBy(col("user_id")).orderBy(col("sort").desc())

    diff = df.select(col("user_id_2").alias("user_id"), col("user_id").alias("user_id_2")).subtract(
        df.select("user_id", "user_id_2"))
    df_rv = diff.join(df, ["user_id", "user_id_2"], "left").select("user_id", "user_id_2", "similarity",
                                                                   col("sort_2").alias("sort"));
    df = df.select("user_id", "user_id_2", "similarity", "sort").union(df_rv)

    topKperUser = df.select("user_id", "user_id_2", "similarity", "sort").withColumn("rn",
                                                                                     f.row_number().over(w)).where(
        col("rn") <= k).drop("rn", "sort")
    return topKperUser
