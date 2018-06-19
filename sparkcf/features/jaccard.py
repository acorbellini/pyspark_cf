from util.dataframe_utils import ren
from util.utils import checkList
from pyspark.sql.functions import col
import pyspark.sql.functions as F


def jaccard_user_reviews(ratingdf, filterCol, outputcol):
    users_nr = ratingdf.groupBy("user_id").agg(
        F.count(F.lit(1)).alias("nr")
    ) \
        .alias("users_nr")
    week_ratings = ratingdf.join(users_nr, "user_id").select("user_id", "business_id", filterCol, "nr")

    renamed = ren(week_ratings, ["business_id", filterCol])

    ratings_join = week_ratings.join(renamed,
                                    ["business_id", filterCol]) \
        .filter(col("user_id") < col("user_id_2"))

    intersection = ratings_join.groupBy("user_id", "user_id_2", "nr", "nr_2").agg(
        F.count(F.lit(1)).alias("intersection"))
    jaccard_df = intersection.withColumn(outputcol,
                                         (col("intersection") / (col("nr") + col("nr_2") - col("intersection"))).cast(
                                             "float"))
    return jaccard_df.select("user_id", "user_id_2", outputcol)


def jaccard(ratingdf, columns, outputcol):
    reduced_df = ratingdf.select(["user_id"] + checkList(columns)).distinct()

    users_nr = reduced_df.groupBy("user_id").agg(
        F.count(F.lit(1)).alias("nr")
    ) \
        .alias("users_nr")

    ratings_join = reduced_df.join(ren(reduced_df, checkList(columns)), columns).filter(
        col("user_id") > col("user_id_2"))

    intersection = ratings_join.groupBy("user_id", "user_id_2").agg(F.count(F.lit(1)).alias("intersection"))

    intersection = intersection.join(users_nr, "user_id").join(ren(users_nr), "user_id_2").filter(
        col("user_id") < col("user_id_2"))

    jaccard_df = intersection.withColumn(outputcol,
                                         (col("intersection") / (col("nr") + col("nr_2") - col("intersection"))).cast(
                                             "float"))
    return jaccard_df.select("user_id", "user_id_2", outputcol)
