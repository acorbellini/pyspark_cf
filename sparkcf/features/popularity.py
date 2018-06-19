from pyspark.sql.functions import col
import pyspark.sql.functions as F

from util.dataframe_utils import ren


def popularity_based_metrics(ratings, tips):
    total_reviews = ratings.groupBy("business_id").agg(F.count(F.lit(1)).alias("total_reviews"))

    all_pairs = ratings.join(ren(ratings, ["business_id"]), "business_id").filter(col("user_id") < col("user_id_2"))

    all_pairs = all_pairs.join(total_reviews, "business_id")

    adamic_ratings = all_pairs.groupBy("user_id", "user_id_2").agg(
        F.sum(1 / F.log("total_reviews")).cast("float").alias("aa_pop_ratings"))

    tips = tips.join(ratings.select("user_id").distinct(), "user_id", "right")

    total_tips = tips.groupBy("business_id").agg(F.count(F.lit(1)).alias("total_tips"))

    all_pairs = ratings.join(ren(ratings, ["business_id"]), "business_id").filter(col("user_id") < col("user_id_2"))

    all_pairs = all_pairs.join(total_tips, "business_id")

    adamic_tips = all_pairs.groupBy("user_id", "user_id_2").agg(
        F.sum(1 / F.log("total_tips")).cast("float").alias("aa_pop_tips"))

    return adamic_ratings.join(adamic_tips, ["user_id", "user_id_2"], "outer")


def usefulness(ratings):
    usefulness_df = ratings.groupBy("user_id").agg(F.mean("useful").alias("avg_usefulness"))
    rating_count = ratings.count()
    usefulness_df = usefulness_df.withColumn("usefulness", col("avg_usefulness") / F.lit(rating_count))
    return usefulness_df.select("user_id", "usefulness")


def liked_tips(ratings, tips):
    tips = tips.join(ratings.select("user_id").distinct(), "user_id", "right")
    tips_df = tips.groupBy("user_id").agg(F.mean("likes").alias("avg_tips"))
    rating_count = tips.count()
    tips_df = tips_df.withColumn("liked_tips", col("avg_tips") / F.lit(rating_count))
    return tips_df.select("user_id", "liked_tips")
