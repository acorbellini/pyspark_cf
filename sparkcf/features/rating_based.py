import math

from pyspark.sql.functions import col
import pyspark.sql.functions as f

from util.dataframe_utils import ren


def rating_based_metrics(ratings):
    ratings_quad = ratings.select("user_id", "business_id", "stars").withColumn("stars_quad",
                                                                                col("stars") * col("stars")).alias(
        "user_business_rating")
    sum_stars = ratings_quad.groupBy("user_id").agg(
        f.sum("stars_quad").alias("sum_quad_stars"),
        f.count(f.lit(1)).alias("nr")
    ) \
        .alias("user_business_stars_quad")

    ratings_sum = ratings_quad.join(sum_stars, "user_id").select("business_id", "user_id", "stars", "stars_quad",
                                                                 "sum_quad_stars", "nr")

    all_pairs = ratings_sum.join(ren(ratings_sum, ["business_id"]), "business_id").filter(
        col("user_id") < col("user_id_2"))

    cosine_data = all_pairs.groupBy("user_id", "user_id_2", "sum_quad_stars", "sum_quad_stars_2").agg(
        f.sum("stars").alias("sum_stars"),
        f.sum("stars_2").alias("sum_stars_2"),
        f.sum(col("stars") * col("stars_2")).alias("sum_xy"),
        f.sum((col("stars") - col("stars_2")) * (col("stars") - col("stars_2"))).alias("sumxy_diff_quad"))
    cosine_rating = cosine_data.withColumn("cosine_rating",
                                           ((col("sum_xy")) / (
                                                   f.sqrt("sum_quad_stars") * f.sqrt("sum_quad_stars_2"))).cast(
                                               "float")).select("user_id", "user_id_2", "cosine_rating").filter(
        col("cosine_rating") > 0)

    item_count = ratings.select("business_id").distinct().count()
    item_count_sqrt = math.sqrt(item_count)

    dfDiff = all_pairs.withColumn("diff",
                                  (col("stars") - col("stars_2")) * (col("stars") - col("stars_2"))
                                  - col("stars_quad") - col("stars_quad_2"))

    euclidean = dfDiff.groupBy("user_id", "user_id_2", "sum_quad_stars", "sum_quad_stars_2").agg(
        f.sum("diff").alias("sum_diff")).withColumn("diff_quad",
                                                    col("sum_diff") + col("sum_quad_stars") + col("sum_quad_stars_2"))

    euclidean_rating = euclidean.withColumn("euclidean_rating",
                                            (1 / (1 + f.sqrt("diff_quad") / item_count_sqrt)).cast("float")).select(
        "user_id", "user_id_2", "euclidean_rating").filter(col("euclidean_rating") > 0)

    intersection = all_pairs.groupBy("user_id", "user_id_2", "nr", "nr_2").agg(f.count(f.lit(1)).alias("intersection"))
    jaccard_rating = intersection.withColumn("jaccard_rating", (
            col("intersection") / (col("nr") + col("nr_2") - col("intersection"))).cast("float")) \
        .select("user_id",
                "user_id_2",
                "jaccard_rating").filter(
        col("jaccard_rating") > 0)

    mean_ratings = ratings_quad.groupBy("user_id").agg(
        f.mean("stars").alias("mean_stars")
    ).alias("mean_ratings")

    centered_stars = ratings_quad.join(mean_ratings, "user_id").withColumn("centered_stars",
                                                                           col("stars") - col("mean_stars")).withColumn(
        "centered_quad_stars", col("centered_stars") * col("centered_stars"))

    centered_stars_sums = centered_stars.groupBy("user_id").agg(f.sum("centered_stars").alias("sum_centered_stars"),
                                                                f.sum("centered_quad_stars").alias(
                                                                    "sum_centered_quad_stars")) \
        .alias("centered_stars_sums")

    centered_stars = centered_stars.join(centered_stars_sums, "user_id")
    centered_stars = centered_stars.join(ren(centered_stars, ["business_id"]), "business_id").filter(
        col("user_id") < col("user_id_2"))

    centered_grouped = centered_stars.groupBy("user_id", "user_id_2", "sum_centered_quad_stars",
                                              "sum_centered_quad_stars_2").agg(
        f.sum(col("centered_stars") * col("centered_stars_2")).alias("sum_xy_centered")) \
        .alias("centered_sum_quad")

    pearson_rating = centered_grouped.withColumn("pearson_rating", (
            (col("sum_xy_centered")) / (f.sqrt("sum_centered_quad_stars") * f.sqrt("sum_centered_quad_stars_2"))).cast(
        "float")).select("user_id", "user_id_2", "pearson_rating").filter(col("pearson_rating") > 0)

    return cosine_rating.join(jaccard_rating, ["user_id", "user_id_2"], "outer").join(euclidean_rating,
                                                                                      ["user_id", "user_id_2"],
                                                                                      "outer").join(pearson_rating,
                                                                                                    ["user_id",
                                                                                                     "user_id_2"],
                                                                                                    "outer")
