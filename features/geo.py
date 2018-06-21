from pyspark.sql import Window
from pyspark.sql.functions import col
import pyspark.sql.functions as F

from features.category_based import getFoldCategoriesAndRatings
from features.jaccard import jaccard
from util.dataframe_utils import ren


def jaccard_distance(ratingdf, distancedf, outputcol):
    ratingdf = ratingdf.select("user_id", "business_id")
    users_nr = ratingdf.groupBy("user_id").agg(
        F.count(F.lit(1)).alias("nr")
    ) \
        .alias("users_nr")
    # Usuarios que fueron al primer business
    ratingsJoin = ratingdf.join(distancedf, "business_id")

    ratingsJoin = ratingsJoin.select("user_id", "business_id", "business_id_2").join(ren(ratingdf),
                                                                                     "business_id_2").filter(
        col("user_id") < col("user_id_2")).join(users_nr, "user_id").join(ren(users_nr), "user_id_2")

    intersection = ratingsJoin.groupBy("user_id", "user_id_2", "nr", "nr_2").agg(
        F.count(F.lit(1)).alias("intersection"))
    jaccard_df = intersection.withColumn(outputcol,
                                         (col("intersection") / (col("nr") + col("nr_2") - col("intersection")))
                                         .cast("float"))
    return jaccard_df.select("user_id", "user_id_2", outputcol)


def geo_distance_metrics(ratings, business_data, business_distance, b_categories):
    cat_rating = getFoldCategoriesAndRatings(ratings, business_data, b_categories)

    fold_business_distance = business_distance.join(ratings.select("business_id").distinct(), "business_id",
                                                    "right").join(
        ratings.select(col("business_id").alias("business_id_2")).distinct(),
        "business_id_2", "right")

    jaccard_business_distance = jaccard_distance(ratings, filter_business_distance(fold_business_distance, 0.2),
                                                 "jaccard_business_distance")

    business_with_cat = fold_business_distance.join(b_categories, "business_id") \
        .join(ren(b_categories), "business_id_2")
    business_with_cat = filter_business_distance(business_with_cat, 0.2).filter(col("category") == col("category_2"))
    jaccard_distance_cat = jaccard_distance(ratings, business_with_cat, "jaccard_distance_cat")

    ratings_with_business = cat_rating.join(business_data.select("business_id", "city"), "business_id")
    jaccard_city_cat = jaccard(ratings_with_business, ["category", "city"], "jaccard_city_cat")
    return jaccard_business_distance.join(jaccard_distance_cat, ["user_id", "user_id_2"], "outer").join(
        jaccard_city_cat, ["user_id", "user_id_2"], "outer")


def user_home(ratings, users, business_data, business_distance, filter_best_ratings=False, max_distance=1):
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
        .agg(F.count(F.lit(1)).alias("neighbours"))

    user_home_df = user_home_df.withColumn("rn", F.row_number().over(w)).where(col("rn") == 1).drop("rn")

    venues_next_to_home_business = user_home_df.join(business_with_radius, "business_id").select("user_id", col(
        "business_id_2").alias("business_id"))

    user_home_df = user_home_df.select("user_id", "business_id").union(venues_next_to_home_business).join(
        business_data.select("business_id", "latitude", "longitude"),
        "business_id") \
        .groupBy("user_id") \
        .agg(F.mean("latitude").alias("latitude"),
             F.mean("longitude").alias("longitude"))

    return user_home_df


def filter_business_distance(business_distance, distance):
    return business_distance.filter((col("distance") < distance))
