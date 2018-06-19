import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col
from util.dataframe_utils import ren


def getFoldCategoriesAndRatings(ratings, business_data, b_categories):
    # Esto filtra por cantidad de categorias
    w = Window.partitionBy(col("business_id")).orderBy(col("business_id").desc())
    b_categories_first_two = b_categories.withColumn("rn", F.row_number().over(w)).where(col("rn") <= 3).drop("rn")
    return ratings.select("user_id", "business_id", "stars").join(b_categories, "business_id", "right")


def jaccard_category(ratingdf, filterCol, outputcol):
    users_nr = ratingdf.groupBy("user_id").agg(
        F.count(F.lit(1)).alias("nr")
    ) \
        .alias("users_nr")

    week_ratings = ratingdf.select("user_id", "category", filterCol)

    ratingsJoin = week_ratings.join(ren(week_ratings, ["category", filterCol]),
                                    ["category", filterCol]) \
        .filter(col("user_id") < col("user_id_2"))

    intersection = ratingsJoin.groupBy("user_id", "user_id_2").agg(F.count(F.lit(1)).alias("intersection"))

    intersection = intersection.join(users_nr, "user_id").join(ren(users_nr), "user_id_2")

    jaccard_df = intersection.withColumn(outputcol,
                                         (col("intersection") / (col("nr") + col("nr_2") - col("intersection"))).cast(
                                             "float"))
    return jaccard_df.select("user_id", "user_id_2", outputcol)


def categoryAndTemporalBasedMetrics(ratings, business_data):
    b_categories, cat_rating = getFoldCategoriesAndRatings(ratings, business_data)

    ucs = cat_rating.select("user_id", "category").distinct()
    # Esto filtra categorias con N apariciones
    categories_appearances = ucs.groupBy("category").agg(F.count(F.lit(1)).alias("nc")).orderBy(F.desc("nc")).filter(
        col("nc") <= 2000).drop("nc")

    cat_rating = cat_rating.join(categories_appearances, "category", "right")

    ucs = ucs.join(categories_appearances, "category", "right")

    ucs_all_pairs = ucs.join(ucs.select("category"
                                        , col("user_id").alias("user_id_2"))
                             , "category") \
        .filter(col("user_id") < col("user_id_2"))
    intersection = ucs_all_pairs.groupBy("user_id", "user_id_2").agg(F.count(F.lit(1)).alias("intersection"))
    ucs_grouped = ucs.select("user_id").groupBy("user_id").agg(F.count(F.lit(1)).alias("nr"))

    intersection = intersection.join(ucs_grouped, "user_id")
    intersection = intersection.join(ren(ucs_grouped), "user_id_2")

    jaccard_cat = intersection.withColumn("jaccard_cat",
                                          (col("intersection") / (col("nr") + col("nr_2") - col("intersection"))).cast(
                                              "float")).select("user_id", "user_id_2", "jaccard_cat")
    return jaccard_cat
