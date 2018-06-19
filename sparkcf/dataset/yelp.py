from pyspark.sql import functions as f
from pyspark.sql.functions import col
from util.dataframe_utils import filter_dataframe, convert_column_to_id, replace_ids, ren
from util.distance_utils import km

rating_features = [
    "euclidean_rating",
    "cosine_rating",
    "jaccard_rating",
    "pearson_rating"
]

social_features = [
    "adamic_adar_graph",
    "jaccard_graph",
    "cosine_graph",
    "preferential_attachment"
    # , "are_friends"
]

social_popularity = [
    # "cc_sim",
    "aa_pop_ratings",
    "aa_pop_tips"
]

# cat_features = [
#     "jaccard_cat",
#     "jaccard_city_cat"
# ]

geo_features = [
    "home_sim",
    "jaccard_business_distance",
    "jaccard_distance_cat"
]

probably_best = [
                    "home_sim"
                ] + rating_features

probably_best_2 = geo_features + rating_features

# Puede usarse para pesar la recomendación.
popularity_metrics = [
    # "pagerank",
    # "pagerank_2",
    "usefulness",
    "usefulness_2",
    "liked_tips",
    "liked_tips_2"
]

all_features = rating_features + social_features + social_popularity + geo_features

test_features = {
    "geo": geo_features,
    "baseline": ["cosine_rating"],
    "ratings": rating_features,
    "plus_geo": geo_features + rating_features,
    "home_sim": ["home_sim"],
    "jaccard_business_distance": ["jaccard_business_distance"],
    "jaccard_distance_cat": ["jaccard_distance_cat"],
    "social_features": social_features,
    "social_popularity": social_popularity,
    "all_social": social_features + social_popularity,
    # "cc_sim": ["cc_sim"],
    "aa_pop_ratings": ["aa_pop_ratings"],
    "aa_pop_tips": ["aa_pop_tips"],
    "all_features": all_features
}

extra_cols = ["usefulness", "usefulness_2",
              # "pagerank", "pagerank_2",
              "liked_tips", "liked_tips_2"]


def replace_all(df, user_id, business_id):
    df = replace_ids(df, "business_id", business_id, "business_name")
    return replace_ids(df, "user_id", user_id, "user_name")


class YelpDataset:
    def __init__(self, conn, convert_ids=False, filter_city="AZ", min_ratings=10, create_distances=False):
        if convert_ids:
            self.create_files(conn, create_distances, filter_city)

        self.business = conn.read_parquet("dataset/yelp_business_converted_" + filter_city)
        self.business_categories = conn.read_parquet("dataset/yelp_business_categories_converted_" + filter_city)
        self.users = conn.read_parquet("dataset/yelp_user_converted_" + filter_city)
        self.checkins = conn.read_parquet("dataset/yelp_checkin_converted_" + filter_city)
        self.tips = conn.read_parquet("dataset/yelp_tip_converted_" + filter_city)
        self.ratings = conn.read_parquet("dataset/yelp_review_converted_" + filter_city)
        self.social_graph = conn.read_parquet("dataset/social_graph_converted_" + filter_city)
        self.business_distance = conn.read_parquet("dataset/business_distance/")

        users_filtered = self.ratings.groupBy("user_id").agg(f.count("business_id").alias("num_ratings")) \
            .filter(col("num_ratings") >= min_ratings) \
            .select("user_id")
        self.ratings = self.ratings.join(users_filtered, "user_id", "right")
        self.users = self.users.join(users_filtered, "user_id", "right")

    def create_files(self, conn, create_distances, filter_city):
        business_df = conn.read_json("dataset/business.json")
        users_df = conn.read_json("dataset/user.json")
        checkins_df = conn.read_json("dataset/checkin.json")
        tips_df = conn.read_json("dataset/tip.json")
        ratings_df = conn.read_json("dataset/review.json")

        if filter_city is not None:
            business_df = business_df.filter(col("state") == "AZ")
        checkins_df = filter_dataframe(checkins_df, business_df, "business_id")
        tips_df = filter_dataframe(tips_df, business_df, "business_id")
        ratings_df = filter_dataframe(ratings_df, business_df, "business_id")
        users_df = filter_dataframe(users_df, ratings_df, "user_id")

        user_id, users = convert_column_to_id(users_df, "user_id", "user_name")
        business_id, business = convert_column_to_id(business_df, "business_id", "business_name")

        checkins = replace_ids(checkins_df, "business_id", business_id, "business_name")
        ratings = replace_all(ratings_df, user_id, business_id)
        tips = replace_all(tips_df, user_id, business_id)

        social_graph = users.select("user_id", "friends") \
            .select("user_id",
                    f.explode(
                        "friends").alias(
                        "friend"),
                    f.size(
                        "friends").alias(
                        "nf")).join(
            user_id.select(col("user_name").alias("friend"), col("user_id").alias("friend_id")), "friend").select(
            "user_id", col("friend_id").alias("friend"), "nf")
        business_categories = business.select(col("business_id"), "categories") \
            .select("business_id", f.explode("categories").alias("category"),
                    f.size("categories").alias("nc"))
        category_id = business_categories.select("category").distinct().rdd.map(lambda x: x[0]).zipWithIndex() \
            .toDF(["category_name", "category"]).select("category_name", col("category").cast("integer"))
        business_categories = business_categories.select("business_id",
                                                         col("category").alias("category_name"),
                                                         "nc").join(category_id, "category_name") \
            .drop("category_name")

        ratings = ratings.select("user_id",
                                 "business_id",
                                 "useful",
                                 "stars")
        users = users.select("user_id",
                             "average_stars")
        business = business.select("business_id",
                                   "city",
                                   "latitude",
                                   "longitude")
        social_graph = social_graph.select("user_id",
                                           "friend",
                                           "nf")
        tips = tips.select("user_id",
                           "business_id",
                           "likes")

        conn.write_parquet(business, "dataset/yelp_business_converted_" + filter_city)
        # conn.write_parquet(business_categories, "dataset/yelp_business_categories_converted_" + filter_city)
        # conn.write_parquet(users, "dataset/yelp_user_converted_" + filter_city)
        # conn.write_parquet(checkins, "dataset/yelp_checkin_converted_" + filter_city)
        # conn.write_parquet(tips, "dataset/yelp_tip_converted_" + filter_city)
        # conn.write_parquet(ratings, "dataset/yelp_review_converted_" + filter_city)
        # conn.write_parquet(social_graph, "dataset/social_graph_converted_" + filter_city)
        if create_distances:
            lat_and_long = business.select("business_id", "longitude", "latitude")
            business_distance = lat_and_long.crossJoin(f.broadcast(ren(lat_and_long))) \
                .withColumn("distance",
                            km(col("latitude"),
                               col("longitude"),
                               col("latitude_2"),
                               col("longitude_2"))) \
                .where(col("distance") <= 1).select("business_id", "business_id_2", "distance")
            conn.write_parquet(business_distance, "dataset/business_distance/")
