# coding: utf-8

from pyspark.sql.functions import broadcast
from pyspark.sql.functions import count, col, lit, size, explode

from features.geo import user_home, geo_distance_metrics
from features.popularity import popularity_based_metrics, usefulness, liked_tips
from features.rating_based import rating_based_metrics
from features.social_graph import socialBasedMetrics
from util.dataframe_utils import ren, train_test_split_randomized
from util.distance_utils import km


class FeatureBuilder:
    def __init__(self, dataset, conn, create_distances=False, create_folds=False, folds=1, split=0.8):
        self._dataset = dataset
        self._conn = conn
        self._create_distances = create_distances
        self._create_folds = create_folds
        self._folds = folds
        self._split = split

    def build(self):
        ###########################################################
        # Esto se puede usar para filtrar por cantidad de train_ratings.
        ###########################################################

        # print(users.count())
        # print(train_ratings.count())

        # all_ratings.orderBy("user_id", "business_id").show()

        # user_friend = user_friend.withColumnRenamed("friend", "friend_name").join(
        #     user_id.select(col("user_id").alias("friend_id"),
        #                    col("user_name").alias("friend_name")),
        #     "friend_name").select("user_id", "friend_id", "nf")

        # *********************************************
        # Agregar Connected Components y PageRank PRECALCULADOS
        # *********************************************
        # cc = self._conn.read_parquet("hdfs://192.168.240.10/yelp-graph/connected_components")
        # pagerank = self._conn.read_parquet("hdfs://192.168.240.10/yelp-graph/user_pagerank")

        ###########################################################
        # Crear Folds Con Ratings Originales
        ###########################################################
        if self._create_folds:
            for fold in range(self._folds):
                print("Building Fold {}".format(fold))
                train_ratings, test_ratings = train_test_split_randomized(self._dataset.ratings, fold, self._folds,
                                                                          self._split)
                self._conn.write_parquet(train_ratings, "fold_" + str(fold) + "/train")
                self._conn.write_parquet(test_ratings, "fold_" + str(fold) + "/test")
                self._conn.clear()

        are_friends = self._dataset.social_graph.select("user_id", col("friend").alias("user_id_2")) \
            .withColumn("are_friends",
                        lit(True).cast(
                            "boolean"))
        for fold in range(self._folds):
            print("Creating features for fold {}".format(fold))
            train_ratings = self._conn.read_parquet("fold_{}/train".format(fold))
            rating_based = rating_based_metrics(train_ratings)
            # Filtro los features sociales por los usuarios del fold
            social_based = socialBasedMetrics(train_ratings, self._dataset.social_graph)
            popularity_based = popularity_based_metrics(train_ratings, self._dataset.tips)

            user_home_df = user_home(train_ratings, self._dataset.users, self._dataset.business,
                                     self._dataset.business_distance,
                                     filter_best_ratings=False)

            user_home_distance = user_home_df.crossJoin(broadcast(ren(user_home_df))).withColumn("home_distance",
                                                                                                 km(col("latitude"),
                                                                                                    col("longitude"),
                                                                                                    col("latitude_2"),
                                                                                                    col("longitude_2"))) \
                .where(col("home_distance") < 1) \
                .select("user_id", "user_id_2", (lit(1) / (lit(1) + col("home_distance"))).alias("home_sim"))

            self._conn.write_parquet(user_home_distance, "tmp/fold_{}_home_distance".format(fold))
            user_home_distance = self._conn.read_parquet("tmp/fold_{}_home_distance".format(fold))

            fold_business_data = self._dataset.business.join(train_ratings.select("business_id").distinct(),
                                                             "business_id",
                                                             "right")
            #     catBased = categoryAndTemporalBasedMetrics(train_ratings, fold_business_data)
            geo_based = geo_distance_metrics(train_ratings, fold_business_data, self._dataset.business_distance,
                                             self._dataset.business_categories)
            dfs = [rating_based, social_based,
                   #            catBased,
                   geo_based, popularity_based, user_home_distance]

            current_df = None
            df_count = 0
            for df in dfs:
                if current_df is None:
                    current_df = df
                else:
                    current_df = current_df.join(df, ["user_id", "user_id_2"], "outer")

                self._conn.write_parquet(current_df, "tmp/fold_{}_{}".format(fold, df_count))
                current_df = self._conn.read_parquet("tmp/fold_{}_{}".format(fold, df_count))
                df_count = df_count + 1

            # all_features = current_df.join(cc, "user_id", "leftouter").join(
            #     cc.select(col("user_id").alias("user_id_2"), col("cc").alias("cc_2")), "user_id_2",
            #     "leftouter").withColumn(
            #     "cc_sim", F.when(col("cc") == col("cc_2"), 1.0).otherwise(0.0))
            #
            # all_features = all_features.join(pagerank, "user_id", "leftouter").join(
            #     pagerank.select(col("user_id").alias("user_id_2"),
            #                     col("pagerank").alias("pagerank_2")), "user_id_2", "leftouter")

            usefulness_df = usefulness(train_ratings)
            all_features = current_df.join(usefulness_df, "user_id", "leftouter").join(
                usefulness_df.select(col("user_id").alias("user_id_2"),
                                     col("usefulness").alias("usefulness_2")), "user_id_2", "leftouter")

            liked_tips_df = liked_tips(train_ratings, self._dataset.tips)
            all_features = all_features.join(liked_tips_df, "user_id", "leftouter").join(
                liked_tips_df.select(col("user_id").alias("user_id_2"),
                                     col("liked_tips").alias("liked_tips_2")), "user_id_2", "leftouter")

            #     with_class = all_features.join(are_friends, ["user_id", "user_id_2"], "leftouter")\
            #                      .na.fill({"are_friends": False}).fillna(0)

            self._conn.write_parquet(all_features, "/fold_{}/train_features".format(fold))

            all_features.printSchema()

            print("Finished building fold " + str(fold))
            return all_features
