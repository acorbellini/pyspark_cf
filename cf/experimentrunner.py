from pyspark.sql.functions import col
import pyspark.sql.functions as F

from cf.pca import pca
from cf.simplenormalization import normalize_data
from cf.sparkcf import SparkCF
from cf.topk import select_topk_users
from util.dataframe_utils import colwise_mean


def get_boost(boost_col):
    return lambda sim: col(boost_col) * col(sim)


boost_usefulness = {
    "user_id": get_boost("usefulness"),
    "user_id_2": get_boost("usefulness_2")
}

boost_pagerank = {
    "user_id": get_boost("pagerank"),
    "user_id_2": get_boost("pagerank_2")
}

boost_likes = {
    "user_id": get_boost("liked_tips"),
    "user_id_2": get_boost("liked_tips_2")
}


def build_config(similarity, selection, w):
    return {
        "similarity": similarity,
        "selection": selection,
        "weighting": w,
        "folds": 1
    }


class ExperimentRunner:
    def __init__(self, conn, dataset):
        self._conn = conn
        self._dataset = dataset

    def run_fold(self, fold, config):
        cf = SparkCF(self._conn)

        train_features = self._conn.read_parquet("fold_" + str(fold) + "/train_features")
        train_features = train_features.where(col("user_id") != col("user_id_2"))

        # train_features = train_features.withColumn("are_friends", F.when(col("are_friends") == True, 1).otherwise(0))

        train_ratings = self._conn.read_parquet("fold_" + str(fold) + "/train")
        test_ratings = self._conn.read_parquet("fold_" + str(fold) + "/test")

        # test_users = test_ratings.select("user_id").distinct()
        with_similarity = config["similarity"](train_features)
        neighbours = config["selection"](with_similarity)
        ratings = neighbours.join(
            train_ratings.select(col("user_id").alias("user_id_2"),
                                 "business_id",
                                 "stars")
            , "user_id_2")

        weighted_ratings = ratings.groupBy("user_id", "business_id").agg(config["weighting"]().alias("pred_stars"))

        mae, coverage, precision, precision_liked = cf.evaluate_batch(test_ratings, weighted_ratings,
                                                                      self._dataset.users)

        return mae, coverage, precision, precision_liked

    def run_experiments(self, config):
        folds = config["folds"]
        sum_mae = 0
        sum_coverage = 0
        sum_precision = {i: 0 for i in [5, 10]}
        sum_precision_liked = {i: 0 for i in [5, 10]}
        for fold in range(config["folds"]):
            mae, coverage, p, p_liked = self.run_fold(fold, config)
            sum_mae = sum_mae + mae
            sum_coverage = sum_coverage + coverage
            sum_precision = {k: v + p[k] for (k, v) in sum_precision.items()}
            sum_precision_liked = {k: v + p_liked[k] for (k, v) in sum_precision_liked.items()}
            # print("Fold {} - MAE: {} - COV: {}".format(fold, mae, coverage))
        avg_mae = sum_mae / folds
        avg_cov = sum_coverage / folds
        avg_prec = {k: v / folds for (k, v) in sum_precision.items()}
        avg_prec_liked = {k: v / folds for (k, v) in sum_precision_liked.items()}
        print("MAE = {}, Cov % = {}, Prec = {}, Prec Liked = {}".format(avg_mae, avg_cov, avg_prec, avg_prec_liked))
        return avg_mae, avg_cov, avg_prec, avg_prec_liked

    def run_all(self, test_features, extra_cols):
        results = []

        knn_values = [10]
        for k in knn_values:
            for name, features in test_features.items():
                top_k = lambda df: select_topk_users(df, k)
                top_k_boost_usefull = lambda df: select_topk_users(df, k, boost_usefulness)
                # top_k_boost_pagerank = lambda df: select_topk_users(df, k, boost_pagerank)
                top_k_boost_likes = lambda df: select_topk_users(df, k, boost_likes)

                normalized = ["n_" + f for f in features]

                mean_on_f = lambda df: normalize_data(df, features, extra_cols).withColumn("similarity",
                                                                                           colwise_mean(normalized))
                pca_on_f = lambda df: pca(df, features, extra_cols).withColumn("similarity",
                                                                               col("pca_features").getItem(0))
                max_on_f = lambda df: normalize_data(df, features, extra_cols).withColumn("similarity",
                                                                                          F.greatest(*normalized))
                min_on_f = lambda df: normalize_data(df, features, extra_cols).withColumn("similarity",
                                                                                          F.least(*normalized))

                mean_w = lambda: F.mean(col("stars"))
                sim_w = lambda: F.mean(col("similarity") * col("stars")) / F.max("similarity")

                configs = {}
                #         if len(features)>1:
                #             configs["pca_" + str(k)] =  build_config(pca_on_f, top_k, mean_w)

                configs["mean_" + str(k)] = build_config(mean_on_f, top_k, mean_w)
                # configs["boost_uf_" + str(k)] = build_config(mean_on_f, top_k_boost_usefull, mean_w)
                # configs["boost_pr_" + str(k)] = build_config(mean_on_f, top_k_boost_pagerank, mean_w)
                # configs["boost_lk_" + str(k)] = build_config(mean_on_f, top_k_boost_likes, mean_w)

                # con similitud ponderada
                #         configs["mean_simw_" + str(k)] = build_config(mean_on_f, top_k, sim_w)
                #         configs["boost_simw_uf_"+ str(k)] =  build_config(mean_on_f, top_k_boost_usefull, sim_w)
                #         configs["boost_simw_pr_"+ str(k)] =  build_config(mean_on_f, top_k_boost_pagerank, sim_w)
                #         configs["boost_simw_lk_"+ str(k)] =  build_config(mean_on_f, top_k_boost_likes, sim_w)

                # if len(features) > 1:
                # configs["min_" + str(k)] = build_config(min_on_f, top_k, mean_w)
                # configs["min_boost_uf_" + str(k)] = build_config(min_on_f, top_k_boost_usefull, mean_w)
                # configs["min_boost_pr_" + str(k)] = build_config(min_on_f, top_k_boost_pagerank, mean_w)
                # configs["min_boost_lk_" + str(k)] = build_config(min_on_f, top_k_boost_likes, mean_w)

                # configs["max_" + str(k)] = build_config(max_on_f, top_k, mean_w)
                # configs["max_boost_uf_" + str(k)] = build_config(max_on_f, top_k_boost_usefull, mean_w)
                # configs["max_boost_pr_" + str(k)] = build_config(max_on_f, top_k_boost_pagerank, mean_w)
                # configs["max_boost_lk_" + str(k)] = build_config(max_on_f, top_k_boost_likes, mean_w)

                #         configs["pca_simw_" + str(k)] =  build_config(pca_on_f, top_k, sim_w)

                for c_type, config in configs.items():
                    print("Features: {}, k={}, config={}".format(features, k, c_type))
                    mae, cov, avg_prec, avg_prec_liked = self.run_experiments(config)
                    #             sql_sc.clearCache()
                    header = name + "_" + str(c_type)
                    results.append((header, mae, cov, avg_prec[5], avg_prec[10], avg_prec_liked[5], avg_prec_liked[10]))
                    results_df = self._conn.sql_sc.createDataFrame(results,
                                                                   schema=["features", "avg_mae", "avg_cov",
                                                                           "avg_prec_5",
                                                                           "avg_prec_liked_5", "avg_prec_10",
                                                                           "avg_prec_liked_10"])
                    self._conn.write_csv(results_df, "cf_results/results.csv")
                # results_df.show(truncate=False)
