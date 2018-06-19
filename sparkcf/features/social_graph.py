from pyspark.sql.functions import col
import pyspark.sql.functions as F

from util.dataframe_utils import ren


def socialBasedMetrics(ratings, user_friend):
    fold_user_friend = user_friend.join(ratings.select("user_id").distinct(), "user_id", "right")
    fu_with_friendsize = fold_user_friend.join(fold_user_friend.select(col("user_id").alias("friend"),
                                                                       col("nf").alias("nf_friend")).distinct(),
                                               "friend") \
        .select("user_id", "nf", "friend", "nf_friend")

    ufJoin = fu_with_friendsize.join(ren(fu_with_friendsize, ["friend"]), "friend").filter(
        col("user_id") < col("user_id_2"))

    intersection = ufJoin.groupBy("user_id", "user_id_2", "nf", "nf_2").agg(F.count(F.lit(1)).alias("intersection"),
                                                                            F.sum(1 / F.log("nf_friend")).cast(
                                                                                "float").alias("adamic_adar_graph"))

    graph = intersection.withColumn("jaccard_graph",
                                    (col("intersection") / (col("nf") + col("nf_2") - col("intersection"))).cast(
                                        "float")) \
        .withColumn("cosine_graph", (col("intersection") / (F.sqrt(col("nf") * col("nf_2")))).cast("float")) \
        .withColumn(
        "preferential_attachment", col("nf") * col("nf_2")).select("user_id", "user_id_2", "adamic_adar_graph",
                                                                   "jaccard_graph", "cosine_graph",
                                                                   "preferential_attachment").filter(
        (col("adamic_adar_graph") > 0) | (col("jaccard_graph") > 0) | (col("cosine_graph") > 0))

    return graph
