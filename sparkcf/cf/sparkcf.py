from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Window
from pyspark.sql.functions import col
import pyspark.sql.functions as F


class SparkCF:
    def __init__(self, conn):
        self._conn = conn

    def evaluate_batch(self, test_ratings, weighted_ratings, users):
        total_users = test_ratings.select("user_id").distinct().count()
        weighted_ratings.cache()
        ratingsPredicted = test_ratings.select("user_id", "business_id", col("stars").cast("float")).join(
            weighted_ratings,
            ["user_id", "business_id"])
        #     predictedVsActual.orderBy("user_id").show()
        #     print(ratingsPredicted.groupBy("user_id", "business_id")\
        #                         .agg(F.count("*").alias("count")).where(col("count")>1).count())

        total = test_ratings.count()
        predicted = ratingsPredicted.count()
        print("Total {}".format(total), " Predicted {}".format(predicted))
        coverage = predicted * 100 / total

        evaluator = RegressionEvaluator(metricName="mae", labelCol="stars", predictionCol="pred_stars")
        mae = evaluator.evaluate(ratingsPredicted)

        precision = {}
        precision_liked = {}
        for i in [5, 10]:
            wi = Window.partitionBy("user_id").orderBy(col("pred_stars").desc())
            topi = weighted_ratings.withColumn("rn", F.row_number().over(wi)).where(col("rn") <= i).drop("rn")
            #         topi.show(100)
            topratings = topi.join(test_ratings.select("user_id", "business_id", "stars"),
                                   ["user_id", "business_id"]) \
                .join(users.select("user_id", "average_stars"), "user_id")
            topratings = topratings.withColumn("liked", col("stars") >= col("average_stars")) \
                .withColumn("pred_liked",
                            col(
                                "pred_stars") >= col(
                                "average_stars"))
            topratings.cache()
            hits = topratings.count()
            hits_liked = topratings.where(col("liked") == col("pred_liked")).count()
            precision[i] = hits / (i * total_users)
            precision_liked[i] = hits_liked / (i * total_users)
            print("Hits : {} ".format(hits), " Hits Liked: {} ".format(hits_liked))
            topratings.unpersist()
        weighted_ratings.unpersist()
        return mae, coverage, precision, precision_liked
