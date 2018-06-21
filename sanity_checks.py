# sql_sc.read.csv(DIR_ROOT + "/cf_results/results.csv", header=True).show(truncate=False)
# sql_sc.read.csv(DIR_ROOT + "/cf_results/results.csv", header=True).show(1000, False)
# sql_sc.clearCache()
#
# train_ratings = sql_sc.read.parquet(DIR_ROOT + "/fold_0/train")
# test_ratings = sql_sc.read.parquet(DIR_ROOT + "/fold_0/test")
# print(train_ratings.select("business_id").union(test_ratings.select("business_id")).distinct().count())
# print(train_ratings.count())
# print(train_ratings.select("user_id").distinct().count())
# print(test_ratings.count())
# print(test_ratings.select("user_id").distinct().count())
#
# train_features = sql_sc.read.parquet(DIR_ROOT + "/fold_0/train_features")
# print(train_features.count())
#
# # train_ratings.select("user_id", "business_id",  "fold").orderBy("user_id", "business_id").show()
# # test_ratings.select("user_id", "business_id",  "fold").orderBy("user_id", "business_id").show()
# intersect = train_ratings.select("user_id", "business_id").intersect(test_ratings.select("user_id", "business_id"))
# train_ratings.select("user_id", "business_id").join(intersect, ["user_id", "business_id"], "right").show()
# test_ratings.select("user_id", "business_id").join(intersect, ["user_id", "business_id"], "right").show()
#
# print(intersect.count())
# intersect.show()
#
# all_ratings = train_ratings.select("user_id", "business_id", "stars").union(
#     test_ratings.select("user_id", "business_id", "stars"))
#
# all_ratings.coalesce(1).write.csv(DIR_ROOT + "/all_ratings", header=True)
#
# train_ratings = sql_sc.read.parquet(DIR_ROOT + "/fold_0/train")
# test_ratings = sql_sc.read.parquet(DIR_ROOT + "/fold_0/test")
# # train_ratings.select("user_id", "business_id",  "fold").orderBy("user_id", "business_id").show()
# # test_ratings.select("user_id", "business_id",  "fold").orderBy("user_id", "business_id").show()
# intersect = train_ratings.select("user_id", "business_id").intersect(test_ratings.select("user_id", "business_id"))
# train_ratings.select("user_id", "business_id").join(intersect, ["user_id", "business_id"], "right").show()
# test_ratings.select("user_id", "business_id").join(intersect, ["user_id", "business_id"], "right").show()
#
# print(intersect.count())
# intersect.show()
from sparkconnection import SparkConnection

if __name__ == '__main__':
    conn = SparkConnection()
    for i in ["business", "user", "review"]:
        converted = conn.read_csv("dataset/yelp_{}_converted_AZ.csv".format(i))
        print(converted.count())
        original = conn.read_csv("dataset/yelp_{}.csv".format(i))
        print(original.count())
