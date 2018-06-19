from features.feature_builder import FeatureBuilder
from sparkconnection import SparkConnection
import pyspark.sql.functions as f
from dataset.yelp import YelpDataset

if __name__ == '__main__':
    conn = SparkConnection()
    dataset = YelpDataset(conn, convert_ids=False, filter_city="AZ", create_distances=False)
    builder = FeatureBuilder(dataset, conn, create_folds=False)
    builder.build()
