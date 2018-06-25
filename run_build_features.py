import argparse
import sys

from dataset.yelp import YelpDataset
from features.feature_builder import FeatureBuilder
from sparkconnection import SparkConnection

if __name__ == '__main__':
    # if len(sys.argv) > 1:
    #     dataset_dir = sys.argv[1]
    #     conn = SparkConnection(dir_root=dataset_dir)
    # else:
    #     conn = SparkConnection()

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="Dataset dir")
    parser.add_argument("-c", "--convert", help="Activate conversion", action="store_true")
    parser.add_argument("-x", "--distance", help="Activate distance", action="store_true")
    parser.add_argument("-f", "--folds", help="Activate folds", action="store_true")

    args = parser.parse_args()

    conn = SparkConnection(dir_root=args.dir)

    dataset = YelpDataset(conn, convert_ids=args.convert, filter_city="AZ", create_distances=args.distance)
    builder = FeatureBuilder(dataset, conn, create_folds=args.folds)
    builder.build()
