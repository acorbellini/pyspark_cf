import sys

from cf.experimentrunner import ExperimentRunner
from dataset.yelp import YelpDataset, test_features, extra_cols
from sparkconnection import SparkConnection

if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
        conn = SparkConnection(dir_root=dataset_dir)
    else:
        conn = SparkConnection()

    dataset = YelpDataset(conn)

    exp = ExperimentRunner(conn, dataset)
    exp.run_all(test_features, extra_cols)
