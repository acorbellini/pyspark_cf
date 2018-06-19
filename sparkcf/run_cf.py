from cf.experimentrunner import ExperimentRunner
from dataset.yelp import YelpDataset, test_features, extra_cols
from sparkconnection import SparkConnection

if __name__ == '__main__':
    conn = SparkConnection()
    dataset = YelpDataset(conn)

    exp = ExperimentRunner(conn, dataset)
    exp.run_all(test_features, extra_cols)
