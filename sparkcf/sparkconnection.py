import os
from unittest.test.test_result import __init__

from pyspark import SparkConf, SparkContext, SQLContext


class SparkConnection:
    def __init__(self, ip="local[*]", folds=5):
        self._spark_ip = ip  # "192.168.240.10"
        if "local[*]" == ip:
            self._spark_url = ip
            current_file = os.path.dirname(os.path.abspath(__file__))
            self._dir_root = os.path.dirname(current_file)
        else:
            self._spark_url = "spark://" + self._spark_ip + ":7077"
            self._dir_root = "hdfs://" + self._spark_ip
        self._folds = folds

        conf = SparkConf()
        # conf.set("spark.executor.memory", "8g")
        conf.set("spark.network.timeout", "2000")
        conf.set("spark.sql.broadcastTimeout", "300000")
        conf.set("spark.driver.memory", "6g")
        # conf.set("spark.sql.shuffle.partitions", "400")
        # conf.set("spark.yarn.executor.memoryOverhead", "256m")
        # conf.set("spark.dynamicAllocation.enabled","true")
        # conf.set("spark.shuffle.service.enabled", "true")
        # conf.set("spark.local.dir", "/yelp-dataset/spark-tmp")
        # conf.set("spark.driver.maxResultSize","10g")
        # sc = SparkContext("local[*]", "Simple App", conf=conf)
        # sc.setCheckpointDir('/tmp')
        # os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3.6'
        os.environ['PYSPARK_PYTHON'] = 'C:/Python36/python.exe'
        conf.setMaster(self._spark_url)
        self.sc = SparkContext(conf=conf)
        self.sc.setLogLevel("ERROR")
        self.sql_sc = SQLContext(self.sc)

    def read_json(self, path):
        return self.sql_sc.read.json(os.path.join(self._dir_root, path))

    def read_parquet(self, path):
        return self.sql_sc.read.parquet(os.path.join(self._dir_root, path))

    def write_parquet(self, df, path, overwrite=True):
        if overwrite:
            df.write.mode("overwrite").parquet(os.path.join(self._dir_root, path))
        else:
            df.write.parquet(os.path.join(self._dir_root, path))

    def clear(self):
        self.sql_sc.clearCache()

    def read_csv(self, path):
        return self.sql_sc.read.csv(os.path.join(self._dir_root, path), header=True, multiLine=True, escape="\"",
                                    quote="\"")

    def write_csv(self, df, path, overwrite=True):
        if overwrite:
            df.write.mode("overwrite").csv(os.path.join(self._dir_root, path), header=True)
        else:
            df.write.csv(os.path.join(self._dir_root, path), header=True)
