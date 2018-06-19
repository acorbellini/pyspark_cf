# coding: utf-8

# In[1]:


import os

import numpy as np
import pyspark.sql.functions as F
from pyspark import SparkContext, SQLContext, SparkConf
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Window
from pyspark.sql.functions import col, row_number, lit
from pyspark.sql.types import FloatType
from pyspark.sql.utils import AnalysisException


# In[2]:



# In[3]:


SPARK_IP = "192.168.240.10"
SPARK_URL = "spark://" + SPARK_IP + ":7077"
MIN_RATINGS = 10
TRAIN_TEST_SPLIT = 0.8
DIR_ROOT = "hdfs://" + SPARK_IP
FOLDS = 5
CREATE_FOLDS = False
CREATE_DISTANCES = False

# In[4]:


conf = SparkConf()
conf.set("spark.executor.memory", "4g")
# conf.set("spark.sql.shuffle.partitions", "400")
# conf.set("spark.yarn.executor.memoryOverhead", "256m")
conf.set("spark.network.timeout", "2000")
conf.set("spark.sql.broadcastTimeout", "300000")
# conf.set("spark.dynamicAllocation.enabled","true")
# conf.set("spark.shuffle.service.enabled", "true")
# conf.set("spark.local.dir", "/yelp-dataset/spark-tmp")
conf.set("spark.driver.memory", "1g")
# conf.set("spark.driver.maxResultSize","10g")
# sc = SparkContext("local[*]", "Simple App", conf=conf)
# sc.setCheckpointDir('/tmp')
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3.6'
conf.setMaster(SPARK_URL)
sc = SparkContext(conf=conf)
sql_sc = SQLContext(sc)

# In[5]:



# users.persist()


# In[6]:





# + cat_features + geo_features










#     train_df = train_df.select(["user_id", "user_id_2"] + feature_columns + other_columns)    
#     assembler = VectorAssembler(inputCols=feature_columns, outputCol="input_features")
#     features_df = assembler.transform(train_df)

#     scaler = StandardScaler(inputCol="input_features", outputCol="scaled_features",
#                             withStd=True, withMean=True)
#     # Compute summary statistics by fitting the StandardScaler
#     scalerModel = scaler.fit(features_df)
#     # Normalize each feature to have unit standard deviation.
#     scaledData = scalerModel.transform(features_df)
#     normalizer = Normalizer(inputCol="scaled_features", outputCol="normalized_features", p=2.0)
#     l1NormData = normalizer.transform(scaledData)
#     return l1NormData











# check_users = sql_sc.createDataFrame([("u1", 3.0),
#                              ("u2", 2.0),
#                             ("u3", 1.0)
#                             ], ["user_id", "average_stars"])

# check_df = sql_sc.createDataFrame([("u1", "a", 3.5, None),
#                              ("u1", "c", 3.5, 3.5),
#                              ("u1", "d", 3.5, 3.5),
#                              ("u1", "e", 3.5, 3.5),
#                              ("u1", "g", 3.5, 3.5),
#                              ("u2", "f", 2.0, None),
#                              ("u2", "a", 5.0, 5.0),
#                              ("u2", "f", 2.0, 2.0),
#                              ("u2", "a", 5.0, 5.0),
#                              ("u2", "f", 2.0, 2.0),
#                             ("u2", "a", 5.0, 5.0),
#                             ("u3", "b", 3.0, 3.0),
#                              ("u3", "b", 3.0, None),
#                              ("u3", "b", 3.0, 3.0),
#                              ("u3", "b", 3.0, 3.0),
#                              ("u3", "b", 3.0, 3.0)
#                             ], ["user_id", "business_id", "stars", "pred_stars"])
# mae, cov, prec, prec_liked = evaluate(check_df, check_users, 100)
# print("{},{},{},{}".format(mae, cov, prec, prec_liked))


# In[7]:


# In[9]:




