from pyspark.sql.functions import col
import pyspark.sql.functions as F


def normalize_data(train_df, feature_columns, other_columns=[]):
    train_df = train_df.select(["user_id", "user_id_2"] + feature_columns + other_columns)
    # train_df.show()
    ops = []
    ops_norm = []
    for f in feature_columns:
        ops.append(F.min(f).alias("n_" + f + "_min"))
        ops.append(F.max(f).alias("n_" + f + "_max"))
        ops_norm.append(((col(f) - col("n_" + f + "_min")) /
                         (col("n_" + f + "_max") - col("n_" + f + "_min"))).alias("n_" + f))
    gen = (op for op in ops)
    norms = train_df.agg(*ops)
    train_df = train_df.crossJoin(F.broadcast(norms)).select(
        ["user_id", "user_id_2"] + ops_norm + other_columns
    )
    return train_df
