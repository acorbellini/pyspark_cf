def pca(train_df, feature_columns, other_columns):
    # normalized = normalize_data(train_df, feature_columns, other_columns)
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="input_features")
    features_df = assembler.transform(train_df)
    pcs = 3
    pca = PCA(k=pcs, inputCol="input_features", outputCol="pca_features")
    model = pca.fit(features_df)
    result = model.transform(features_df)
    #     result = result.rdd.map(extract(["user_id", "user_id_2"] + other_columns, "pca_features"))\
    #                        .toDF(["user_id", "user_id_2"] + other_columns + ["pc_"+str(i) for i in range(pcs)])
    #     result.show()
    return result