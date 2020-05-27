"""
@author: LuisFalva
"""

import random
import numpy as np

from sklearn import neighbors
from pyspark.sql import Row
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import VectorAssembler


class SparkSmote:
    
    @staticmethod
    def vector_assembling(data_input, target_name):
        """
        Vector assembling function will create a vector filled with features for each row

        :param data_input: df, spark Dataframe with target label
        :param target_name: str, string name from target label
        :return: Dataframe, table that includes the feature vector and label
        """

        if data_input.select(target_name).distinct().count() != 2:
            raise ValueError("Target field must have only 2 distinct classes")

        column_names = list(data_input.drop(target_name).columns)
        vec_assembler = VectorAssembler(inputCols=column_names, outputCol='features')
        vec_transform = vec_assembler.transform(data_input)
        vec_feature = vec_transform.select('features', (vec_transform[target_name]).alias("label"))

        return vec_feature

    @staticmethod
    def split_target(df, field, minor=1, major=0):
        """
        Split target will split in two distinct Dataframe from label 1 and 0

        :param df: Dataframe, spark Dataframe with target label
        :param field: str, string name from taget label
        :param minor: int, integer number for minority class; '1' set as default
        :param major: int, integer number for majority class; '0' set as default
        :return: dict, python dictionary with separated Dataframe
        """
        minor = df[df[field] == minor]
        major = df[df[field] == major]

        return {"minor": minor, "major": major}

    @staticmethod
    def spkdf_to_nparr(df, feature):
        """
        Spkdf to nparr function will help to parse from spark Dataframe to numpy array
        in a distributed way

        :param df: Dataframe, spark Dataframe with features column
        :param feature: str, string name of column features name
        :return: np.array, numpy array object with features
        """
        feature_df = df.select(feature)

        return np.asarray(feature_df.rdd.map(lambda x: x[0]).collect())

    @staticmethod
    def nparr_to_spkdf(spark_session, arr):
        """
        Nparr to spkdf function will help to parse from numpy array to spark Dataframe
        in a distributed way

        :param spark_session: spark session object
        :param arr: Dataframe, spark Dataframe with features column
        :return: Dataframe, with features and label; 'features' and 'label' set as default
        """
        sc = spark_session.sparkContext
        data_set = sc.parallelize(arr)
        data_rdd = data_set.map(lambda x: (Row(features=DenseVector(x), label=1)))

        return data_rdd.toDF()

    @staticmethod
    def smote_sampling(spark, df, k=2, algth="auto", pct_over_min=100, pct_under_max=100):
        """
        Smote sampling function will create an oversampling with SMOTE technique

        :param spark: spark session object
        :param df: Dataframe, spark Dataframe with features column
        :param k: int, integer k folds for k-NN's groups; '2' set as default
        :param algth: str, string name for k-NN's algorithm choice; 'auto' set as default
        :param pct_over_min: int, integer number for sampling minority class; '100' set as default
        :param pct_under_max: int, integer number for sampling majority class; '100' set as default
        :return: Dataframe, with new SMOTE features sampled
        """
        def k_neighbor(k_n, algo, feature):
            """
            k neighbor will compute k-Nearest Neighbors sklearn algorithm

            :param k_n: int, integer number for k nearest neighbors groups
            :param algo: str, string name for k-NN's algorithm choice; 'auto' set as default
            :param feature: str, string name of column features name
            :return: list, python list with numpy array object for each neighbor
            """
            n_neighbor = neighbors.NearestNeighbors(n_neighbors=k_n, algorithm=algo)
            model_fit = n_neighbor.fit(feature)
            return model_fit.kneighbors(feature)

        def compute_smo(neighbor_list, min_pct, min_arr):
            """
            Compute smo function will compute the SMOTE oversampling technique

            :param neighbor_list: list, python list with numpy array object for each neighbor
            :param min_pct: int, integer pct for over min
            :param min_arr: list, python list with minority class rows
            :param k: int, integer number for k nearest neighbors groups
            :return: list, python list with sm class oversampled
            """
            if min_pct < 100:
                raise ValueError("pct_over_min can't be less than 100")

            smo = []
            counter = 0
            pct_over = int(min_pct / 100)

            while len(min_arr) > counter:
                for i in range(pct_over):
                    random_neighbor = random.randint(0, len(neighbor)-1)
                    diff = neighbor_list[random_neighbor][0] - min_arr[i][0]
                    new_record = (min_arr[i][0] + random.random() * diff)
                    smo.insert(0, new_record)
                counter += 1
            return smo

        data_input_min = SparkSmote.split_target(df=df, field="label")["minor"]
        data_input_max = SparkSmote.split_target(df=df, field="label")["major"]

        feature_mat = SparkSmote.spkdf_to_nparr(df=data_input_min, feature="features")
        neighbor = k_neighbor(k_n=k, algo=algth, feature=feature_mat)[1]

        min_array = data_input_min.drop("label").rdd.map(lambda x: list(x)).collect()
        new_row = compute_smo(neighbor_list=neighbor, min_pct=pct_over_min, min_arr=min_array)
        smo_data_df = SparkSmote.nparr_to_spkdf(spark_session=spark, arr=new_row)
        smo_data_minor = data_input_min.unionAll(smo_data_df)

        if (pct_under_max < 10) | (pct_under_max > 100):
            raise ValueError("pct_under_max can't be less than 10 either higher than 100")
        new_data_major = data_input_max.sample(False, (float(pct_under_max / 100)))

        return new_data_major.unionAll(smo_data_minor)
