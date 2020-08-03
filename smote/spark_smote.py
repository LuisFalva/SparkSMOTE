import random
import numpy as np

from sklearn import neighbors
from pyspark.sql import Row, DataFrame
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import VectorAssembler

"""
This class is based on the original solution of Angkirat Singh Sandhu derived to a better performance and generalized method over spark.
link: https://github.com/Angkirat/Smote-for-Spark/blob/master/PythonCode.py
"""

class SparkSmote:

    @staticmethod
    def vector_assembling(df, target_name):
        """
        Vector assembling function will create a vector filled with features for each row

        :param df: df, spark Dataframe with target label
        :param target_name: str, string name from target label
        :return: Dataframe, table that includes the feature vector and label
        """

        if not isinstance(target_name, str):
            raise ValueError("target name must be specified")

        if not isinstance(df, DataFrame):
            raise ValueError("spark dataframe is required")

        column_names = list(df.drop(target_name).columns)
        vec_assembler = VectorAssembler(inputCols=column_names, outputCol='features')
        vec_transform = vec_assembler.transform(df)
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
        if not isinstance(field, str):
            raise ValueError("field must be specified")

        if not isinstance(df, DataFrame):
            raise ValueError("spark dataframe is required")

        minor = df[df[field] == minor]
        major = df[df[field] == major]

        return {"minor": minor, "major": major}

    @staticmethod
    def spark_to_numpy(df, feature):
        """
        Spark to numpy function will help to parse from spark Dataframe to numpy array
        in a distributed manner

        :param df: Dataframe, spark Dataframe with features column
        :param feature: str, string name of column features name
        :return: np.array, numpy array object with feature elements
        """
        if not isinstance(feature, str):
            raise ValueError("feature must be specified")

        if not isinstance(df, DataFrame):
            raise ValueError("spark dataframe is required")

        feature_df = df.select(feature)

        return np.asarray(feature_df.rdd.map(lambda x: x[0]).collect())

    @staticmethod
    def numpy_to_spark(spark_session, arr):
        """
        Numpy to spark function will help to parse from numpy array to spark Dataframe
        in a distributed manner

        :param spark_session: spark session object
        :param arr: np.array, numpy array object with feature elements
        :return: Dataframe, with features and label; 'features' and 'label' set as default
        """
        if spark_session is None:
            raise ValueError("spark session is required")

        if not isinstance(arr, np.ndarray):
            raise ValueError("(M, p) np.array or np.asarray object is required")

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
        if (pct_under_max < 10) | (pct_under_max > 100) | (not isinstance(pct_under_max, int)):
            raise ValueError("please enter a valid max percentage, this can't be < 10 either > 100")

        def k_neighbor(k_n, algm, feature):
            """
            k neighbor will compute k-Nearest Neighbors sklearn algorithm

            :param k_n: int, integer number for k nearest neighbors groups; 2 set as default
            :param algm: str, string name for k-NN's algorithm choice; 'auto' set as default
            :param feature: str, string name of column features name
            :return: list, python list with numpy array object for each neighbor
            """
            if (not isinstance(k_n, int)) | (k_n is None):
                raise ValueError("please set K integer for groups")

            if (not isinstance(algm, str)) | (algm is None):
                raise ValueError("please choose a valid algorithm")

            if (not isinstance(feature, str)) | (feature is None):
                raise ValueError("you must specify feature column name")

            n_neighbor = neighbors.NearestNeighbors(n_neighbors=k_n, algorithm=algm)
            model_fit = n_neighbor.fit(feature)
            return model_fit.kneighbors(feature)

        def compute_smo(neighbor_list, min_pct, min_arr):
            """
            Compute smo function will compute the SMOTE oversampling technique

            :param neighbor_list: list, python list with numpy array object for each neighbor
            :param min_pct: int, integer pct for over min; 100 set as default
            :param min_arr: list, python list with minority class rows
            :return: list, python list with sm class oversampled
            """
            if (not isinstance(neighbor_list, list)) | (neighbor_list is None):
                raise ValueError("please enter a valid neighborhood list")

            if (not isinstance(min_arr, list)) | (min_arr is None):
                raise ValueError("please enter a valid minority class list")

            if (min_pct < 100) | (not isinstance(min_pct, int)):
                raise ValueError("please enter a valid min percentage, this must be > 100")

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

        feature_mat = SparkSmote.spark_to_numpy(df=data_input_min, feature="features")
        neighbor = k_neighbor(k_n=k, algm=algth, feature=feature_mat)[1]

        min_array = data_input_min.drop("label").rdd.map(lambda x: list(x)).collect()
        new_row = compute_smo(neighbor_list=neighbor, min_pct=pct_over_min, min_arr=min_array)
        smo_data_df = SparkSmote.numpy_to_spark(spark_session=spark, arr=new_row)
        smo_data_minor = data_input_min.union(smo_data_df)
        new_data_major = data_input_max.sample(False, (float(pct_under_max / 100)))

        return new_data_major.union(smo_data_minor)
