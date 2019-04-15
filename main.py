SEED = 42

import numpy as np
np.random.seed(SEED)

from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# http://www.learnbymarketing.com/644/recsys-pyspark-als/
# https://github.com/narenkmanoharan/Movie-Recommender-System/blob/master/recommender.py
# https://www.codementor.io/jadianes/building-a-recommender-with-apache-spark-python-example-app-part1-du1083qbw


FORMAT_STD_OUT = lambda x, w=12: \
    '[{message: <{width}}]'.format(message=x, width=w)

DATA_DIR = 'data/masters/movies/'

if __name__ == '__main__':
    # Getting the SparkContext
    sc = SparkContext()

    # Initializing the SQLContext
    sqlContext = SQLContext(sc)

    # Initializing Spark Session
    spark = SparkSession.builder.appName('recommender-system').getOrCreate()

    # Loading data using SQLContext
    # movies_header = 'MovieID::Title::Genres'.split('::')
    # movies_df = sqlContext.read.csv(
    #     DATA_DIR + 'movies.dat', header=True, sep=':', inferSchema=True
    # )

    ratings_header = 'UserID::MovieID::Rating::Timestamp'.split('::')
    ratings_df = sqlContext.read.csv(
        DATA_DIR + 'ratings.dat', header=False, sep=':', inferSchema=True
    )

    # Extract correct columns (avoid those created by double ':')
    correct_columns = np.array(ratings_df.columns)
    correct_columns = correct_columns[range(0, len(ratings_df.columns), 2)]
    ratings_df = ratings_df.select(correct_columns.tolist())

    # Rename columns
    assert len(ratings_header) == len(correct_columns)

    for i in range(len(ratings_header)):
        ratings_df = \
            ratings_df.withColumnRenamed(correct_columns[i], ratings_header[i])
    ratings_df.printSchema()

    # Create test and train set
    train, test = ratings_df.randomSplit([0.7, 0.3], seed=SEED)

    # Create ALS model
    als = ALS(
        userCol='UserID',
        itemCol='MovieID',
        ratingCol='Rating',
        nonnegative=True,
        coldStartStrategy='drop',
        implicitPrefs=False
    )
    print(FORMAT_STD_OUT('ALS'), type(als))

    # Add hyperparameters and their respective values to param_grid
    param_grid = ParamGridBuilder() \
                .addGrid(als.maxIter, [5]) \
                .addGrid(als.rank, [10, 50]) \
                .addGrid(als.regParam, [.01, .05, .15]) \
                .build()
    print(FORMAT_STD_OUT('PARAM GRID'), len(param_grid))

    # Define evaluator as RMSE and print length of evaluator
    evaluator = RegressionEvaluator(
        metricName='rmse',
        labelCol='Rating',
        predictionCol='prediction'
    )
    print(FORMAT_STD_OUT('EVALUATOR'), evaluator)

    # Build cross validator
    cv = CrossValidator(
        estimator=als,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=5
    )
    print(FORMAT_STD_OUT('CV'), cv)

    # Fit cross validator to the 'train' dataset
    model = cv.fit(train)

    # Extract best model from the cv model above
    best_model = model.bestModel
    print(FORMAT_STD_OUT('BEST MODEL'), best_model)

    get_reg_param = lambda model: model._java_obj.parent().getRegParam()
    get_max_iter  = lambda model: model._java_obj.parent().getMaxIter()

    print('rank:', best_model.rank)
    print('regParam:', get_reg_param(best_model))
    print('maxIter:', get_max_iter(best_model))

    print('\n')

    # View the predictions
    # test_predictions = best_model.transform(test)
    test_predictions.show(10)

    # Calculate and print the RMSE of the test_predictions
    rmse = evaluator.evaluate(test_predictions)
    print(FORMAT_STD_OUT('RMSE'), rmse)
