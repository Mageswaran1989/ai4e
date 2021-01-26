import pyspark
import pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from context import src
# import pytest
import unittest

stockPriceInputDir = "./resources/input/stock_prices"
stockPriceDirtyInputDir = "./resources/input/stock_prices_dirty"
stockInfoInputDir = "./resources/input/stock_info"


def checkAnswer(actualDF, expectedDF):
    diffDF = actualDF.union(expectedDF).exceptAll(actualDF.intersect(expectedDF))
    if diffDF.count() != 0 :
        print("Expected no difference between actual and expected, but found following difference:")
        diffDF.show()
        # This will fail test, as count should be different anyhow
        assert actualDF.count() == expectedDF.count()

# @pytest.fixture(scope="session")
def test_moving_average(spark_session):
    actualDF = src.MovingAverage(spark_session, stockPriceInputDir, 3).calculate()

    schema = StructType([
        StructField("stockId", IntegerType()),
        StructField("timeStamp", IntegerType()),
        StructField("stockPrice", DoubleType()),
        StructField("moving_average", DoubleType())])

    expectedDF = spark_session.read.format("csv")\
        .option("header", "true")\
        .schema(schema)\
        .load("./resources/expectedOutput/moving_average")

    assert type(actualDF) == pyspark.sql.dataframe.DataFrame
    assert type(actualDF.schema['stockId'].dataType) == pyspark.sql.types.IntegerType
    assert type(actualDF.schema['timeStamp'].dataType) == pyspark.sql.types.IntegerType
    assert type(actualDF.schema['stockPrice'].dataType) == pyspark.sql.types.DoubleType
    assert type(actualDF.schema['moving_average'].dataType) == pyspark.sql.types.DoubleType
    checkAnswer(actualDF,expectedDF)

# @pytest.fixture(scope="session")
def test_moving_average_with_stockInfo(spark_session):
    actualDF = src.MovingAverageWithStockInfo(spark_session, stockPriceInputDir,stockInfoInputDir, 3).calculate()

    schema = StructType([
        StructField("stockId", IntegerType()),
        StructField("timeStamp", IntegerType()),
        StructField("stockPrice", DoubleType()),
        StructField("moving_average", DoubleType()),
        StructField("stockName", StringType()),
        StructField("stockCategory", StringType())])

    expectedDF = spark_session.read.format("csv") \
        .option("header", "true") \
        .schema(schema) \
        .load("./resources/expectedOutput/moving_average_with_stockinfo")

    assert type(actualDF) == pyspark.sql.dataframe.DataFrame
    assert type(actualDF.schema['stockId'].dataType) == pyspark.sql.types.IntegerType
    assert type(actualDF.schema['timeStamp'].dataType) == pyspark.sql.types.IntegerType
    assert type(actualDF.schema['stockPrice'].dataType) == pyspark.sql.types.DoubleType
    assert type(actualDF.schema['moving_average'].dataType) == pyspark.sql.types.DoubleType
    assert type(actualDF.schema['stockName'].dataType) == pyspark.sql.types.StringType
    assert type(actualDF.schema['stockCategory'].dataType) == pyspark.sql.types.StringType

    checkAnswer(actualDF,expectedDF)

# @pytest.fixture(scope="session")
def test_moving_average_with_stockInfo_for_a_stock(spark_session):
    actualDF = src.MovingAverageWithStockInfo(spark_session, stockPriceInputDir,stockInfoInputDir, 3)\
        .calculate_for_a_stock("101")

    schema = StructType([
        StructField("stockId", IntegerType()),
        StructField("timeStamp", IntegerType()),
        StructField("stockPrice", DoubleType()),
        StructField("moving_average", DoubleType()),
        StructField("stockName", StringType()),
        StructField("stockCategory", StringType())])

    expectedDF = spark_session.read.format("csv") \
        .option("header", "true") \
        .schema(schema) \
        .load("./resources/expectedOutput/moving_average_with_stockinfo_for_a_stock")

    assert type(actualDF) == pyspark.sql.dataframe.DataFrame
    assert type(actualDF.schema['stockId'].dataType) == pyspark.sql.types.IntegerType
    assert type(actualDF.schema['timeStamp'].dataType) == pyspark.sql.types.IntegerType
    assert type(actualDF.schema['stockPrice'].dataType) == pyspark.sql.types.DoubleType
    assert type(actualDF.schema['moving_average'].dataType) == pyspark.sql.types.DoubleType
    assert type(actualDF.schema['stockName'].dataType) == pyspark.sql.types.StringType
    assert type(actualDF.schema['stockCategory'].dataType) == pyspark.sql.types.StringType

    checkAnswer(actualDF,expectedDF)

# @pytest.fixture(scope="session")
def test_moving_average_on_dirty_data(spark_session):
    actualDF = src.MovingAverage(spark_session, stockPriceDirtyInputDir, 3).calculate()

    schema = StructType([
        StructField("stockId", IntegerType()),
        StructField("timeStamp", IntegerType()),
        StructField("stockPrice", DoubleType()),
        StructField("moving_average", DoubleType())])

    expectedDF = spark_session.read.format("csv") \
        .option("header", "true") \
        .schema(schema) \
        .load("./resources/expectedOutput/moving_average_on_dirty_data")

    assert type(actualDF) == pyspark.sql.dataframe.DataFrame
    assert type(actualDF.schema['stockId'].dataType) == pyspark.sql.types.IntegerType
    assert type(actualDF.schema['timeStamp'].dataType) == pyspark.sql.types.IntegerType
    assert type(actualDF.schema['stockPrice'].dataType) == pyspark.sql.types.DoubleType
    assert type(actualDF.schema['moving_average'].dataType) == pyspark.sql.types.DoubleType
    checkAnswer(actualDF,expectedDF)




class TWTest(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSession.builder. \
        master("local[4]"). \
        appName("TWTest"). \
        getOrCreate()

    def test1(self):
        test_moving_average(spark_session=self.spark)
        # test_moving_average_with_stockInfo(spark_session=self.spark)

if __name__ == '__main__':
    unittest.main()

