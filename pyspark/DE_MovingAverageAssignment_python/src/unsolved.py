import sys
from pyspark.sql.window import Window
from pyspark.sql.functions import *
from pyspark.sql.types import *

class MovingAverage:

    def __init__(self, spark, stockPriceInputDir, size):
        self.spark = spark
        self.stockPriceInputDir = stockPriceInputDir
        self.size = size

    def calculate(self):
        schema = StructType([
            StructField("stockId", IntegerType()),
            StructField("timeStamp", IntegerType()),
            StructField("stockPrice", DoubleType())])

        df = self.spark.read.option("header", True).schema(schema).csv(self.stockPriceInputDir)
        # df.show()
        print(df.rdd.getNumPartitions())

        df = df.filter(df["timeStamp"] >= 3)
        window = Window.partitionBy("stockId").orderBy('stockId','timeStamp').rowsBetween(-1, 1)
        res_df: DataFrame = df.withColumn("moving_average", avg(df['stockPrice']).over(window))
        # win_customers01.show(10, truncate=False)
        print(res_df.rdd.getNumPartitions())
        # res_df.write.csv("/tmp/tw/")
        res_df.explain()
        return res_df

class MovingAverageWithStockInfo:
    spark = None
    stockPriceInputDir = None
    stockInfoInputDir = None
    size = 0
    def __init__(self, spark, stockPriceInputDir, stockInfoInputDir, size):
        self.spark = spark
        self.stockPriceInputDir = stockPriceInputDir
        self.stockInfoInputDir = stockInfoInputDir
        self.size = size

    def calculate(self):
        pass

    def calculate_for_a_stock(self,stockId):
        pass