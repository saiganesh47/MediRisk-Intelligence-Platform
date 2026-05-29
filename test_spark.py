from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("test") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

schema = StructType([
    StructField("name", StringType(), True),
    StructField("id", IntegerType(), True)
])

data = [("Alice", 1), ("Bob", 2), ("Charlie", 3)]
df = spark.createDataFrame(data, schema)
df.show()
print("PySpark is working!")
spark.stop()