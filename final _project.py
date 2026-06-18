from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, hour, when, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# 1. Start Spark Session
spark = SparkSession.builder \
    .appName("NYC_Taxi_Weather_Cleaned") \
    .getOrCreate()

# 2. Load Taxi Data
df_jan = spark.read.parquet(r"C:\Users\akanksha dhoundiyal\Desktop\pbl big data\yellow_tripdata_2024-01.parquet")
df_aug = spark.read.parquet(r"C:\Users\akanksha dhoundiyal\Desktop\pbl big data\yellow_tripdata_2024-08.parquet")
df_taxi = df_jan.union(df_aug)

# 3. Convert datetime & filter valid fares
df_taxi = df_taxi.withColumn("trip_date", to_date("tpep_pickup_datetime")) \
                 .withColumn("pickup_hour", hour("tpep_pickup_datetime")) \
                 .filter((col("fare_amount") > 0) & col("tpep_pickup_datetime").isNotNull())


# 4. Load Weather Data
weather_jan = spark.read.csv(r"C:\Users\akanksha dhoundiyal\Desktop\pbl big data\New York 2024-01-01 to 2024-01-31.csv", header=True, inferSchema=True)
weather_aug = spark.read.csv(r"C:\Users\akanksha dhoundiyal\Desktop\pbl big data\New York 2024-08-01 to 2024-08-31.csv", header=True, inferSchema=True)
weather_df = weather_jan.union(weather_aug)


# 5. Format date & select useful columns
weather_df = weather_df.withColumn("trip_date", to_date("datetime", "dd-MM-yyyy")) \
                       .select("trip_date", "temp", "feelslike", "humidity", "precip", "windspeed", "preciptype") \
                       .dropna(subset=["trip_date", "temp", "humidity", "precip", "windspeed"])

# 6. Join taxi and weather data
df_combined = df_taxi.join(weather_df, on="trip_date", how="inner")


# 7. Drop rows with any nulls in essential columns
df_combined = df_combined.dropna(subset=["fare_amount", "temp", "humidity", "precip", "windspeed"])


# 8. Remove duplicates (optional)
df_combined = df_combined.dropDuplicates()


# 9. Bucket temperature for analysis
df_combined = df_combined.withColumn("temp_bucket", when(col("temp") < 30, "Cold")
                                                  .when(col("temp") < 60, "Mild")
                                                  .otherwise("Warm"))


# 10. Basic fare analysis
df_combined.groupBy("preciptype").avg("fare_amount").show()
df_combined.groupBy("temp_bucket").avg("fare_amount").show()


# 11. Correlation Matrix
features_df = df_combined.select("temp", "humidity", "precip", "windspeed", "fare_amount")
assembler = VectorAssembler(inputCols=features_df.columns, outputCol="features")
assembled = assembler.transform(features_df).select("features")


correlation_matrix = Correlation.corr(assembled, "features").head()
print("Correlation Matrix:\n", correlation_matrix[0])


# 12. Save to CSV
df_combined.coalesce(1).write.mode("overwrite").option("header", True).csv(r"C:\Users\akanksha dhoundiyal\Desktop\pbl big data\output_cleaned")
