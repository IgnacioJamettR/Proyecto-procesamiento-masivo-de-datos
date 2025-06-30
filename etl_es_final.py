import argparse
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


path1 = "/uhadoop2025/projects/grupo6/ratingSteam"
path2 = "/uhadoop2025/projects/grupo6/reviews_cluster_ids"

cols1 = [
    "Review_id", "Review_text", "App_id", "Score_individual", "Avg_score",
    "Max_score", "Total_reviews"
]
cols1_rename = {
    "Review_id": "review_id",
    "Review_text": "review_text",
    "App_id": "app_id",
    "Score_individual": "review_score",
    "Avg_score": "app_avg_score",
    "Max_score": "app_max_score",
    "Total_reviews": "total_reviews"
}

spark = (
    SparkSession.builder
    .appName("ETL-HDFS-to-ES")
    .getOrCreate()
)

print(f"\nCargando el dataset de reviews desde: {path1}")
df_reviews = spark.read.csv(
    path1,
    header=True,
    inferSchema=True,
    sep="\t"
)
df_reviews = df_reviews.toDF(*cols1)
for old_col, new_col in cols1_rename.items():
    df_reviews = df_reviews.withColumnRenamed(old_col, new_col)

df_reviews = df_reviews.filter(col("review_id").isNotNull())
print(f"\nNúmero de filas no nulas: {df_reviews.count()}")

print(f"\nCargando el dataset de clústeres desde: {path2} ---")
df_clusters = spark.read.csv(
    path2,
    header=True,
    inferSchema=True
).select("review_id", "cluster_id")

df_clusters = df_clusters.filter(
    col("cluster_id").isNotNull() & col("review_id").isNotNull())
print(f"\nNúmero de filas no nulas: {df_clusters.count()}")

print("\nRealizando el JOIN de los DataFrames por 'review_id'")
df_joined = df_reviews.join(
    df_clusters,
    on="review_id",
    how="inner"
).limit(1000000)

# Subir a ElasticSearch
es_nodes = "cm"
es_port = "9200"
es_index = "steam_reviews"
es_type = "review"

(df_joined.write
    .format("org.elasticsearch.spark.sql")
    .option("es.nodes", es_nodes)
    .option("es.port", es_port)
    .option("es.resource", f"{es_index}/{es_type}")
    .option("es.mapping.id", "review_id")
    .option("es.mapping.type", "_doc")
    .mode("overwrite")
    .save())

spark.stop()
