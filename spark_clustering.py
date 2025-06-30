import argparse
import gensim
import os
import sys

import gensim.downloader as api
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import PCA, Word2Vec
from pyspark.ml.linalg import Vectors, VectorUDT

from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess


MODEL_NAME = 'glove-twitter-25'
SAVE_OUTPUT = True
OUTPUT_PATH = "/uhadoop2025/projects/grupo6/reviews_cluster_ids"


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--limit", type=int, default=None)
args = parser.parse_args()

if args.limit:
    OUTPUT_PATH = OUTPUT_PATH + "_sample"

os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-11-openjdk-amd64"

hdfs_path = "/uhadoop2025/projects/grupo6/steam_reviews.csv"
# hdfs_path = "/uhadoop2025/projects/grupo6/pruebaSpark.csv"

# spark = SparkSession.builder.appName("SteamReviews").getOrCreate()
spark = (
    SparkSession.builder
    .appName("SteamReviews")
    .config("spark.executor.memory", "8g")
    .config("spark.driver.memory", "4g")
    .config("spark.memory.fraction", "0.7")
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
    .getOrCreate()
)

# Leer el archivo CSV desde HDFS
try:
    df_reader = spark.read.csv(hdfs_path, header=True, inferSchema=True)

    if args.limit:
        df = df_reader.limit(1000000)
        print("Cargando las primeras 1000000 filas del DataFrame.")
    else:
        df = df_reader
        print("Cargando el DataFrame completo.")

    print(f"Número total de filas cargadas: {df.count()}")
except Exception as e:
    print(f"Error al leer el CSV desde HDFS: {e}")
    spark.stop()
    exit()

# Limitar dataset
if args.limit:
    df = df.limit(args.limit)

# Filtrar reviews en ingles
df_en = df.filter((df.language == "english") & (df.recommended == True))
print("\nFiltered English Recommended Reviews:")
df_en.show(5, truncate=False)

# Correjir errores de parseo
df_en = df_en.withColumn("review", regexp_replace("review", "[,\t]", " "))

# Eliminar nulos en review_id
df_en = df_en.filter(col("review_id").isNotNull())

df_en = df_en.cache()


# Función Python que tokeniza texto
def tokenize_text(text):
    if text:
        # simple_preprocess is good for basic tokenization
        return simple_preprocess(text, deacc=True)
    return []


# Registrar UDF
tokenize_udf = udf(tokenize_text, ArrayType(StringType()))

# Aplicar a la columna 'review'
df_tok = df_en.withColumn(
    "tokens", tokenize_udf(df_en["review"])
).select("review_id", "review", "tokens").cache()

# Mostrar results
print("\nDataFrame with Tokens:")
df_tok.select("review", "tokens").show(5, truncate=False)

# Word2Vec
print("\nApplying Word2Vec:")

print(f"\nLoading pre-trained Word2Vec model '{MODEL_NAME}' from the internet")

pretrained_model = api.load(MODEL_NAME)
VECTOR_SIZE = pretrained_model.vector_size
broadcast_model = spark.sparkContext.broadcast(pretrained_model)

print(f"Modelo pre-entrenado '{MODEL_NAME}' cargado y distribuido.")
print(f"Tamaño del vector: {VECTOR_SIZE}")


# Función de embedding
def get_avg_word_vector(tokens):
    model_wv = broadcast_model.value
    valid_tokens = [word for word in tokens if word in model_wv.key_to_index]
    if valid_tokens:
        vectors = [model_wv[word] for word in valid_tokens]
        avg_vector = np.mean(vectors, axis=0)
        return Vectors.dense(avg_vector)
    else:
        return Vectors.dense([0.0] * VECTOR_SIZE)


get_avg_word_vector_udf = udf(get_avg_word_vector, VectorUDT())

df_w2v = df_tok.withColumn(
    "w2v_features",
    get_avg_word_vector_udf("tokens")
).select("review_id", "review", "w2v_features")

print("\nDataFrame with Pre-trained Word2Vec Features:")
df_w2v.show(5, truncate=False)

# K-Means en Word2Vec y Evaluación
print("\nApplying K-Means Clustering on Word2Vec Features")
df_w2v_repartitioned = df_w2v.repartition(
    spark.sparkContext.defaultParallelism
)

kmeans_w2v = BisectingKMeans(featuresCol="w2v_features", k=2, seed=1)
kmeans_w2v_model = kmeans_w2v.fit(df_w2v_repartitioned)
df_clustered_w2v = kmeans_w2v_model.transform(df_w2v)

print("DataFrame con predicciones de clúster (Word2Vec):")
df_clustered_w2v.select("review", "prediction").show(5, truncate=False)

# Evaluar el clustering en Word2Vec
evaluator_w2v = ClusteringEvaluator(featuresCol="w2v_features")
silhouette_w2v = evaluator_w2v.evaluate(df_clustered_w2v)
print(f"\nCoeficiente de silueta para el clustering en Word2Vec = {silhouette_w2v}")

print("\nAdding Cluster ID to the Original DataFrame (using Word2Vec clusters)")
# Unir el DataFrame original (df) con los resultados del clustering de Word2Vec
df_w2v_for_join = df_clustered_w2v.select("review_id", "prediction")
df_with_cluster_id = df.join(
    df_w2v_for_join.select("review_id", "prediction"),
    on="review_id",
    how="left"
).withColumnRenamed("prediction", "cluster_id")

print("\nDataFrame original con la nueva columna 'cluster_id':")
df_with_cluster_id.select("review_id", "review", "cluster_id").show(10, truncate=False)
print("\nSe ha añadido la columna 'cluster_id' al DataFrame original.")

# Saving the Final DataFrame to HDFS as CSV
if SAVE_OUTPUT:
    print(f"\nSaving the selected columns to CSV in HDFS at: {OUTPUT_PATH}")

    df_to_save = df_with_cluster_id.select(
        "review_id", "cluster_id"
    )
    df_to_save.write.mode("overwrite").csv(OUTPUT_PATH, header=True)

    print("\nFile saved successfully!")

spark.stop()
