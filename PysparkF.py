from __future__ import print_function

from pyspark.sql import SparkSession
import sys

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: PysparkF.py <filein> <fileout>", file=sys.stderr)
        sys.exit(-1)

    filein = sys.argv[1]
    fileout = sys.argv[2]

    # in pyspark shell start with:
    #   filein = "hdfs://cm:9000/uhadoop2025/projects/grupo6/steamDataF2.tsv"
    #   fileout = "hdfs://cm:9000/uhadoop2025/projects/grupo6/IJ2/"
    # and continue line-by-line from here

    spark = SparkSession.builder.appName("PysparkF").getOrCreate()
    input = spark.read.text(filein).rdd.map(lambda r: r[0])
    #filter en caso de errores de limpieza imprevistos
    lines = input.map(lambda line: line.split("\t")).filter(lambda row: len(row) == 24)

    #se combierten las filas a diccionarios.
    header=lines.first()
    data = lines.filter(lambda row: row != header)
    data = data.map(lambda row: dict(zip(header, row)))

    #linea de comando para testing de limpieza de datos, porfavor ignorar
    #test=lines.map(lambda row: len(row))

    #Se calcula el maximo por videojuego para cantidad de juegos que tiene el usuario y para cantidad de criticas que ha escrito el usuario
    owned_max = data.map(lambda row: (row["app_name"], int(row["author.num_games_owned"]))).reduceByKey(lambda a, b: max(a, b))
    reviewed_max = data.map(lambda row: (row["app_name"], int(row["author.num_reviews"]))).reduceByKey(lambda a, b: max(a, b))

    #Se utiliza collect para extraer la data
    owned_max_dict = dict(owned_max.collect())
    reviewed_max_dict = dict(reviewed_max.collect())

    #Se agrega la data extraída, gracias a que son diccionarios, utilizan el nombre del juego para esto
    lines2 = data.map(lambda row: {
        **row,
        'maxowned': int(row['author.num_games_owned']) / owned_max_dict.get(row['app_name'], 1.0),
        'maxreviewd': int(row['author.num_reviews']) / reviewed_max_dict.get(row['app_name'], 1.0)
    })

    #Se crea una función que asigna etiquetas dependiendo de el rango en el que se encuentra el valor,
    #esto debido a que se van a usar estas etiquetas como una cualidad que puede variar en el perfil de usuario
    def bin(value):
        if value <= 0.25:
            return '1'
        elif value <= 0.5:
            return '2'
        elif value <= 0.75:
            return '3'
        else:
            return '4'

    #Se utiliza la función para producir y agragar las columnas con las etiquetas
    lines3 = lines2.map(lambda row: {
        **row,
        'ownedClass': bin(row['maxowned']),
        'reviewedClass': bin(row['maxreviewd']),
    })

    #Se combinan las 2 etiquetas para hacer una columna que define por completo la "Clase" del autor
    #Ej.: 41 es alguien con muchos juegos que comenta poco
    lines4 = lines3.map(lambda row: {
        **row,
        'authorClass': row['ownedClass'] + row['reviewedClass']
    })

    #Esta función reemplaza valores bool por etiquetas más descriptivas
    def map_bool(val, mapping):
        return mapping.get(bool(val), str(val))

    #En este paso se crea la columna profile que define un perfil de tipo de comentador, originalmente se incluia app_name,
    #pero se removió para facilitar la comparación de comenatrios de un mismo perfil a lo largo de distintas aplicaciones
    lines5 = lines4.map(lambda row: {
        **row,
        'profile': #row['app_name'] + '|' +
                   row['language'] + '|' +
                   map_bool(row['recommended'], {True: 'Positive', False: 'Negative'}) + '|' +
                   row['authorClass'] + '|' +
                   map_bool(row['steam_purchase'], {True: 'Steam', False: 'Not Steam'}) + '|' +
                   map_bool(row['received_for_free'], {True: 'Purchased', False: 'Free'})
    })

    lines5.cache()

    #Se calcula y guarda el maximo y minimo de "timestamp_updated" para luego usarlo al normalizar, esta vez se requiere el minimo
    #debido a que este es probablemente mucho mayor a cero para la mayoría de aplicaciones
    timestamp_minmax = lines5.map(lambda row: (row['app_name'], float(row['timestamp_updated']))) \
        .aggregateByKey((float('inf'), float('-inf')),
                        lambda acc, v: (min(acc[0], v), max(acc[1], v)),
                        lambda a, b: (min(a[0], b[0]), max(a[1], b[1])))
    timestamp_minmax_dict = dict(timestamp_minmax.collect())

    #Se calcula el maximo de comment_count para normalizar, el minimo para este dato es cero
    comment_max = lines5.map(lambda row: (row['app_name'], int(row['comment_count']))).reduceByKey(lambda a, b: max(a, b))
    comment_max_dict = dict(comment_max.collect())

    # Se calcula y guarda el maximo y minimo de "author.playtime_at_review" para luego usarlo al normalizar, esta vez se requiere el minimo
    # debido a que este es probablemente mucho mayor a cero para la mayoría de aplicaciones
    playtime_minmax = lines5.map(lambda row: (row['app_name'], float(row['author.playtime_at_review']))) \
        .aggregateByKey((float('inf'), float('-inf')),
                        lambda acc, v: (min(acc[0], v), max(acc[1], v)),
                        lambda a, b: (min(a[0], b[0]), max(a[1], b[1])))
    playtime_minmax_dict = dict(playtime_minmax.collect())

    #funcion que normaliza cuando hay minimo
    def normalize(value, min_val, max_val):
        if max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)

    #Se agregan las columnas normalizadas a la tabla
    lines6 = lines5.map(lambda row: {
        **row,
        'timestampN': normalize(
            float(row['timestamp_updated']),
            timestamp_minmax_dict[row['app_name']][0],
            timestamp_minmax_dict[row['app_name']][1]
        ),
        'comment_countN': int(row['comment_count']) / comment_max_dict.get(row['app_name'], 1.0),
        'playtimeN': normalize(
            float(row['author.playtime_at_review']),
            playtime_minmax_dict[row['app_name']][0],
            playtime_minmax_dict[row['app_name']][1]
        )
    })

    #Se calcula y agrega la relevancia como un promedio ponderado de los 3 valores calculados y el "weighted_vote_score"
    lines7 = lines6.map(lambda row: {
        **row,
        'relevancy': 0.1 * float(row['timestampN']) +
                     0.3 * float(row['weighted_vote_score']) +
                     0.3 * float(row['comment_countN']) +
                     0.3 * float(row['playtimeN'])
    })

    #Se obtiene el promedio de la relevancia de cada perfil
    profile_avg = lines7.map(lambda row: (row['profile'], (float(row['relevancy']), 1))) \
                     .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])) \
                     .mapValues(lambda v: v[0]/v[1])
    profile_avg_dict = dict(profile_avg.collect())

    #Se agrega la relevancia promedio
    linesF = lines7.map(lambda row: {
        **row,
        'profileRelevancy': profile_avg_dict.get(row['profile'], 0.0)
    })

    linesF2 = linesF.map(lambda row: {
        row['data_name'],
        row['review_id'],
        row['profile'],
        row['relevancy'],
        row['profileRelevancy']
    })

    #Se guarda como fileout
    linesF2.saveAsTextFile(fileout)

    spark.stop()