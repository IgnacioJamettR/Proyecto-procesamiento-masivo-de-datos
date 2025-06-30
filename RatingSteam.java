package org.mdp.spark.cli;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import scala.Tuple2;
import scala.Tuple3;

public class RatingSteam {
	/**
	 * This will be called by spark
	 */
    public static void main(String[] args) {
        if(args.length != 2) {
			System.err.println("Usage arguments: inputPath outputPath");
            System.exit(0);
        }
        new RatingSteam().run(args[0], args[1]);
    }

    /**
	 * The task body
	 */
    public void run(String inputFilePath, String outputFilePath) {
		/*
		 * Initialises a Spark context with the name of the application
		 *   and the (default) master settings.
		 */        
        SparkConf conf = new SparkConf()
            .setAppName(RatingSteam.class.getName());
        JavaSparkContext context = new JavaSparkContext(conf);

        /*
		* Load the first RDD from the input location (a local file, HDFS file, etc.)
		 */
        JavaRDD<String> inputRDD = context.textFile(inputFilePath);
        
        JavaRDD<String> dataRDD = inputRDD.filter(line -> !line.contains("app_id"));

        // Parsea cada línea y calcula el score
        JavaPairRDD<String, Tuple3<String, Double, String>> reviewsRDD = dataRDD
	    .filter(line -> {
	        String[] tokens = line.split("\t", -1);
	        // Asegura que existan al menos 20 columnas
	        if (tokens.length < 20) {
	            return false;
	        }
	        return true;
	    })
	    .mapToPair(line -> {
	        String[] tokens = line.split("\t", -1);
	        String appId = tokens[2];
	        String reviewId = tokens[4];
	        String reviewText = tokens[6];
	        boolean recommended = Boolean.parseBoolean(tokens[9]);
	        int votesHelpful = Integer.parseInt(tokens[10]);
	        double playtime = Double.parseDouble(tokens[20]);

	        double alpha = 10;
	        double beta = 0.02;
	        double gamma = 1;
	        double score = recommended ? (alpha + beta * playtime + gamma * votesHelpful) : 0;

	        return new Tuple2<>(appId, new Tuple3<>(reviewId, score, reviewText));
	    });

        // Cachear pues se reutiliza
        reviewsRDD.cache();

        // por cada juego:
		//suma los scores
		//numero de reseñas
		//agrupa por app_id
        JavaPairRDD<String, Tuple2<Double, Integer>> sumCount = reviewsRDD.mapToPair(
	        t -> new Tuple2<>(t._1, t._2._2())
	    ).aggregateByKey(
	        new Tuple2<>(0.0,0),
	        (acc, s) -> new Tuple2<>(acc._1 + s, acc._2 +1),
	        (a,b) -> new Tuple2<>(a._1+b._1, a._2+b._2)
	    );

		//calcula el promedio
        JavaPairRDD<String, Tuple2<Double, Integer>> avgCount = sumCount.mapToPair(
            t -> new Tuple2<>(t._1, new Tuple2<>(t._2._1 / t._2._2, t._2._2))
        );
        
        // Toma el máximo de score de cada juego y agrupa por app_id
        JavaPairRDD<String, Double> maxScore = reviewsRDD.mapToPair(
            t -> new Tuple2<>(t._1, t._2._2())
        ).reduceByKey(Math::max);

		//juntar las partes
       // Juntar los agregados
       JavaPairRDD<String, Tuple2<Tuple2<Double, Integer>, Double>> joinedAggs = avgCount.join(maxScore);
        

       // Ahora join con cada reseña
       JavaPairRDD<String, Tuple2<Tuple3<String, Double, String>, Tuple2<Tuple2<Double, Integer>, Double>>> finalJoin = reviewsRDD.join(joinedAggs);

       // Mapeamos a una línea por reseña
       JavaRDD<String> outputRDD = finalJoin.map(t -> {
           String appId = t._1;
           Tuple3<String, Double, String> review = t._2._1;
           Tuple2<Double, Integer> avgAndCount = t._2._2._1;
           Double max = t._2._2._2;

           return review._1() + "\t" +                              // review_id
                  review._3().replace("\t"," ") + "\t" +            // review_text
                  appId + "\t" +                                    // app_id // no va
                  String.format("%.3f", review._2()) + "\t" +       // score individual	 -> el score calculado con la formula
                  String.format("%.3f", avgAndCount._1) + "\t" +    // avg_score -> es el promedio del score por el mismo juego
                  String.format("%.3f", max) + "\t" +               // score maximo por juego
                  avgAndCount._2;                                   // total de reviews	
       });

        /*
		 * Write the output to local FS or HDFS
		 */
        outputRDD.saveAsTextFile(outputFilePath);

        context.close();
    }
}
