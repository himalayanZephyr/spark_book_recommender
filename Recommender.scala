import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.log4j.{LogManager,Level}
import org.apache.spark.sql.SparkSession



object Recommender{
    def main(args: Array[String]){
        
        val log = LogManager.getRootLogger
        log.setLevel(Level.INFO)
        
         //Create connection
        val conf = new SparkConf()
            .setMaster("local[*]")
            .setAppName("BookRecommender")
            .set("spark.executor.memory", "4g")
        val sc = new SparkContext(conf)
       
        
        // Path to dataset
        val dataDir = "./data"
        
        // Get the ratings data
        val ratingsData = sc.textFile(dataDir+"/ratings.csv")
        val ratings = ratingsData.map(_.split(",") match { 
            case Array(userId, bookId, rating) => Rating(userId.toInt, bookId.toInt, rating.toDouble)
        })

        ratings.take(1).foreach(println)
        
        /* Train the ALS model with rank=50, iterations=10, lambda=0.01 */
        val model = ALS.train(ratings, 50, 10, 0.01)
        
        //println("======== User features ==============")
        
        //model.userFeatures.take(1).foreach(println)
        
        //println("======== Product features ===========")
        
        //model.productFeatures.take(1).foreach(println)
        
        // SAMPLE BOOK : 3753,10,10,21457570,6,439827604,9780439827610,J.K. Rowling,2005,"Harry Potter Collection (Harry Potter, #1-6)","Harry Potter Collection (Harry Potter, #1-6)",eng,4.73,24618,26274,882,203,186,946,3891,21048,https://images.gr-assets.com/books/1328867351m/10.jpg,https://images.gr-assets.com/books/1328867351s/10.jpg
        //35,865,865,4835472,458,61122416,9780061122420,"Paulo Coelho, Alan R. Clarke",1988,O Alquimista,The Alchemist,eng,3.82,1299566,1403995,55781,74846,123614,289143,412180,504212,https://images.gr-assets.com/books/1483412266m/865.jpg,https://images.gr-assets.com/books/1483412266s/865.jpg
        //19,34,34,3204327,566,618346252,9780618346260,J.R.R. Tolkien,1954, The Fellowship of the Ring,"The Fellowship of the Ring (The Lord of the Rings, #1)",eng,4.34,1766803,1832541,15333,38031,55862,202332,493922,1042394,https://images.gr-assets.com/books/1298411339m/34.jpg,https://images.gr-assets.com/books/1298411339s/34.jpg


        
        val sampleUserId = 100
        val sampleBookId = 3753
        
        val predictedRating = model.predict(sampleUserId,sampleBookId)
        println(s"Predicted rating is $predictedRating")
        log.info("Bla Bla")
        
        val k = 10
        val topKRecs = model.recommendProducts(sampleUserId, k)
        topKRecs.foreach(println)
        
        
                 //Fields in books.csv  book_id,goodreads_book_id,best_book_id,work_id,books_count,isbn,isbn13,authors,original_publication_year,original_title,title,language_code,average_rating,ratings_count,work_ratings_count,work_text_reviews_count,ratings_1,ratings_2,ratings_3,ratings_4,ratings_5,image_url,small_image_url

        //Checking the titles
        val sparkSession = SparkSession.builder
          .master("local")
          .appName("app")
          .getOrCreate()
        
        val booksDataInitial = sparkSession.read.format("csv")
                        .option("header", "false")
                        .load(dataDir+"/books.csv").rdd//sc.textFile(dataDir+"/books.csv")
        val booksData = booksDataInitial.filter(x => (x.getString(0)!= null))
        
        val titlesMap = booksData.map(row => (row.getString(0).toInt, row.getString(9))).collectAsMap()//booksData.map(_.split(",")).map(array=> (array(0).toInt, array(9))).collectAsMap()
        
        
        //val titlesMap = titlesMap1.collectAsMap()
        //for ((k,v) <- titlesMap) println(k.toString)
        titlesMap.foreach {
        case(id, title) => println(s"key: $id, value: $title")
        }
        
        println("Title for book id %d is %s".format(sampleBookId,titlesMap(sampleBookId)))
        
        //Getting books rated by sample user
        val sampleUserRatedBooks = ratings.keyBy(_.user).lookup(sampleUserId)
        
        println("Total books rated by user id %d is %d".format(sampleUserId, sampleUserRatedBooks.size))
        
        sampleUserRatedBooks.sortBy(-_.rating).take(k).map(rating => (titlesMap(rating.product),rating.rating)).foreach(println)
        
        //Checking the top5 recs for sample user
        topKRecs.map(rating => (titlesMap(rating.product), rating.rating)).foreach(println)
        println("The title is %s".format(titlesMap(975)))
        
       
        
        
        // Finding books similar to a given book
        val sampleBookFeatures = model.productFeatures.lookup(sampleBookId).head // Without head it gives a Wrapped Seq and we just want Seq
        
        println("==============SIMILARITY WITH ITSELF===========")
        println(cosineSimilarity(Vectors.dense(sampleBookFeatures), Vectors.dense(sampleBookFeatures)))
        println("==================================================")
        
        // Similarity of sample book with all 
        /*val sims = model.productFeatures.map{ case(id, featureSeq) =>
            val zippedSeq = featureSeq zip sampleBookFeatures
            val rows = sc.parallelize(zippedSeq)//(featureSeq zip sampleBookSeq)
            //val sim = cosineSimilarity(zipT)//cosineSimilarity(sc, featureSeq, sampleBookFeatures)
            //val vecRows = rows.map{case(u,v) => Vectors.dense(u,v)}
            //val mat = new RowMatrix(vecRows)
            val sim = 1.0//mat.columnSimilarities.entries.take(1)(0).value
            (id, sim)
        }
        
        val zipped = model.productFeatures.map{
            case(id, featureSeq) => (featureSeq zip sampleBookFeatures).toSeq
        }
        
        val sims = zipped.map{case(u) =>
            val sim = cosineSimilarity(u)
            println(sim)
            (sim)
        }
        
        val sortedSims = sims.top(k)(Ordering.by[(Int, Double), Double]{case (id, similarity) => similarity})
        
        println(sortedSims.mkString("\n"))*/
        
        val sims = model.productFeatures.map{ case(id, featureSeq) =>
            val sim = cosineSimilarity(Vectors.dense(featureSeq), Vectors.dense(sampleBookFeatures))
            (titlesMap(id), sim)
        }
        
        val topKSimilarBooks = sims.top(k)(Ordering.by[(String, Double), Double]{case (id, similarity) => similarity})
        
        println(topKSimilarBooks.mkString("\n"))
        
        // Printing by titles instead
        //topKSimilarBooks.map{case(id,sim) => (titlesMap(id), sim)}.foreach(println)
        
        // Sample MSE : Computing squared error between actual and predicted rating
        val actualRating = sampleUserRatedBooks.take(1)(0)
        val predictedR = model.predict(sampleUserId, actualRating.product)
        val squaredError = math.pow(predictedR - actualRating.rating, 2.0)
        
        
        // Evaluate the model on rating data
        val usersProducts = ratings.map { case Rating(user, product, rate) =>(user, product)}

        val predictions = model.predict(usersProducts).map { 
            case Rating(user, product, rate) => ((user, product), rate)
        }

        val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
            ((user, product), rate)
        }.join(predictions)

        val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
            val err = (r1 - r2)
            err * err
        }.mean()

        println(s"Mean Squared Error = $MSE")
        
        val predictedAndTrue = ratesAndPreds.map { 
            case ((user, product), (actual, predicted)) => (actual, predicted) 
        }
        
        val regressionMetrics = new RegressionMetrics(predictedAndTrue)
        println("Mean Squared Error = " + regressionMetrics.meanSquaredError)
        println("Root Mean Squared Error = " + regressionMetrics.rootMeanSquaredError)
        
        
        println("==========================================================================================")
        println("===================== OUTPUT START =======================================================")
        println("\n")
        println("Title for book id %d is %s".format(sampleBookId,titlesMap(sampleBookId)))
        println("\n")
        println("Total books rated by user id %d is %d".format(sampleUserId, sampleUserRatedBooks.size))
        println("\n")
        println(s"Predicted rating for chosen book and user id is $predictedRating")
        println("\n")
        println("Top K recommendations for user are :")
        topKRecs.map(rating => (titlesMap(rating.product), rating.rating)).foreach(println)
        println("\n")
        println("Top K books similar to chosen book are :")
        println(topKSimilarBooks.mkString("\n"))
        println("\n")
        println("===================== OUTPUT END =========================================================")
        println("==========================================================================================")
        sc.stop()
    }
    
    /*def cosineSimilarity(rows:Seq[(Double,Double)]):Double = {//(sc: SparkContext, seq1: Seq[Double], seq2: Seq[Double]): Double = {
        //val rows = sc.parallelize(seq1 zip seq2)
        val vecRows = rows.map{case(u,v) => Vectors.dense(u,v)}
        val mat = new RowMatrix(vecRows)
        val sim = mat.columnSimilarities.entries.take(1)(0).value
        return sim
    }*/ 
    def cosineSimilarity(vectorA: Vector, vectorB: Vector) = {

        var dotProduct = 0.0
        var normA = 0.0
        var normB = 0.0
        var index = vectorA.size - 1

        for (i <- 0 to index) {
            dotProduct += vectorA(i) * vectorB(i)
            normA += Math.pow(vectorA(i), 2)
            normB += Math.pow(vectorB(i), 2)
        }
        (dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)))
    }
    
}