import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.clustering.KMeans
import org.apache.log4j.{LogManager,Level}
import org.apache.spark.sql.SparkSession

object Clustering{
    def main(args: Array[String]){
        
        val log = LogManager.getRootLogger
        log.setLevel(Level.WARN)
        
         //Create connection
        val conf = new SparkConf()
            .setMaster("local[*]")
            .setAppName("BookRecommender")
            .set("spark.executor.memory", "4g")
        val sc = new SparkContext(conf)
        
        // Path to dataset
        val dataDir = "./data"
        
        // Get the books data
        // COLUMNS:      book_id,goodreads_book_id,best_book_id,work_id,books_count,isbn,isbn13,authors,original_publication_year,original_title,title,language_code,average_rating,ratings_count,work_ratings_count,work_text_reviews_count,ratings_1,ratings_2,ratings_3,ratings_4,ratings_5,image_url,small_image_url

        val sparkSession = SparkSession.builder
          .master("local")
          .appName("app")
          .getOrCreate()
        
        val booksDataInitial = sparkSession.read.format("csv")
                        .option("header", "false")
                        .load(dataDir+"/books.csv").rdd
        
        val booksData = booksDataInitial.filter(x => (x.getString(0)!= null && x.getString(9) != null))
        //val booksData = sc.textFile(dataDir+"/books.csv")
        //println(booksData.first) //PRINT
        val titlesMap = booksData.map(row => (row.getString(0).toInt, row.getString(9))).collectAsMap()
        
        // Get the goodreads tags
        // COLUMNS: tag_id,tag_name
        
        val tagsData = sc.textFile(dataDir+"/tags.csv")
        //tagsData.take(5).foreach(println) //PRINT
        
        // Generating a map of tags
        val tagsMap = tagsData.map(line=> line.split(",")).map(array => (array(0).toInt, array(1))).collectAsMap
        
        //print(tagsMap) //PRINT
        
        // Getting bookids and tags associated with them
        // COLUMNS: goodreads_book_id,tag_id,count

        val bookIdsTags = sc.textFile(dataDir+"/book_tags.csv")
        
        val booksTagsMap = bookIdsTags.map(line => line.split(",")).map(array => (array(0).toInt, array(1).toInt)).groupByKey().mapValues(_.toSeq).collectAsMap
        
        /* print(booksTagsMap) //PRINT
        
        println("================ Tags for Book =======================")
        for(t <- booksTagsMap(1)){
            println(t)
        }
        println("======================================================")*/
        
        val titlesAndTags = booksData.map{ array =>
            val goodReadsId = array.getString(1).toInt
            val tagIds = booksTagsMap(goodReadsId)
            val assignedTags = tagIds.map(x => tagsMap(x))
            (array.getString(0).toInt, (array.getString(9), assignedTags))
        }
        
        // println(titlesAndTags.first) //PRINT
        
        // =========== Generate user and book features ==================
        
        // Get the ratings data
        val ratingsData = sc.textFile(dataDir+"/ratings.csv")
        val ratings = ratingsData.map(_.split(",") match { 
            case Array(userId, bookId, rating) => Rating(userId.toInt, bookId.toInt, rating.toDouble)
        })
        
        val model = ALS.train(ratings, 50, 10, 0.1)
        
        // Get factor vectors
        val bookFactors = model.productFeatures.mapValues(x => Vectors.dense(x)).map(_._2).cache()
        val userFactors = model.userFeatures.mapValues(x => Vectors.dense(x)).map(_._2)
        
        /*println("================ View features ================")
        println(bookFactors.first)
        println(userFactors.first)
        println("===============================================")*/
        
        // Investigating distribution of features //TODO: Plot to view visually
        val bookMatrix = new RowMatrix(bookFactors)
        val bookMatrixSummary = bookMatrix.computeColumnSummaryStatistics()
        val userMatrix = new RowMatrix(userFactors)
        val userMatrixSummary = userMatrix.computeColumnSummaryStatistics()
        println("Book factors mean: " + bookMatrixSummary.mean)
        println("Book factors variance: " + bookMatrixSummary.variance)
        println("User factors mean: " + userMatrixSummary.mean)
        println("User factors variance: " + userMatrixSummary.variance)
        //Seems no outliers
        
        // running K-means model on book factor vectors
        val numClusters = 10 // Using 5 after cross-validation #TODO: plot curve(check elbow)
        val numIterations = 50
        var bookClusterModel = KMeans.train(bookFactors, numClusters, numIterations)
        
        
        // Evaluate clustering by computing Within Set Sum of Squared Errors
        val WSSSE = bookClusterModel.computeCost(bookFactors)
        println(s"Within Set Sum of Squared Errors = $WSSSE")
        
        // Not sure if this is the best value of K, so trying cross-validation
        /*val trainTestSplit = bookFactors.randomSplit(Array(0.8, 0.2), seed = 12345)
        val trainingBookData = trainTestSplit(0)
        val testBookData = trainTestSplit(1)
        
        val costsBooks = Seq(2,3,5,10,20).map{ k => (k, KMeans.train(trainingBookData, k, numIterations).computeCost(testBookData))}
        
        println("======== Book Clustering Cross Validation Results =====")
        
        costsBooks.foreach{ case(k, cost) => println(s"WCSS for K=$k is $cost")}*/
        
        /*
        ======== Book Clustering Cross Validation Results =====

        WCSS for K=2 is 1993.2469974371381
        WCSS for K=3 is 1947.105690213165
        WCSS for K=5 is 1888.1340219025037
        WCSS for K=10 is 1818.4268052367956
        WCSS for K=20 is 1758.8531074740106

        */
        
        // Predicting a book cluster given a bookId
        val sampleBookId = 3753
        val sampleBookFeatures = model.productFeatures.lookup(sampleBookId).head
        val sampleBookVec = Vectors.dense(sampleBookFeatures)
        
        val sampleBookCluster = bookClusterModel.predict(sampleBookVec)
        println(s"Cluster of book id $sampleBookId is $sampleBookCluster")
        
        // Printing some books in same cluster sorted by distance from cluster center
        val allBookFeatures = model.productFeatures.map{case(id, factor) => (id,Vectors.dense(factor))}
        val titlesTagsFeatures = titlesAndTags.join(allBookFeatures)
        // JOIN : (id, ((title, tags), features))
        //println(titlesTagsFeatures.first) //PRINT
        
        val titlesTagsFeatDists = titlesTagsFeatures.map { case(id,((title,tags),vector)) =>
            val predictedCluster = bookClusterModel.predict(vector)
            val clusterCentre = bookClusterModel.clusterCenters(predictedCluster)
            val dist = Vectors.sqdist(vector,clusterCentre)
            (id, title, tags, predictedCluster, dist)
        }
        
        val clusterAssignments = titlesTagsFeatDists.groupBy{ 
            case(id, title, tags, pred, dist) => pred}.collectAsMap
        
        println(titlesTagsFeatDists.first)
        
        val sampleBookClusterMembers = clusterAssignments(sampleBookCluster)
        val sortedMembers = sampleBookClusterMembers.toSeq.sortBy(_._5)
        val topK = sortedMembers.take(10).map{case(id, title, tags, _ , dist)=>(title, tags, dist)}
        
        //println("=========== BOOKS IN SAME CLUSTER ===============")
        //println(topK.mkString("\n"))
        //println("=================================================")
        
        
        println("==========================================================================================")
        println("===================== OUTPUT START =======================================================")
        println("\n")
        println("Title for book id %d is %s".format(sampleBookId, titlesMap(sampleBookId)))
        println("\n")
        println(s"Cluster of book id $sampleBookId is $sampleBookCluster\n")
        println("\n")
        println("Books in the same cluster: <Book, List(Tags), Distance from Cluster Centre>")
        println(topK.mkString("\n\n"))
        println("\n")
        println("===================== OUTPUT END =========================================================")
        println("==========================================================================================")
        
        sc.stop()
    }
    
    
}