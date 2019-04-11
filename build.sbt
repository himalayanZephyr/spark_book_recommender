name := "Recommender"

version := "1.0"

scalaVersion := "2.11.8"

val sparkVersion = "2.3.1"


libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion
)