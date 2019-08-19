package util

import org.apache.spark.sql.SparkSession

/**
  *
  * @Author Mike
  * @Email 597290963@qq.com
  * @Time 2019年8月19日14:26:43
  *
  */

object CreateSparkSessionUtil {
  def getSparkSessionObj(name: String): SparkSession = {

    val spark = SparkSession.builder().appName(name)
//      .config("spark.master", "local[*]")
      .config("spark.driver.memory", "4g")
      .config("spark.num-executors", "3")
      .config("spark.executor-memory", "4g")
      .config("spark.executor-cores", "2")
      .config("spark.default.parallelism", "30")
      .config("spark.storage.memoryFraction", "0.6")
      .config("spark.storage.safetyFraction", "0.9")
      .master("local[*]")
      .getOrCreate()

    // 设置日志等级
    spark.sparkContext.setLogLevel("ERROR")
    spark
  }
}
