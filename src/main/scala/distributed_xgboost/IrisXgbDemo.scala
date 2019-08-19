package distributed_xgboost

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import util.CreateSparkSessionUtil

/**
  *
  * @Author Mike
  * @Email 597290963@qq.com
  * @Time 2019年8月19日14:26:43
  *
  */
class IrisXgbDemo(sparkSession: SparkSession, input: String) {

  val spark: SparkSession = sparkSession
  val inputPath: String = input

  /**
    * 初始化模型所需数据
    *
    * @param spark
    * @return
    */
  def createData(): DataFrame = {
    // 创建schema
    val schema = new StructType(
      Array(
        StructField("sepal length", DoubleType, true),
        StructField("sepal width", DoubleType, true),
        StructField("petal length", DoubleType, true),
        StructField("petal width", DoubleType, true),
        StructField("class", StringType, true)
      )
    )

    val df = spark.read.schema(schema).csv(inputPath)
    df
  }

  /**
    * To convert String-typed label to Double
    *
    * @param df
    * @return
    */
  def stringIndexerDataFrame(df: DataFrame): DataFrame = {
    val stringIndexer = new StringIndexer()
      .setInputCol("class")
      .setOutputCol("classIndex")
      .fit(df)
    val labelTransformed = stringIndexer.transform(df).drop("class")
    labelTransformed
  }

  /**
    * To convert data frame to vector
    *
    * @param df
    */
  def transformDataFrameToVector(df: DataFrame): DataFrame = {
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("sepal length", "sepal width", "petal length", "petal width"))
      .setOutputCol("features")

    val xgbInput = vectorAssembler.transform(df).select("features", "classIndex")
    xgbInput
  }

  /**
    * xgb 训练
    *
    * @param df
    */
  def xgbTraining(df: DataFrame): XGBoostClassificationModel = {
    // 设置参数
    val xbgParam = Map(
      "eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "num_workers" -> 2
    )

    val xgbClassifier = new XGBoostClassifier(xbgParam)
      .setFeaturesCol("features")
      .setLabelCol("classIndex")

    // 另一种设置方式
    xgbClassifier.setMaxDepth(2)

    df.show()
    // training
    val xgbClassificationModel = xgbClassifier.fit(df)
    xgbClassificationModel
  }

  /**
    * 模型运行主类
    */
  def executeModel(): Unit = {
    val df = createData()
    val labelTransformed = stringIndexerDataFrame(df)
    val xgbInput = transformDataFrameToVector(labelTransformed)
    val model = xgbTraining(xgbInput)


    val result = model.transform(xgbInput)
    result.show()
  }
}

object IrisXgbDemo {
  def main(args: Array[String]): Unit = {
    // 初始化spark session
    val spark = CreateSparkSessionUtil.getSparkSessionObj("IrisXgbDemo")

    // 创建模型类对象
    val demo = new IrisXgbDemo(sparkSession = spark, input = "F:\\IDEAWorkSpace\\DistributedAlgorithmResearch\\data\\iris_demo\\iris.data")

    // 执行模型
    demo.executeModel()
  }
}
