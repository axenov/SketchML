package org.dma.sketchml.ml

import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.streaming.api.scala.StreamExecutionEnvironment
import org.dma.sketchml.ml.algorithm.{LRModel, LinearRegModel, SVMModel}
import org.dma.sketchml.ml.common.Constants
import org.dma.sketchml.ml.conf.MLConf


object SketchML extends App {
  override def main(args: Array[String]): Unit = {
    val env: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment
    //This allows getting arguments like --input hdfs:///mydata --elements 42 from the command line.
    val parameters: ParameterTool = ParameterTool.fromArgs(args)
    val mlConf = MLConf(parameters)
    env.setParallelism(mlConf.workerNum)
    val model = mlConf.algo match {
      case Constants.ML_LOGISTIC_REGRESSION => LRModel(mlConf, env)
      case Constants.ML_SUPPORT_VECTOR_MACHINE => SVMModel(mlConf, env)
      case Constants.ML_LINEAR_REGRESSION => LinearRegModel(mlConf, env)
      case _ => throw new UnknownError("Unsupported algorithm: " + mlConf.algo)
    }

    model.loadData()
    model.train()
    env.execute("SketchML on data streams")
  }
}
