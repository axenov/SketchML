package org.dma.sketchml.ml

import org.apache.flink.api.java.utils.ParameterTool
import org.dma.sketchml.ml.algorithm._
import org.dma.sketchml.ml.common.Constants
import org.dma.sketchml.ml.conf.MLConf

object SketchML extends App {
  override def main(args: Array[String]): Unit = {

    //This allows getting arguments like --input hdfs:///mydata --elements 42 from the command line.
    val parameters: ParameterTool = ParameterTool.fromArgs(args)
    val mlConf = MLConf(parameters)
    val model = mlConf.algo match {
      case Constants.ML_LOGISTIC_REGRESSION => LRModel(mlConf)
      case Constants.ML_SUPPORT_VECTOR_MACHINE => SVMModel(mlConf)
      case Constants.ML_LINEAR_REGRESSION => LinearRegModel(mlConf)
      case _ => throw new UnknownError("Unsupported algorithm: " + mlConf.algo)
    }

    model.loadData()
    model.train()

    // TODO: test data
  }

}
