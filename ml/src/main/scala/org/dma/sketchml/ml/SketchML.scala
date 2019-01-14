package org.dma.sketchml.ml


//import org.dma.sketchml.ml.algorithm._
//import org.dma.sketchml.ml.conf.MLConf
// import org.apache.flink.api.java.utils.ParameterTool

import org.dma.sketchml.ml.data.Parser
import org.dma.sketchml.ml.common.Constants
import org.apache.flink.streaming.api.scala.StreamExecutionEnvironment


object SketchML extends App {

  @transient protected implicit val sc: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment

  override def main(args: Array[String]): Unit = {
    //val parameters: ParameterTool = ParameterTool.fromArgs(args)
    //val mlConf = MLConf(parameters)

    // val model = mlConf.algo match {
    //   case Constants.ML_LOGISTIC_REGRESSION => LRModel(mlConf)
    //   case Constants.ML_SUPPORT_VECTOR_MACHINE => SVMModel(mlConf)
    //   case Constants.ML_LINEAR_REGRESSION => LinearRegModel(mlConf)
    //   case _ => throw new UnknownError("Unsupported algorithm: " + mlConf.algo)
    // }

    // model.loadData()
    // model.train() TODO: test data


     val data = Parser.loadStreamData("", Constants.FORMAT_CSV, 10, 4)



    }


}
