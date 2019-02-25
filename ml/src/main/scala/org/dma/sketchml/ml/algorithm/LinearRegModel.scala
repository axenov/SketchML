package org.dma.sketchml.ml.algorithm

import org.apache.flink.ml.math.DenseVector
import org.apache.flink.streaming.api.scala.StreamExecutionEnvironment
import org.dma.sketchml.ml.algorithm.GeneralizedLinearModel.Model.{loss, optimizer, weights}
import org.dma.sketchml.ml.common.Constants
import org.dma.sketchml.ml.conf.MLConf
import org.dma.sketchml.ml.objective.{Adam, L2SquareLoss}
import org.slf4j.{Logger, LoggerFactory}


object LinearRegModel {
  private val logger: Logger = LoggerFactory.getLogger(LinearRegModel.getClass)

  def apply(conf: MLConf, env: StreamExecutionEnvironment): LinearRegModel = new LinearRegModel(conf, env)

  def getName: String = Constants.ML_LINEAR_REGRESSION
}

class LinearRegModel(_conf: MLConf, _env: StreamExecutionEnvironment) extends GeneralizedLinearModel(_conf, _env) {
  @transient override protected val logger: Logger = LinearRegModel.logger

  override protected def initModel(): Unit = {
    weights = new DenseVector(new Array[Double](_conf.featureNum))
    optimizer = Adam(_conf)
    loss = new L2SquareLoss(_conf.l2Reg)
  }

  override def getName: String = LinearRegModel.getName

}
