package org.dma.sketchml.ml.algorithm

import org.dma.sketchml.ml.algorithm.GeneralizedLinearModel.Model._
import org.apache.flink.ml.math.DenseVector
import org.apache.flink.streaming.api.scala.StreamExecutionEnvironment
import org.dma.sketchml.ml.common.Constants
import org.dma.sketchml.ml.conf.MLConf
import org.dma.sketchml.ml.objective.{Adam, L2LogLoss}
import org.slf4j.{Logger, LoggerFactory}

object LRModel {
  private val logger: Logger = LoggerFactory.getLogger(LRModel.getClass)

  def apply(conf: MLConf, env: StreamExecutionEnvironment): LRModel = new LRModel(conf, env)

  def getName: String = Constants.ML_LOGISTIC_REGRESSION
}

class LRModel(_conf: MLConf, _env: StreamExecutionEnvironment) extends GeneralizedLinearModel(_conf, _env) {
  @transient override protected val logger: Logger = LRModel.logger

  override protected def initModel(): Unit = {
    weights = new DenseVector(new Array[Double](_conf.featureNum))
    optimizer = Adam(_conf)
    loss = new L2LogLoss(_conf.l2Reg)
  }

  override def getName: String = LRModel.getName
}
